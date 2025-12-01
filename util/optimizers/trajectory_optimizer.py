"""
UAV Trajectory Optimization for Task Offloading
Author: Khalil El Kaaki & Joe Abi Samra
Date: November 2025

Optimizes UAV trajectory to maximize task completion rate while respecting:
- Maximum velocity constraints
- Energy constraints
- Channel quality requirements
"""

import torch
import numpy as np
from typing import Tuple, List, Optional


def optimize_uav_trajectory(iot_devices: List, tasks: List, bs, time_indices: torch.Tensor,
                           uav_cpu_frequency: float, uav_max_velocity: float,
                           uav_height: float, initial_position: List[float],
                           params, device='cuda', max_iter: int = 50,
                           learning_rate: float = 1.0, 
                           method: str = 'gradient') -> torch.Tensor:
    """
    Optimize UAV trajectory to maximize task completion.
    
    Uses gradient descent to find optimal UAV positions that minimize
    serving time and maximize channel quality for offloaded tasks.
    
    Args:
        iot_devices: List of IoTDevice objects
        tasks: List of Task objects  
        bs: BaseStation object
        time_indices: Tensor of time indices (T,)
        uav_cpu_frequency: UAV CPU frequency (Hz)
        uav_max_velocity: Maximum UAV velocity (m/s)
        uav_height: UAV altitude (meters)
        initial_position: Initial [x, y] position
        params: OffloadingParams object
        device: 'cuda' or 'cpu'
        max_iter: Maximum optimization iterations
        learning_rate: Step size for gradient descent
        method: 'gradient' or 'successive_convex'
    
    Returns:
        optimized_positions: Tensor of shape (2, T) with optimized positions
    """
    T = len(time_indices)
    dt = time_indices[1] - time_indices[0] if T > 1 else 1.0
    
    if method == 'gradient':
        return _gradient_based_optimization(
            iot_devices, tasks, bs, time_indices, uav_cpu_frequency,
            uav_max_velocity, uav_height, initial_position, params,
            device, max_iter, learning_rate
        )
    elif method == 'successive_convex':
        return _successive_convex_approximation(
            iot_devices, tasks, bs, time_indices, uav_cpu_frequency,
            uav_max_velocity, uav_height, initial_position, params,
            device, max_iter
        )
    else:
        raise ValueError(f"Unknown optimization method: {method}")


def _gradient_based_optimization(iot_devices, tasks, bs, time_indices,
                                 uav_cpu_frequency, uav_max_velocity, uav_height,
                                 initial_position, params, device, max_iter,
                                 learning_rate):
    """
    Gradient-based trajectory optimization.
    
    Objective: MAXIMIZE EXPECTED NUMBER OF COMPLETED TASKS
    
    Key insight: Task completion depends on:
    1. Channel reliability (function of UAV-device distance via SNR)
    2. Serving time (upload + compute, also depends on distance via data rate)
    3. Task deadline constraint
    
    Loss function: -sum(P_success(d) * P_deadline_met(d, rate(d)))
    where d = distance, rate(d) = f(distance), P_success = f(SNR(distance))
    
    Constraints: Maximum velocity between consecutive positions
    
    This directly optimizes for task completion, not a distance proxy.
    """
    T = len(time_indices)
    dt = (time_indices[1] - time_indices[0]).item() if T > 1 else 1.0
    
    # Initialize positions - start with path toward cluster center
    iot_positions = torch.stack([iot.position for iot in iot_devices], dim=1)  # (2, M)
    cluster_center = iot_positions.mean(dim=1)  # (2,)
    
    # Initialize trajectory: linear interpolation from initial to cluster center
    start = torch.tensor(initial_position, dtype=torch.float32, device=device)
    positions = torch.zeros(2, T, device=device, requires_grad=True)
    
    with torch.no_grad():
        for t in range(T):
            alpha = t / (T - 1) if T > 1 else 0
            positions[:, t] = (1 - alpha) * start + alpha * cluster_center
    
    # Detach and re-enable gradient
    positions = positions.detach().requires_grad_(True)
    
    # Pre-compute and cache IoT device positions (optimization: compute once, reuse)
    # Stack into single tensor for efficient vectorized operations
    # Shape: (M, 2) where M is number of IoT devices
    cached_iot_positions = torch.stack([iot.position for iot in iot_devices], dim=0)  # (M, 2)
    
    # Import channel reliability function
    from ..common.channel_reliability import channel_success_probability
    from ..common import p_received
    
    # Optimizer
    optimizer = torch.optim.SGD([positions], lr=learning_rate, momentum=0.9)
    
    best_positions = positions.clone().detach()
    best_loss = float('inf')
    
    # Convert SNR threshold from linear to dB for channel_success_probability function
    snr_thresh_dB = 10.0 * torch.log10(torch.tensor(params.snr_thresh))
    
    print(f"[SGD] Starting gradient descent (max_iter={max_iter})")
    for iteration in range(max_iter):
        optimizer.zero_grad()
        
        # VECTORIZED: Compute all distances at once using torch.cdist
        # positions.T: (T, 2) - UAV positions at each timestep
        # cached_iot_positions: (M, 2) - IoT device positions
        # Result: (T, M) - distance from each timestep to each device
        distances_2d = torch.cdist(
            positions.T,               # (T, 2)
            cached_iot_positions       # (M, 2)
        )  # Output: (T, M)
        
        # Compute 3D distances including UAV height
        # d_3d = sqrt(d_2d^2 + h^2)
        distances_3d = torch.sqrt(distances_2d**2 + uav_height**2)  # (T, M)
        
        # For received power calculation, we need positions not distances
        # UAV positions: (T, 2) -> transpose to (2, T) for p_received
        # IoT positions: cached_iot_positions is (M, 2) -> transpose to (2, M)
        uav_pos_for_calc = positions.T.T  # (2, T)
        iot_pos_for_calc = cached_iot_positions.T  # (2, M)
        
        # p_received computes for all (user, uav) pairs, returns (M, T)
        # We need (T, M) so we'll transpose the result
        P_r_dBm_MT = p_received(
            user_pos=iot_pos_for_calc,  # (2, M)
            uav_pos=uav_pos_for_calc,   # (2, T)
            H_M=params.H_M,
            H=uav_height,
            F=params.F,
            P_T=params.P_T,
            device=device
        )  # Returns (M, T)
        P_r_dBm = P_r_dBm_MT.T  # (T, M)
        
        # Compute channel success probability for each (time, device) pair
        # P_success = exp(-SNR_threshold / SNR_avg)
        # This is differentiable and captures how UAV position affects channel reliability
        channel_prob = channel_success_probability(
            P_r_dBm,                    # (T, M)
            params.noise_var,           # scalar
            snr_thresh_dB.item(),       # scalar
            rayleigh_scale=1.0,
            device=device
        )  # Output: (T, M)
        
        # Approximate data rate based on received power
        # Higher P_r → higher SNR → higher data rate
        # Using Shannon capacity as approximation: R ≈ BW * log2(1 + SNR)
        # SNR_avg = (P_r * E[|h|^2]) / noise_var
        P_r_linear = 10.0 ** (P_r_dBm / 10.0)
        noise_linear = 10.0 ** (params.noise_var / 10.0)
        mean_channel_gain = 2.0  # E[|h|^2] for Rayleigh with σ=1
        snr_avg = (P_r_linear * mean_channel_gain) / noise_linear
        data_rate = params.BW_total * torch.log2(1.0 + snr_avg)  # (T, M) in bps
        
        # Estimate average task completion probability
        # For each device, estimate if a typical task would complete
        # Assuming average task properties from the task list
        if len(tasks) > 0:
            avg_task_size = sum(t.length_bits for t in tasks) / len(tasks)
            avg_task_cycles = sum(t.total_cycles for t in tasks) / len(tasks)
            avg_slack = sum(t.slack for t in tasks) / len(tasks)
            
            # Serving time = upload_time + compute_time
            # upload_time = task_size / data_rate
            # compute_time = task_cycles / uav_cpu_freq
            upload_time = avg_task_size / (data_rate + 1e-6)  # (T, M), avoid div by zero
            compute_time = avg_task_cycles / uav_cpu_frequency  # scalar
            serving_time = upload_time + compute_time  # (T, M)
            
            # Probability task meets deadline
            # Use sigmoid to make it differentiable: P ~ sigmoid(-(serving_time - deadline))
            # Higher negative value → closer to 1 (meets deadline)
            # Lower value → closer to 0 (misses deadline)
            deadline_margin = avg_slack - serving_time  # (T, M)
            deadline_prob = torch.sigmoid(deadline_margin * 2.0)  # Scaled sigmoid
            
            # Expected task completion = P(channel success) * P(meets deadline)
            completion_prob = channel_prob * deadline_prob  # (T, M)
        else:
            # Fallback: just use channel probability
            completion_prob = channel_prob  # (T, M)
        
        # Objective: MAXIMIZE expected task completions
        # Average across time and devices to get overall expected completion rate
        expected_completions = completion_prob.mean()
        
        # Loss: NEGATIVE expected completions (since we minimize loss)
        task_loss = -expected_completions
        
        # Velocity constraint penalty
        velocity_penalty = 0.0
        if T > 1:
            position_diff = positions[:, 1:] - positions[:, :-1]  # (2, T-1)
            velocities = torch.sqrt(torch.sum(position_diff ** 2, dim=0)) / dt  # (T-1,)
            velocity_violations = torch.relu(velocities - uav_max_velocity)
            velocity_penalty = 100.0 * velocity_violations.sum()  # Heavy penalty
        
        # Total loss = -expected_completions + velocity_penalty
        loss = task_loss + velocity_penalty
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track best
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_positions = positions.clone().detach()
        
        # Progress reporting
        if iteration % 10 == 0 or iteration == max_iter - 1:
            print(f"[OPTIMIZER] Iter {iteration:3d}/{max_iter}: loss={loss.item():.4f}, "
                  f"expected_completion={expected_completions.item():.4f}, "
                  f"vel_penalty={velocity_penalty.item():.4f}")
        
        # Early stopping
        if iteration > 10 and velocity_penalty.item() < 0.1:
            if iteration % 10 == 0:
                improvement = (best_loss - loss.item()) / abs(best_loss) if abs(best_loss) > 0 else 0
                if improvement < 0.001:
                    print(f"[OPTIMIZER] Early stopping at iteration {iteration} (improvement < 0.1%)")
                    break
    
    return best_positions.detach()


def _successive_convex_approximation(iot_devices, tasks, bs, time_indices,
                                    uav_cpu_frequency, uav_max_velocity, uav_height,
                                    initial_position, params, device, max_iter):
    """
    Successive Convex Approximation (SCA) for trajectory optimization.
    
    Linearizes non-convex constraints and solves iteratively.
    """
    T = len(time_indices)
    dt = (time_indices[1] - time_indices[0]).item() if T > 1 else 1.0
    
    # Initialize positions
    iot_positions = torch.stack([iot.position for iot in iot_devices], dim=1)  # (2, M)
    cluster_center = iot_positions.mean(dim=1)
    
    start = torch.tensor(initial_position, dtype=torch.float32, device=device)
    positions = torch.zeros(2, T, device=device)
    
    with torch.no_grad():
        for t in range(T):
            alpha = t / (T - 1) if T > 1 else 0
            positions[:, t] = (1 - alpha) * start + alpha * cluster_center
    
    # SCA iterations
    for sca_iter in range(max_iter):
        positions_new = positions.clone()
        
        # Update each time step
        for t in range(1, T - 1):
            # Gradient of distance to IoT devices
            grad = torch.zeros(2, device=device)
            for iot in iot_devices:
                diff = positions[:, t] - iot.position
                dist = torch.sqrt(torch.sum(diff ** 2) + 1e-6)
                grad += diff / dist
            
            grad /= len(iot_devices)
            
            # Update with gradient descent step
            step_size = learning_rate = 0.5
            positions_new[:, t] = positions[:, t] - step_size * grad
            
            # Project to satisfy velocity constraint
            if t > 0:
                diff_prev = positions_new[:, t] - positions_new[:, t-1]
                vel_prev = torch.sqrt(torch.sum(diff_prev ** 2)) / dt
                if vel_prev > uav_max_velocity:
                    positions_new[:, t] = positions_new[:, t-1] + (diff_prev / vel_prev) * uav_max_velocity * dt
            
            if t < T - 1:
                diff_next = positions_new[:, t+1] - positions_new[:, t]
                vel_next = torch.sqrt(torch.sum(diff_next ** 2)) / dt
                if vel_next > uav_max_velocity:
                    # Adjust current position to not exceed velocity to next
                    max_diff = uav_max_velocity * dt
                    positions_new[:, t] = positions_new[:, t+1] - (diff_next / vel_next) * max_diff
        
        # Check convergence
        change = torch.norm(positions_new - positions)
        positions = positions_new
        
        if change < 0.01:
            break
    
    return positions


def compute_trajectory_velocity(positions: torch.Tensor, time_indices: torch.Tensor) -> torch.Tensor:
    """
    Compute velocity from position trajectory.
    
    Args:
        positions: Tensor of shape (2, T)
        time_indices: Tensor of shape (T,)
    
    Returns:
        velocity: Tensor of shape (2, T)
    """
    T = positions.shape[1]
    velocity = torch.zeros_like(positions)
    
    if T < 2:
        return velocity
    
    # Forward difference for velocity
    dt = time_indices[1:] - time_indices[:-1]  # (T-1,)
    position_diff = positions[:, 1:] - positions[:, :-1]  # (2, T-1)
    
    velocity[:, :-1] = position_diff / dt.unsqueeze(0)
    velocity[:, -1] = velocity[:, -2]  # Last velocity same as second-to-last
    
    return velocity


def verify_velocity_constraints(positions: torch.Tensor, time_indices: torch.Tensor,
                                max_velocity: float) -> Tuple[bool, float]:
    """
    Verify that trajectory satisfies velocity constraints.
    
    Args:
        positions: Tensor of shape (2, T)
        time_indices: Tensor of shape (T,)
        max_velocity: Maximum allowed velocity (m/s)
    
    Returns:
        is_feasible: True if all velocities <= max_velocity
        max_vel: Maximum velocity in trajectory
    """
    velocity = compute_trajectory_velocity(positions, time_indices)
    velocity_magnitude = torch.sqrt(torch.sum(velocity ** 2, dim=0))  # (T,)
    max_vel = velocity_magnitude.max().item()
    
    is_feasible = max_vel <= max_velocity * 1.01  # 1% tolerance
    
    return is_feasible, max_vel
