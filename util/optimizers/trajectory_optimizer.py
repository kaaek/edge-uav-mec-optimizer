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
    
    Objective: Minimize average distance to IoT devices (proxy for channel quality)
    Constraints: Maximum velocity between consecutive positions
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
    
    # Optimizer
    optimizer = torch.optim.Adam([positions], lr=learning_rate)
    
    best_positions = positions.clone().detach()
    best_loss = float('inf')
    
    print(f"[OPTIMIZER] Starting gradient descent (max_iter={max_iter})")
    for iteration in range(max_iter):
        optimizer.zero_grad()
        
        # Compute loss: average distance to IoT devices (want to minimize)
        # This encourages UAV to stay close to IoT cluster
        distances = torch.zeros(T, device=device)
        for iot in iot_devices:
            iot_pos = iot.position.unsqueeze(1).expand(2, T)  # (2, T)
            dist = torch.sqrt(torch.sum((positions - iot_pos) ** 2, dim=0))  # (T,)
            distances += dist
        
        avg_distance = distances.mean() / len(iot_devices)
        
        # Velocity constraint penalty
        velocity_penalty = 0.0
        if T > 1:
            position_diff = positions[:, 1:] - positions[:, :-1]  # (2, T-1)
            velocities = torch.sqrt(torch.sum(position_diff ** 2, dim=0)) / dt  # (T-1,)
            velocity_violations = torch.relu(velocities - uav_max_velocity)
            velocity_penalty = 10.0 * velocity_violations.sum()  # Heavy penalty
        
        # Total loss
        loss = avg_distance + velocity_penalty
        
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
                  f"avg_dist={avg_distance.item():.2f}, vel_penalty={velocity_penalty.item():.4f}")
        
        # Early stopping
        if iteration > 10 and velocity_penalty.item() < 0.1:
            if iteration % 10 == 0:
                improvement = (best_loss - loss.item()) / best_loss if best_loss > 0 else 0
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
