"""
Online UAV Trajectory Optimization with Receding Horizon
Author: Khalil El Kaaki & Joe Abi Samra
Date: November 2025

Implements Model Predictive Control (MPC) / Receding Horizon approach:
- At each time step, optimize next N positions
- Execute first position, then re-optimize
- Accounts for current tasks, deadlines, and channel conditions
"""

import torch
import numpy as np
from typing import List, Tuple, Optional
from ..common import p_received
from ..common.channel_reliability import channel_success_probability


class RecedingHorizonOptimizer:
    """
    Online trajectory optimizer using receding horizon control.
    
    Objective: Maximize expected completed tasks within planning horizon
    
    At each time step:
    1. Observe current state (position, pending tasks, deadlines)
    2. Optimize next H positions to maximize task completion probability
    3. Execute first position
    4. Repeat
    """
    
    def __init__(self, horizon: int = 10, learning_rate: float = 5.0,
                 opt_iterations: int = 20, device='cuda',
                 uav_cpu_freq: float = 5e9, uav_height: float = 50.0,
                 params=None):
        """
        Initialize receding horizon optimizer.
        
        Args:
            horizon: Planning horizon (number of future steps to optimize)
            learning_rate: Step size for gradient descent
            opt_iterations: Number of optimization iterations per step
            device: 'cuda' or 'cpu'
            uav_cpu_freq: UAV CPU frequency for computation time calculation
            uav_height: UAV altitude for 3D distance calculation
            params: OffloadingParams object for channel calculations
        """
        self.horizon = horizon
        self.learning_rate = learning_rate
        self.opt_iterations = opt_iterations
        self.device = device
        self.uav_cpu_freq = uav_cpu_freq
        self.uav_height = uav_height
        self.params = params
    
    def optimize_next_position(self, current_pos: torch.Tensor, current_time: float,
                               pending_tasks: List, task_devices: List, iot_devices: List,
                               bs, time_indices: torch.Tensor, current_idx: int,
                               max_velocity: float, dt: float) -> torch.Tensor:
        """
        Optimize next position using receding horizon.
        
        Args:
            current_pos: Current UAV position (2,)
            current_time: Current time
            pending_tasks: List of tasks not yet completed
            task_devices: List of IoTDevice objects for each pending task
            iot_devices: All IoT devices in system
            bs: BaseStation object
            time_indices: Full time index tensor (T,)
            current_idx: Current time index
            max_velocity: Maximum UAV velocity (m/s)
            dt: Time step size
        
        Returns:
            next_position: Optimized next position (2,)
        """
        if not pending_tasks:
            # No pending tasks - stay at current position
            return current_pos
        
        # Optimize trajectory over horizon
        H = min(self.horizon, len(time_indices) - current_idx)
        
        if H <= 1:
            # Last time step - simple heuristic: move toward urgent tasks
            device_positions = []
            weights = []
            
            for task, iot_device in zip(pending_tasks, task_devices):
                device_positions.append(iot_device.position)
                remaining_slack = task.slack - (current_time - task.time_generated)
                weight = 1.0 / max(remaining_slack, 0.1)
                weights.append(weight)
            
            if device_positions:
                device_positions = torch.stack(device_positions, dim=1)
                weights = torch.tensor(weights, device=self.device)
                weights = weights / weights.sum()
                target_pos = (device_positions * weights.unsqueeze(0)).sum(dim=1)
            else:
                target_pos = current_pos
            
            # Clamp to max velocity
            direction = target_pos - current_pos
            distance = torch.sqrt(torch.sum(direction ** 2))
            
            if distance > max_velocity * dt:
                next_pos = current_pos + (direction / distance) * max_velocity * dt
            else:
                next_pos = target_pos
            
            return next_pos
        
        # Initialize horizon trajectory - start with straight line toward task cluster
        horizon_positions = torch.zeros(2, H, device=self.device, requires_grad=True)
        
        # Compute initial target as weighted centroid
        device_positions = []
        weights = []
        for task, iot_device in zip(pending_tasks, task_devices):
            device_positions.append(iot_device.position)
            remaining_slack = task.slack - (current_time - task.time_generated)
            weight = 1.0 / max(remaining_slack, 0.1)
            weights.append(weight)
        
        if device_positions:
            device_positions_tensor = torch.stack(device_positions, dim=1)
            weights_tensor = torch.tensor(weights, device=self.device)
            weights_tensor = weights_tensor / weights_tensor.sum()
            target_pos = (device_positions_tensor * weights_tensor.unsqueeze(0)).sum(dim=1)
        else:
            target_pos = current_pos
        
        with torch.no_grad():
            # Linear path toward target
            for h in range(H):
                alpha = (h + 1) / H
                horizon_positions[:, h] = (1 - alpha) * current_pos + alpha * target_pos
        
        horizon_positions = horizon_positions.detach().requires_grad_(True)
        optimizer = torch.optim.Adam([horizon_positions], lr=self.learning_rate)
        
        # Optimize horizon to maximize task completion
        avg_tdma_wait = 0.1  # Estimated TDMA wait time
        
        for opt_iter in range(self.opt_iterations):
            optimizer.zero_grad()
            
            # Compute expected completions for pending tasks
            expected_completions = torch.tensor(0.0, device=self.device)
            
            for task_idx, (task, iot_device) in enumerate(zip(pending_tasks, task_devices)):
                # Determine which horizon step is closest to task generation/deadline
                task_age = current_time - task.time_generated
                remaining_slack = task.slack - task_age
                
                if remaining_slack <= 0:
                    continue  # Task already expired
                
                # Use average position over horizon for this task
                # Weight earlier positions more for urgent tasks
                urgency_weights = torch.exp(-torch.arange(H, device=self.device, dtype=torch.float32) * (1.0 / max(remaining_slack, 0.1)))
                urgency_weights = urgency_weights / urgency_weights.sum()
                
                avg_uav_pos = (horizon_positions * urgency_weights.unsqueeze(0)).sum(dim=1)  # (2,)
                
                # ============================================================
                # OPTION 1: LOCAL PROCESSING
                # ============================================================
                t_local = (task.length_bits * task.computation_density) / iot_device.cpu_frequency
                local_slack_margin = torch.tensor(remaining_slack - t_local, device=self.device)
                local_score = torch.sigmoid(5.0 * local_slack_margin) * 1.0
                
                # ============================================================
                # OPTION 2: BASE STATION OFFLOADING (Okumura-Hata)
                # ============================================================
                # Note: bs is passed through task_devices context, using device position
                iot_pos = iot_device.position.unsqueeze(1)  # (2, 1)
                
                # Get BS from params or use default position
                bs_pos_x, bs_pos_y = 200.0, 200.0  # Default BS position
                bs_pos = torch.tensor([[bs_pos_x], [bs_pos_y]], dtype=torch.float32, device=self.device)
                bs_cpu = 10e9  # Default BS CPU
                bs_height = 30.0  # Default BS height
                
                p_r_bs = p_received(iot_pos, bs_pos, self.params.H_M, bs_height, self.params.F, self.params.P_T, device=self.device)
                P_r_bs_lin = 10.0 ** (p_r_bs / 10.0)
                P_n_lin = 10.0 ** (self.params.noise_var / 10.0)
                snr_bs = P_r_bs_lin / P_n_lin
                data_rate_bs = self.params.BW_total * torch.log2(1.0 + snr_bs).squeeze()
                
                transmission_time_bs = task.length_bits / (data_rate_bs + 1e-6)
                computation_time_bs = (task.length_bits * task.computation_density) / bs_cpu
                t_bs = transmission_time_bs + computation_time_bs + avg_tdma_wait
                
                p_bs_reliable = torch.sigmoid(2.0 * (snr_bs.squeeze() - self.params.snr_thresh))
                bs_slack_margin = torch.tensor(remaining_slack, device=self.device) - t_bs
                bs_score = torch.sigmoid(5.0 * bs_slack_margin) * p_bs_reliable
                
                # ============================================================
                # OPTION 3: UAV OFFLOADING (Okumura-Hata)
                # ============================================================
                uav_pos = avg_uav_pos.unsqueeze(1)  # (2, 1)
                p_r_uav = p_received(iot_pos, uav_pos, self.params.H_M, self.params.H, self.params.F, self.params.P_T, device=self.device)
                P_r_uav_lin = 10.0 ** (p_r_uav / 10.0)
                snr_uav = P_r_uav_lin / P_n_lin
                data_rate_uav = self.params.BW_total * torch.log2(1.0 + snr_uav).squeeze()
                
                transmission_time_uav = task.length_bits / (data_rate_uav + 1e-6)
                computation_time_uav = (task.length_bits * task.computation_density) / self.uav_cpu_freq
                t_uav = transmission_time_uav + computation_time_uav + avg_tdma_wait
                
                p_uav_reliable = torch.sigmoid(2.0 * (snr_uav.squeeze() - self.params.snr_thresh))
                uav_slack_margin = torch.tensor(remaining_slack, device=self.device) - t_uav
                uav_score = torch.sigmoid(5.0 * uav_slack_margin) * p_uav_reliable
                
                # ============================================================
                # SOFT DECISION: Weighted combination
                # ============================================================
                scores = torch.stack([local_score, bs_score, uav_score])
                weights = torch.softmax(10.0 * scores, dim=0)
                expected_completion = (weights * scores).sum()
                expected_completions += expected_completion
            
            # Velocity constraints
            # Velocity constraints
            # First step from current position
            vel_0 = torch.sqrt(torch.sum((horizon_positions[:, 0] - current_pos) ** 2)) / dt
            vel_penalty_0 = 100.0 * torch.relu(vel_0 - max_velocity)
            
            # Between horizon steps
            vel_penalty = 0.0
            for h in range(H - 1):
                vel_h = torch.sqrt(torch.sum((horizon_positions[:, h+1] - horizon_positions[:, h]) ** 2)) / dt
                vel_penalty += 100.0 * torch.relu(vel_h - max_velocity)
            
            # Total loss: MAXIMIZE completions = MINIMIZE negative completions
            total_loss = -expected_completions + vel_penalty_0 + vel_penalty
            total_loss.backward()
            optimizer.step()
        
        # Return first position in optimized horizon
        next_position = horizon_positions[:, 0].detach()
        
        return next_position


def optimize_trajectory_online(iot_devices: List, tasks: List, bs,
                               time_indices: torch.Tensor, uav_cpu_frequency: float,
                               uav_max_velocity: float, uav_height: float,
                               initial_position: List[float], params,
                               device='cuda', horizon: int = 10) -> torch.Tensor:
    """
    Online trajectory optimization using receding horizon control.
    
    Simulates the UAV executing tasks and optimizing its trajectory online
    based on remaining tasks and their deadlines.
    
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
        horizon: Planning horizon steps
    
    Returns:
        trajectory: Tensor of shape (2, T) with online optimized positions
    """
    T = len(time_indices)
    dt = (time_indices[1] - time_indices[0]).item() if T > 1 else 1.0
    
    # Initialize trajectory
    trajectory = torch.zeros(2, T, device=device)
    trajectory[:, 0] = torch.tensor(initial_position, dtype=torch.float32, device=device)
    
    # Create optimizer with UAV parameters
    optimizer = RecedingHorizonOptimizer(
        horizon=horizon,
        learning_rate=2.0,
        opt_iterations=25,
        device=device,
        uav_cpu_freq=uav_cpu_frequency,
        uav_height=uav_height,
        params=params
    )
    
    # Create task-device mapping from task metadata
    # Each task knows its source device via device_id
    task_devices = []
    for task in tasks:
        if task.device_id is not None and 0 <= task.device_id < len(iot_devices):
            # Use task's recorded device ID
            device = iot_devices[task.device_id]
        else:
            # Fallback: assign to first device (should not happen with proper task generation)
            device = iot_devices[0]
        task_devices.append(device)
    
    # Online optimization loop
    pending_tasks = tasks.copy()
    pending_devices = task_devices.copy()
    
    for t in range(1, T):
        current_pos = trajectory[:, t-1]
        current_time = time_indices[t].item()
        
        # Remove completed/expired tasks
        new_pending_tasks = []
        new_pending_devices = []
        for task, device in zip(pending_tasks, pending_devices):
            deadline = task.time_generated + task.slack
            if current_time < deadline:
                new_pending_tasks.append(task)
                new_pending_devices.append(device)
        
        pending_tasks = new_pending_tasks
        pending_devices = new_pending_devices
        
        # Optimize next position
        next_pos = optimizer.optimize_next_position(
            current_pos=current_pos,
            current_time=current_time,
            pending_tasks=pending_tasks,
            task_devices=pending_devices,
            iot_devices=iot_devices,
            bs=bs,
            time_indices=time_indices,
            current_idx=t,
            max_velocity=uav_max_velocity,
            dt=dt
        )
        
        trajectory[:, t] = next_pos
    
    return trajectory
