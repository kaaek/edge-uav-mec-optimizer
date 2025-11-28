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


class RecedingHorizonOptimizer:
    """
    Online trajectory optimizer using receding horizon control.
    
    At each time step:
    1. Observe current state (position, pending tasks, deadlines)
    2. Optimize next H time steps (horizon)
    3. Execute first step
    4. Repeat
    """
    
    def __init__(self, horizon: int = 10, learning_rate: float = 5.0,
                 opt_iterations: int = 20, device='cuda'):
        """
        Initialize receding horizon optimizer.
        
        Args:
            horizon: Planning horizon (number of future steps to optimize)
            learning_rate: Step size for gradient descent
            opt_iterations: Number of optimization iterations per step
            device: 'cuda' or 'cpu'
        """
        self.horizon = horizon
        self.learning_rate = learning_rate
        self.opt_iterations = opt_iterations
        self.device = device
    
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
        
        # Compute target: weighted centroid of devices with pending tasks
        device_positions = []
        weights = []
        
        for task, iot_device in zip(pending_tasks, task_devices):
            device_positions.append(iot_device.position)
            # Weight by urgency (inverse of remaining slack)
            remaining_slack = task.slack - (current_time - task.time_generated)
            weight = 1.0 / max(remaining_slack, 0.1)  # Avoid division by zero
            weights.append(weight)
        
        if device_positions:
            device_positions = torch.stack(device_positions, dim=1)  # (2, M_pending)
            weights = torch.tensor(weights, device=self.device)
            weights = weights / weights.sum()  # Normalize
            
            # Weighted centroid
            target_pos = (device_positions * weights.unsqueeze(0)).sum(dim=1)  # (2,)
        else:
            # Fallback: move toward cluster center
            iot_positions = torch.stack([iot.position for iot in iot_devices], dim=1)
            target_pos = iot_positions.mean(dim=1)
        
        # Optimize trajectory over horizon
        H = min(self.horizon, len(time_indices) - current_idx)
        
        if H <= 1:
            # Last time step or horizon too short - move directly toward target
            direction = target_pos - current_pos
            distance = torch.sqrt(torch.sum(direction ** 2))
            
            if distance > max_velocity * dt:
                # Clamp to max velocity
                next_pos = current_pos + (direction / distance) * max_velocity * dt
            else:
                next_pos = target_pos
            
            return next_pos
        
        # Initialize horizon trajectory
        horizon_positions = torch.zeros(2, H, device=self.device, requires_grad=True)
        
        with torch.no_grad():
            # Linear path toward target
            for h in range(H):
                alpha = (h + 1) / H
                horizon_positions[:, h] = (1 - alpha) * current_pos + alpha * target_pos
        
        horizon_positions = horizon_positions.detach().requires_grad_(True)
        optimizer = torch.optim.Adam([horizon_positions], lr=self.learning_rate)
        
        # Optimize horizon
        for _ in range(self.opt_iterations):
            optimizer.zero_grad()
            
            # Loss: distance to target positions weighted by urgency
            loss = 0.0
            for h in range(H):
                pos_h = horizon_positions[:, h]
                
                # Distance to pending task devices
                for task, iot_device in zip(pending_tasks, task_devices):
                    remaining_slack = task.slack - (current_time + h * dt - task.time_generated)
                    if remaining_slack > 0:  # Task still viable
                        urgency = 1.0 / max(remaining_slack, 0.1)
                        dist = torch.sqrt(torch.sum((pos_h - iot_device.position) ** 2))
                        loss += urgency * dist
            
            # Velocity constraints
            # First step from current position
            vel_0 = torch.sqrt(torch.sum((horizon_positions[:, 0] - current_pos) ** 2)) / dt
            vel_penalty_0 = 100.0 * torch.relu(vel_0 - max_velocity)
            
            # Between horizon steps
            vel_penalty = 0.0
            for h in range(H - 1):
                vel_h = torch.sqrt(torch.sum((horizon_positions[:, h+1] - horizon_positions[:, h]) ** 2)) / dt
                vel_penalty += 100.0 * torch.relu(vel_h - max_velocity)
            
            total_loss = loss + vel_penalty_0 + vel_penalty
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
    
    # Create optimizer
    optimizer = RecedingHorizonOptimizer(
        horizon=horizon,
        learning_rate=5.0,
        opt_iterations=20,
        device=device
    )
    
    # Create task-device mapping
    # Simple approach: assume tasks distributed across devices
    # (In real scenario, you'd have this from task generation)
    task_devices = []
    for task in tasks:
        # Assign task to closest device (simplified)
        min_dist = float('inf')
        closest_device = iot_devices[0]
        for iot in iot_devices:
            dist = np.random.rand()  # Simplified - in reality use task metadata
            if dist < min_dist:
                min_dist = dist
                closest_device = iot
        task_devices.append(closest_device)
    
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
