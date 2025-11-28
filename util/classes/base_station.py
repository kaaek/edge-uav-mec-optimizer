"""
Ground Base Station data structure for edge computing system
Author: Khalil El Kaaki & Joe Abi Samra
Date: November 2025
"""

import torch
import numpy as np


class BaseStation:
    """
    Ground Base Station structure with static position, CPU frequency, and energy tracking.
    
    Unlike UAVs, base stations have fixed positions and unlimited power (grid-connected),
    but track energy consumption for cost analysis.
    
    Attributes:
        position: Tensor of shape (2,) - Static (x, y) position
        cpu_frequency: Scalar - Base station CPU frequency in Hz
        energy: Tensor of shape (T,) - Energy consumption at each time index
        time_indices: Tensor of shape (T,) - Global time indices array
        height: Scalar - Antenna height in meters
    """
    
    def __init__(self, position, cpu_frequency, time_indices, height=30.0, device='cuda'):
        """
        Initialize Ground Base Station.
        
        Args:
            position: (x, y) position as tuple, list, array, or tensor
            cpu_frequency: CPU frequency in Hz (scalar)
            time_indices: Array or tensor of time indices (T,)
            height: Antenna height in meters (default: 30m for typical BS)
            device: 'cuda' or 'cpu'
        """
        self.device = device
        
        # Static position (2,) - does not change over time
        if isinstance(position, (list, tuple, np.ndarray)):
            self.position = torch.tensor(position, dtype=torch.float32, device=device)
        elif isinstance(position, torch.Tensor):
            self.position = position.to(device)
        else:
            raise ValueError(f"Invalid position type: {type(position)}")
        
        if self.position.shape != torch.Size([2]):
            raise ValueError(f"Position must be (2,) shaped, got {self.position.shape}")
        
        # CPU frequency (scalar)
        self.cpu_frequency = float(cpu_frequency)
        if self.cpu_frequency <= 0:
            raise ValueError(f"CPU frequency must be positive, got {self.cpu_frequency}")
        
        # Antenna height (scalar)
        self.height = float(height)
        
        # Convert time_indices to tensor
        if not isinstance(time_indices, torch.Tensor):
            self.time_indices = torch.tensor(time_indices, dtype=torch.float32, device=device)
        else:
            self.time_indices = time_indices.to(device)
        
        self.T = len(self.time_indices)
        
        # Initialize energy consumption vector (T,) - tracks cumulative energy used
        # Base stations are grid-connected (unlimited power), but we track consumption
        self.energy = torch.zeros(self.T, device=device)
    
    def get_position(self, t_idx=None):
        """
        Get position of the base station.
        
        Args:
            t_idx: Time index (ignored, included for API consistency with UAV)
        
        Returns:
            position: Tensor of shape (2,) - (x, y) position
        """
        return self.position
    
    def get_x(self):
        """Get x-coordinate."""
        return self.position[0].item()
    
    def get_y(self):
        """Get y-coordinate."""
        return self.position[1].item()
    
    def get_height(self):
        """Get antenna height."""
        return self.height
    
    def compute_processing_time(self, task, time_idx=None):
        """
        Compute processing time for a task on this base station's CPU.
        
        Args:
            task: Task object with required_cycles
            time_idx: Time index (ignored, included for API consistency)
        
        Returns:
            processing_time: Time in seconds to process task
        """
        return task.compute_processing_time(self.cpu_frequency)
    
    def compute_computation_energy(self, task_cycles_per_timestep, kappa=1e-27):
        """
        Compute total computation energy consumption.
        
        Power model: P_compute(t) = κ × f_CPU² × C(t)
        where C(t) is the CPU cycles executed at time t
        
        Args:
            task_cycles_per_timestep: Tensor (T,) - CPU cycles processed at each time step
            kappa: Effective switched capacitance (default: 1e-27)
        
        Returns:
            total_energy: Total computation energy in Joules
        """
        if self.T < 2:
            return 0.0
        
        # Ensure input is tensor
        if not isinstance(task_cycles_per_timestep, torch.Tensor):
            task_cycles_per_timestep = torch.tensor(task_cycles_per_timestep, 
                                                    dtype=torch.float32, 
                                                    device=self.device)
        
        # Compute time steps Δt
        delta_t = torch.diff(self.time_indices)  # (T-1,)
        
        # Power at each time step: κ × f² × C(t)
        f_squared = self.cpu_frequency ** 2
        power = kappa * f_squared * task_cycles_per_timestep[:-1]  # Watts, (T-1,)
        
        # Energy = Σ Power × Δt
        energy = torch.sum(power * delta_t)  # Joules
        
        return energy.item()
    
    def update_energy_consumption(self, task_cycles_per_timestep, kappa=1e-27):
        """
        Update the energy consumption vector based on computation.
        
        Unlike UAVs, base stations don't have flight energy.
        Energy is cumulative (grid provides unlimited power).
        
        Args:
            task_cycles_per_timestep: Tensor (T,) - CPU cycles processed at each time step
            kappa: Effective switched capacitance
        """
        if self.T < 2:
            return
        
        # Compute time steps
        delta_t = torch.diff(self.time_indices)  # (T-1,)
        
        # Computation power at each time step
        if not isinstance(task_cycles_per_timestep, torch.Tensor):
            task_cycles_per_timestep = torch.tensor(task_cycles_per_timestep, 
                                                    dtype=torch.float32, 
                                                    device=self.device)
        
        f_squared = self.cpu_frequency ** 2
        power_compute = kappa * f_squared * task_cycles_per_timestep[:-1]  # (T-1,)
        
        # Energy consumed at each interval
        energy_consumed = power_compute * delta_t  # (T-1,)
        
        # Update energy vector (cumulative increase - tracking total consumption)
        self.energy[0] = 0.0
        for t in range(1, self.T):
            self.energy[t] = self.energy[t-1] + energy_consumed[t-1]
    
    def get_total_energy_consumed(self):
        """
        Get total energy consumed by the base station.
        
        Returns:
            total_energy: Total energy consumed in Joules
        """
        return self.energy[-1].item() if self.T > 0 else 0.0
    
    def distance_to(self, other_position):
        """
        Compute Euclidean distance to another position (horizontal distance).
        
        Args:
            other_position: (x, y) position as tuple, list, array, or tensor
        
        Returns:
            distance: Horizontal Euclidean distance in meters
        """
        if isinstance(other_position, (list, tuple, np.ndarray)):
            other_position = torch.tensor(other_position, dtype=torch.float32, device=self.device)
        elif isinstance(other_position, torch.Tensor):
            other_position = other_position.to(self.device)
        
        diff = self.position - other_position
        distance = torch.sqrt(torch.sum(diff ** 2))
        return distance.item()
    
    def distance_3d_to(self, other_position, other_height):
        """
        Compute 3D Euclidean distance to another position.
        
        Args:
            other_position: (x, y) position as tuple, list, array, or tensor
            other_height: Height of the other object in meters
        
        Returns:
            distance: 3D Euclidean distance in meters
        """
        horizontal_dist = self.distance_to(other_position)
        height_diff = abs(self.height - other_height)
        distance_3d = np.sqrt(horizontal_dist**2 + height_diff**2)
        return distance_3d
    
    def __repr__(self):
        """String representation of Base Station."""
        pos_str = f"({self.position[0].item():.1f}, {self.position[1].item():.1f})"
        return (f"BaseStation(pos={pos_str}, height={self.height:.1f}m, "
                f"cpu_freq={self.cpu_frequency/1e9:.1f} GHz, "
                f"T={self.T}, device={self.device})")
    
    def to(self, device):
        """
        Move Base Station tensors to a different device.
        
        Args:
            device: 'cuda' or 'cpu'
        
        Returns:
            self: BaseStation instance with tensors on new device
        """
        self.device = device
        self.position = self.position.to(device)
        self.time_indices = self.time_indices.to(device)
        self.energy = self.energy.to(device)
        return self
