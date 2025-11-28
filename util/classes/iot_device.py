"""
IoT Device data structure for edge computing system
Author: Khalil El Kaaki & Joe Abi Samra
Date: November 2025
"""

import torch
import numpy as np


class IoTDevice:
    """
    IoT Device structure with static position, local CPU frequency, and Poisson task arrival.
    
    Attributes:
        position: Tensor of shape (2,) - Static (x, y) position
        cpu_frequency: Scalar - Local CPU frequency in Hz
        lambda_rate: Scalar - Average task arrival rate (tasks per second) for Poisson process
        device_id: Optional device identifier
    """
    
    def __init__(self, position, cpu_frequency, lambda_rate, device_id=None, device='cuda'):
        """
        Initialize IoT Device.
        
        Args:
            position: (x, y) position as tuple, list, array, or tensor
            cpu_frequency: Local CPU frequency in Hz (scalar)
            lambda_rate: Average task arrival rate λ for Poisson distribution (tasks/second)
            device_id: Optional identifier for the device (int or string)
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
        
        # Local CPU frequency (scalar)
        self.cpu_frequency = float(cpu_frequency)
        
        # Task arrival rate λ for Poisson process (scalar)
        self.lambda_rate = float(lambda_rate)
        if self.lambda_rate < 0:
            raise ValueError(f"Lambda rate must be non-negative, got {self.lambda_rate}")
        
        # Optional device ID
        self.device_id = device_id
    
    def get_position(self):
        """
        Get the static position of the IoT device.
        
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
    
    def generate_task_arrivals(self, time_duration):
        """
        Generate number of task arrivals using Poisson distribution.
        
        Args:
            time_duration: Time duration in seconds
        
        Returns:
            num_tasks: Number of tasks arrived (sampled from Poisson(λ * time_duration))
        """
        expected_arrivals = self.lambda_rate * time_duration
        if self.device == 'cuda':
            # Use torch for GPU compatibility
            num_tasks = torch.poisson(torch.tensor(expected_arrivals, device=self.device))
            return int(num_tasks.item())
        else:
            # Use numpy for CPU
            return np.random.poisson(expected_arrivals)
    
    def generate_task_arrival_times(self, time_duration):
        """
        Generate task arrival times using Poisson process.
        
        Args:
            time_duration: Time duration in seconds
        
        Returns:
            arrival_times: Array of arrival times within [0, time_duration]
        """
        num_tasks = self.generate_task_arrivals(time_duration)
        if num_tasks == 0:
            return torch.tensor([], dtype=torch.float32, device=self.device)
        
        # Generate uniform random arrival times and sort
        arrival_times = torch.rand(num_tasks, device=self.device) * time_duration
        arrival_times = torch.sort(arrival_times)[0]
        return arrival_times
    
    def compute_local_processing_time(self, task):
        """Use this device's CPU to process task."""
        return task.compute_processing_time(self.cpu_frequency)
    
    def can_meet_deadline(self, current_time, task_complexity, deadline):
        """
        Check if the IoT device can meet the deadline for a given task.
        
        Args:
            task_complexity: Computational complexity in CPU cycles
            deadline: Deadline time in seconds

        Returns:
            can_meet: True if the device can meet the deadline
        """
        processing_time = self.compute_local_processing_time(task_complexity)
        return processing_time <= deadline - current_time

    def distance_to(self, other_position):
        """
        Compute Euclidean distance to another position.
        
        Args:
            other_position: (x, y) position as tuple, list, array, or tensor
        
        Returns:
            distance: Euclidean distance in meters
        """
        if isinstance(other_position, (list, tuple, np.ndarray)):
            other_position = torch.tensor(other_position, dtype=torch.float32, device=self.device)
        elif isinstance(other_position, torch.Tensor):
            other_position = other_position.to(self.device)
        
        diff = self.position - other_position
        distance = torch.sqrt(torch.sum(diff ** 2))
        return distance.item()
    
    def __repr__(self):
        """String representation of IoT Device."""
        pos_str = f"({self.position[0].item():.1f}, {self.position[1].item():.1f})"
        id_str = f"ID={self.device_id}, " if self.device_id is not None else ""
        return (f"IoTDevice({id_str}pos={pos_str}, "
                f"cpu_freq={self.cpu_frequency/1e9:.2f} GHz, "
                f"λ={self.lambda_rate:.2f} tasks/s)")
    
    def to(self, device):
        """
        Move IoT Device tensors to a different device.
        
        Args:
            device: 'cuda' or 'cpu'
        
        Returns:
            self: IoTDevice instance with tensors on new device
        """
        self.device = device
        self.position = self.position.to(device)
        return self
