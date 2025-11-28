"""
Task data structure for edge computing system
Author: Khalil El Kaaki & Joe Abi Samra
Date: November 2025
"""

import torch
import numpy as np


class Task:
    """
    Task structure for offloading decisions in edge computing.
    
    Attributes:
        length_bits: Task data size in bits
        computation_density: Computational complexity in cycles per bit
        required_cycles: Total required CPU cycles (length_bits × computation_density)
        time_generated: Time when task was generated (seconds)
        deadline: Task deadline (time_generated + slack)
        task_id: Optional task identifier
    """
    
    def __init__(self, length_bits, computation_density, time_generated, slack, task_id=None):
        """
        Initialize Task.
        
        Args:
            length_bits: Task data size in bits (scalar)
            computation_density: Computational complexity in cycles per bit (scalar)
            time_generated: Time when task was generated in seconds (scalar)
            slack: Deadline slack time in seconds (scalar)
            task_id: Optional identifier for the task (int or string)
        """
        # Task data size (bits)
        self.length_bits = float(length_bits)
        if self.length_bits <= 0:
            raise ValueError(f"Task length must be positive, got {self.length_bits}")
        
        # Computation density (cycles/bit)
        self.computation_density = float(computation_density)
        if self.computation_density <= 0:
            raise ValueError(f"Computation density must be positive, got {self.computation_density}")
        
        # Total required cycles (cycles)
        self.required_cycles = self.length_bits * self.computation_density
        
        # Time generated (seconds)
        self.time_generated = float(time_generated)
        if self.time_generated < 0:
            raise ValueError(f"Time generated must be non-negative, got {self.time_generated}")
        
        # Deadline slack (seconds)
        self.slack = float(slack)
        if self.slack <= 0:
            raise ValueError(f"Slack must be positive, got {self.slack}")
        
        # Deadline time (seconds)
        self.deadline = self.time_generated + self.slack
        
        # Optional task ID
        self.task_id = task_id
    
    def get_remaining_time(self, current_time):
        """
        Get remaining time until deadline.
        
        Args:
            current_time: Current time in seconds
        
        Returns:
            remaining_time: Time remaining until deadline (can be negative if overdue)
        """
        return self.deadline - current_time
    
    def is_deadline_met(self, completion_time):
        """
        Check if task completed before deadline.
        
        Args:
            completion_time: Time when task was completed in seconds
        
        Returns:
            met: True if completion_time <= deadline
        """
        return completion_time <= self.deadline
    
    def compute_processing_time(self, cpu_frequency):
        """
        Compute processing time for this task on a given CPU.
        
        Args:
            cpu_frequency: CPU frequency in Hz
        
        Returns:
            processing_time: Time in seconds to process task
        """
        if cpu_frequency <= 0:
            raise ValueError(f"CPU frequency must be positive, got {cpu_frequency}")
        return self.required_cycles / cpu_frequency
    
    def compute_transmission_time(self, data_rate):
        """
        Compute transmission time for this task's data.
        
        Args:
            data_rate: Data rate in bits per second
        
        Returns:
            transmission_time: Time in seconds to transmit task data
        """
        if data_rate <= 0:
            raise ValueError(f"Data rate must be positive, got {data_rate}")
        return self.length_bits / data_rate
    
    def compute_offload_time(self, uplink_rate, cpu_frequency, downlink_rate):
        """
        Compute total time for offloading task (uplink + compute + downlink).
        
        Assumes symmetric uplink/downlink for result transmission.
        
        Args:
            uplink_rate: Uplink data rate in bits per second
            cpu_frequency: Remote CPU frequency in Hz
            downlink_rate: Downlink data rate in bits per second
        
        Returns:
            total_time: Total offloading time in seconds
        """
        t_uplink = self.compute_transmission_time(uplink_rate)
        t_compute = self.compute_processing_time(cpu_frequency)
        t_downlink = self.compute_transmission_time(downlink_rate)
        return t_uplink + t_compute + t_downlink
    
    def get_urgency(self, current_time):
        """
        Get task urgency (inverse of remaining time).
        
        Higher urgency means task is closer to deadline.
        
        Args:
            current_time: Current time in seconds
        
        Returns:
            urgency: Urgency score (higher = more urgent)
        """
        remaining = self.get_remaining_time(current_time)
        if remaining <= 0:
            return float('inf')  # Overdue tasks have infinite urgency
        return 1.0 / remaining
    
    def get_slack_ratio(self, current_time):
        """
        Get ratio of remaining time to original slack.
        
        Args:
            current_time: Current time in seconds
        
        Returns:
            slack_ratio: Fraction of slack remaining (0 to 1, or negative if overdue)
        """
        remaining = self.get_remaining_time(current_time)
        return remaining / self.slack
    
    def __repr__(self):
        """String representation of Task."""
        id_str = f"ID={self.task_id}, " if self.task_id is not None else ""
        return (f"Task({id_str}"
                f"size={self.length_bits/1e6:.2f} Mb, "
                f"density={self.computation_density/1e3:.1f} kcycles/bit, "
                f"total={self.required_cycles/1e9:.2f} Gcycles, "
                f"gen={self.time_generated:.2f}s, "
                f"deadline={self.deadline:.2f}s)")
    
    def to_dict(self):
        """
        Convert task to dictionary for serialization.
        
        Returns:
            task_dict: Dictionary with task attributes
        """
        return {
            'task_id': self.task_id,
            'length_bits': self.length_bits,
            'computation_density': self.computation_density,
            'required_cycles': self.required_cycles,
            'time_generated': self.time_generated,
            'slack': self.slack,
            'deadline': self.deadline
        }
    
    @classmethod
    def from_dict(cls, task_dict):
        """
        Create Task from dictionary.
        
        Args:
            task_dict: Dictionary with task attributes
        
        Returns:
            task: Task instance
        """
        return cls(
            length_bits=task_dict['length_bits'],
            computation_density=task_dict['computation_density'],
            time_generated=task_dict['time_generated'],
            slack=task_dict['slack'],
            task_id=task_dict.get('task_id')
        )
