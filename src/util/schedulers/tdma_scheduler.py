"""
TDMA (Time Division Multiple Access) scheduler for edge computing offloading
Author: Khalil El Kaaki & Joe Abi Samra
Date: November 2025
"""

import torch
from typing import List, Tuple, Optional


class TDMAQueue:
    """
    TDMA transmission queue manager.
    
    Ensures only one device transmits at a time by maintaining a queue
    of scheduled transmissions with start and end times.
    
    Attributes:
        queue: List of (task_id, start_time, end_time) tuples
        device: 'cuda' or 'cpu'
    """
    
    def __init__(self, device='cuda'):
        """
        Initialize empty TDMA queue.
        
        Args:
            device: 'cuda' or 'cpu'
        """
        self.queue = []  # List of (task_id, start_time, end_time)
        self.device = device
    
    def add_task(self, task_id: int, start_time: float, duration: float):
        """
        Add a task transmission to the TDMA queue.
        
        Args:
            task_id: Unique identifier for the task
            start_time: Time when transmission starts (seconds)
            duration: Duration of transmission (seconds)
        """
        end_time = start_time + duration
        self.queue.append((task_id, start_time, end_time))
        
        # Sort by start time to maintain chronological order
        self.queue.sort(key=lambda x: x[1])
    
    def get_next_available_slot(self, current_time: float) -> float:
        """
        Find the next available TDMA time slot.
        
        Args:
            current_time: Current time in seconds
        
        Returns:
            next_slot: Time when next transmission can start
        """
        if not self.queue:
            return current_time
        
        # Find the latest end time of all queued transmissions
        latest_end = max(end_time for _, _, end_time in self.queue)
        
        # Next slot is either now (if channel free) or after all queued tasks
        return max(current_time, latest_end)
    
    def remove_completed(self, current_time: float):
        """
        Remove completed transmissions from the queue.
        
        Args:
            current_time: Current time in seconds
        """
        # Keep only tasks that haven't finished yet
        self.queue = [(tid, start, end) for tid, start, end in self.queue 
                      if end > current_time]
    
    def get_queue_length(self) -> int:
        """Get number of tasks currently in queue."""
        return len(self.queue)
    
    def get_total_waiting_time(self, current_time: float) -> float:
        """
        Get total waiting time for all queued transmissions.
        
        Args:
            current_time: Current time in seconds
        
        Returns:
            total_wait: Sum of remaining transmission times
        """
        total = 0.0
        for _, start, end in self.queue:
            remaining = end - current_time
            if remaining > 0:
                total += remaining
        return total
    
    def is_channel_busy(self, current_time: float) -> bool:
        """
        Check if channel is currently busy.
        
        Args:
            current_time: Current time in seconds
        
        Returns:
            busy: True if a transmission is ongoing
        """
        for _, start, end in self.queue:
            if start <= current_time < end:
                return True
        return False
    
    def clear(self):
        """Clear all tasks from the queue."""
        self.queue = []
    
    def __repr__(self):
        """String representation of TDMA queue."""
        return f"TDMAQueue(tasks={len(self.queue)}, device={self.device})"
    
    def __len__(self):
        """Return number of tasks in queue."""
        return len(self.queue)


def schedule_task_tdma(task_id: int, decision: str, uplink_duration: float,
                       tdma_queue: TDMAQueue, current_time: float) -> Tuple[float, float]:
    """
    Schedule a task in the TDMA queue if offloaded.
    
    Args:
        task_id: Unique task identifier
        decision: Offloading decision ('local', 'bs', or 'uav')
        uplink_duration: Duration of uplink transmission (seconds)
        tdma_queue: TDMAQueue object
        current_time: Current time in seconds
    
    Returns:
        start_time: When transmission starts
        end_time: When transmission ends
    """
    if decision == 'local':
        # No TDMA needed for local processing
        return current_time, current_time
    
    # Get next available TDMA slot
    start_time = tdma_queue.get_next_available_slot(current_time)
    
    # Add to queue
    tdma_queue.add_task(task_id, start_time, uplink_duration)
    
    end_time = start_time + uplink_duration
    
    return start_time, end_time


def estimate_tdma_wait_time(tdma_queue: TDMAQueue, current_time: float) -> float:
    """
    Estimate waiting time until TDMA slot becomes available.
    
    This is used by the offloading decision algorithm to account for
    queuing delay when choosing between local and offloaded execution.
    
    Args:
        tdma_queue: TDMAQueue object
        current_time: Current time in seconds
    
    Returns:
        wait_time: Estimated waiting time in seconds
    """
    if not tdma_queue.queue:
        return 0.0
    
    # Sum remaining transmission times of all tasks in queue
    total_wait = 0.0
    for _, start, end in tdma_queue.queue:
        remaining = end - current_time
        if remaining > 0:
            total_wait += remaining
    
    return total_wait
