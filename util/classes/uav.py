"""
UAV data structure for tracking UAV state over time
Author: Khalil El Kaaki & Joe Abi Samra
Date: November 2025
"""

import torch
import numpy as np


class UAV:
    """
    UAV structure with position, velocity, CPU frequency, energy, and max velocity.
    
    All vectors (position, velocity, energy) have size equal to the time indices array.
    
    Attributes:
        position: Tensor of shape (2, T) - (x, y) positions at each time index
        velocity: Tensor of shape (2, T) - (vx, vy) velocities at each time index
        cpu_frequency: Scalar - UAV CPU frequency in Hz
        energy: Tensor of shape (T,) - Energy level at each time index
        max_velocity: Scalar - Maximum velocity in m/s
        time_indices: Tensor of shape (T,) - Global time indices array
    """
    
    def __init__(self, time_indices, cpu_frequency, max_velocity, initial_position=None, 
                 initial_velocity=None, initial_energy=None, device='cuda'):
        """
        Initialize UAV structure.
        
        Args:
            time_indices: Array or tensor of time indices (T,)
            cpu_frequency: CPU frequency in Hz (scalar)
            max_velocity: Maximum velocity in m/s (scalar)
            initial_position: Initial (x, y) position (optional, defaults to zeros)
            initial_velocity: Initial (vx, vy) velocity (optional, defaults to zeros)
            initial_energy: Initial energy level (optional, defaults to max energy)
            device: 'cuda' or 'cpu'
        """
        # Convert time_indices to tensor
        if not isinstance(time_indices, torch.Tensor):
            self.time_indices = torch.tensor(time_indices, dtype=torch.float32, device=device)
        else:
            self.time_indices = time_indices.to(device)
        
        self.T = len(self.time_indices)
        self.device = device
        
        # Initialize position vector (2, T) - (x, y) at each time step
        if initial_position is not None:
            if isinstance(initial_position, (list, tuple, np.ndarray)):
                initial_position = torch.tensor(initial_position, dtype=torch.float32, device=device)
            self.position = initial_position.unsqueeze(1).expand(2, self.T).clone()
        else:
            self.position = torch.zeros(2, self.T, device=device)
        
        # Initialize velocity vector (2, T) - (vx, vy) at each time step
        if initial_velocity is not None:
            if isinstance(initial_velocity, (list, tuple, np.ndarray)):
                initial_velocity = torch.tensor(initial_velocity, dtype=torch.float32, device=device)
            self.velocity = initial_velocity.unsqueeze(1).expand(2, self.T).clone()
        else:
            self.velocity = torch.zeros(2, self.T, device=device)
        
        # CPU frequency (scalar)
        self.cpu_frequency = float(cpu_frequency)
        
        # Max velocity (scalar)
        self.max_velocity = float(max_velocity)
        
        # Initialize energy vector (T,) - energy at each time step
        if initial_energy is not None:
            if isinstance(initial_energy, (int, float)):
                self.energy = torch.full((self.T,), initial_energy, dtype=torch.float32, device=device)
            else:
                self.energy = torch.tensor(initial_energy, dtype=torch.float32, device=device)
        else:
            # Default: assume max energy (e.g., 1.0 normalized or battery capacity)
            self.energy = torch.ones(self.T, device=device)
    
    def update_position(self, t_idx, position):
        """
        Update position at a specific time index.
        
        Args:
            t_idx: Time index (0 to T-1)
            position: New (x, y) position as tuple, list, or tensor
        """
        if isinstance(position, (list, tuple, np.ndarray)):
            position = torch.tensor(position, dtype=torch.float32, device=self.device)
        self.position[:, t_idx] = position
    
    def update_velocity(self, t_idx, velocity):
        """
        Update velocity at a specific time index.
        
        Args:
            t_idx: Time index (0 to T-1)
            velocity: New (vx, vy) velocity as tuple, list, or tensor
        """
        if isinstance(velocity, (list, tuple, np.ndarray)):
            velocity = torch.tensor(velocity, dtype=torch.float32, device=self.device)
        self.velocity[:, t_idx] = velocity
    
    def update_energy(self, t_idx, energy):
        """
        Update energy at a specific time index.
        
        Args:
            t_idx: Time index (0 to T-1)
            energy: New energy level (scalar)
        """
        self.energy[t_idx] = float(energy)
    
    def get_position(self, t_idx):
        """
        Get position at a specific time index.
        
        Args:
            t_idx: Time index (0 to T-1)
        
        Returns:
            position: Tensor of shape (2,) - (x, y) position
        """
        return self.position[:, t_idx]
    
    def get_velocity(self, t_idx):
        """
        Get velocity at a specific time index.
        
        Args:
            t_idx: Time index (0 to T-1)
        
        Returns:
            velocity: Tensor of shape (2,) - (vx, vy) velocity
        """
        return self.velocity[:, t_idx]
    
    def get_energy(self, t_idx):
        """
        Get energy at a specific time index.
        
        Args:
            t_idx: Time index (0 to T-1)
        
        Returns:
            energy: Scalar energy level
        """
        return self.energy[t_idx].item()
    
    def get_speed(self, t_idx):
        """
        Get speed (velocity magnitude) at a specific time index.
        
        Args:
            t_idx: Time index (0 to T-1)
        
        Returns:
            speed: Scalar speed in m/s
        """
        v = self.velocity[:, t_idx]
        return torch.sqrt(v[0]**2 + v[1]**2).item()
    
    def is_velocity_valid(self, t_idx):
        """
        Check if velocity at time index is within max velocity constraint.
        
        Args:
            t_idx: Time index (0 to T-1)
        
        Returns:
            valid: Boolean - True if speed <= max_velocity
        """
        speed = self.get_speed(t_idx)
        return speed <= self.max_velocity
    
    def compute_processing_time(self, task, time_idx):
        """Process task on this UAV's CPU at time index."""
        return task.compute_processing_time(self.cpu_frequency)
    
    def compute_flight_energy(self, P_hover=100.0, k_drag=0.5):
        """
        Compute total flight energy consumption using power-based model.
        
        Power model: P_flight(t) = P_hover + k_drag × ||v(t)||²
        Energy: E = Σ P_flight(t) × Δt
        
        Args:
            P_hover: Hovering power in Watts (default: 100W for typical quadcopter)
            k_drag: Drag coefficient in W·s²/m² (default: 0.5)
        
        Returns:
            total_energy: Total flight energy in Joules
        """
        if self.T < 2:
            return 0.0
        
        # Compute time steps Δt
        delta_t = torch.diff(self.time_indices)  # (T-1,)
        
        # Compute speed squared ||v(t)||² at each time step
        speeds_squared = self.velocity[0, :-1]**2 + self.velocity[1, :-1]**2  # (T-1,)
        
        # Power at each time step: P_hover + k × ||v||²
        power = P_hover + k_drag * speeds_squared  # Watts, (T-1,)
        
        # Energy = Σ Power × Δt
        energy = torch.sum(power * delta_t)  # Joules
        
        return energy.item()
    
    def compute_computation_energy(self, task_cycles_per_timestep, kappa=1e-27):
        """
        Compute total computation energy consumption.
        
        Power model: P_compute(t) = κ × f_CPU² × C(t)
        where C(t) is the CPU cycles executed at time t
        
        Args:
            task_cycles_per_timestep: Tensor (T,) - CPU cycles processed at each time step
            kappa: Effective switched capacitance (default: 1e-27, typical for processors)
        
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
        # C(t) is cycles per second, so C(t) × Δt gives total cycles in interval
        f_squared = self.cpu_frequency ** 2
        power = kappa * f_squared * task_cycles_per_timestep[:-1]  # Watts, (T-1,)
        
        # Energy = Σ Power × Δt
        energy = torch.sum(power * delta_t)  # Joules
        
        return energy.item()
    
    def compute_total_energy(self, task_cycles_per_timestep, P_hover=100.0, k_drag=0.5, kappa=1e-27):
        """
        Compute total UAV energy consumption (flight + computation).
        
        E_total = E_flight + E_compute
        
        Args:
            task_cycles_per_timestep: Tensor (T,) - CPU cycles processed at each time step
            P_hover: Hovering power in Watts
            k_drag: Drag coefficient in W·s²/m²
            kappa: Effective switched capacitance
        
        Returns:
            total_energy: Total energy in Joules
            flight_energy: Flight energy component in Joules
            compute_energy: Computation energy component in Joules
        """
        flight_energy = self.compute_flight_energy(P_hover, k_drag)
        compute_energy = self.compute_computation_energy(task_cycles_per_timestep, kappa)
        total_energy = flight_energy + compute_energy
        
        return total_energy, flight_energy, compute_energy
    
    def update_energy_consumption(self, task_cycles_per_timestep, P_hover=100.0, k_drag=0.5, kappa=1e-27):
        """
        Update the energy vector based on flight and computation consumption.
        
        Decreases energy at each time step based on power consumption.
        Assumes energy[0] is the initial battery capacity.
        
        Args:
            task_cycles_per_timestep: Tensor (T,) - CPU cycles processed at each time step
            P_hover: Hovering power in Watts
            k_drag: Drag coefficient in W·s²/m²
            kappa: Effective switched capacitance
        """
        if self.T < 2:
            return
        
        # Compute time steps
        delta_t = torch.diff(self.time_indices)  # (T-1,)
        
        # Flight power at each time step
        speeds_squared = self.velocity[0, :-1]**2 + self.velocity[1, :-1]**2
        power_flight = P_hover + k_drag * speeds_squared  # (T-1,)
        
        # Computation power at each time step
        if not isinstance(task_cycles_per_timestep, torch.Tensor):
            task_cycles_per_timestep = torch.tensor(task_cycles_per_timestep, 
                                                    dtype=torch.float32, 
                                                    device=self.device)
        f_squared = self.cpu_frequency ** 2
        power_compute = kappa * f_squared * task_cycles_per_timestep[:-1]  # (T-1,)
        
        # Total power
        power_total = power_flight + power_compute  # (T-1,)
        
        # Energy consumed at each interval
        energy_consumed = power_total * delta_t  # (T-1,)
        
        # Update energy vector (cumulative decrease)
        for t in range(1, self.T):
            self.energy[t] = self.energy[t-1] - energy_consumed[t-1]
            # Clamp to zero (battery can't go negative)
            self.energy[t] = torch.maximum(self.energy[t], torch.tensor(0.0, device=self.device))
    
    def __repr__(self):
        """String representation of UAV."""
        return (f"UAV(T={self.T}, cpu_freq={self.cpu_frequency/1e9:.1f} GHz, "
                f"max_vel={self.max_velocity} m/s, device={self.device})")
    
    def to(self, device):
        """
        Move UAV tensors to a different device.
        
        Args:
            device: 'cuda' or 'cpu'
        
        Returns:
            self: UAV instance with tensors on new device
        """
        self.device = device
        self.time_indices = self.time_indices.to(device)
        self.position = self.position.to(device)
        self.velocity = self.velocity.to(device)
        self.energy = self.energy.to(device)
        return self
