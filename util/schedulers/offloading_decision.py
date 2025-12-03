"""
Offloading decision algorithms for edge computing with TDMA scheduling
Author: Khalil El Kaaki & Joe Abi Samra
Date: November 2025
"""

import torch
import numpy as np
from typing import Optional, Tuple, Dict, Any

from ..common import compute_total_offload_time, p_received
from ..common.channel_reliability import channel_success_probability
from .tdma_scheduler import TDMAQueue, estimate_tdma_wait_time


def compute_task_completion(task, serving_time: float, channel_prob: float, 
                           P_min: float = 0.95) -> int:
    """
    Determine if a task completes successfully.
    
    A task is completed if:
    1. Serving time <= deadline (slack time)
    2. Channel reliability >= minimum threshold
    
    Args:
        task: Task object with slack attribute
        serving_time: Total time to serve task (seconds)
        channel_prob: Channel success probability
        P_min: Minimum required reliability (default: 0.95)
    
    Returns:
        completion: 1 if task completed successfully, 0 otherwise
    """
    meets_deadline = (serving_time <= task.slack)
    reliable_channel = (channel_prob >= P_min)
    
    return int(meets_deadline and reliable_channel)


class OffloadingParams:
    """
    System parameters for offloading decisions.
    
    Attributes:
        BW_total: Total bandwidth in Hz
        H_M: Mobile device height in meters
        H: UAV/BS height in meters
        F: Carrier frequency in Hz
        P_T: Transmit power in dBm
        noise_var: Noise variance in dBm
        snr_thresh: SNR threshold for successful reception (linear)
        P_min: Minimum required channel reliability
        current_time: Current simulation time in seconds
    """
    
    def __init__(self, BW_total: float, H_M: float, H: float, F: float,
                 P_T: float, noise_var: float, snr_thresh: float = 10.0,
                 P_min: float = 0.95, current_time: float = 0.0):
        self.BW_total = BW_total
        self.H_M = H_M
        self.H = H
        self.F = F
        self.P_T = P_T
        self.noise_var = noise_var
        self.snr_thresh = snr_thresh
        self.P_min = P_min
        self.current_time = current_time


def make_offloading_decision(task, iot_device, uav, bs, time_idx: int,
                            tdma_queue: TDMAQueue, params: OffloadingParams,
                            device='cuda', precomputed_bs_dist: Optional[float] = None,
                            precomputed_uav_dist: Optional[float] = None,
                            verbose: bool = False) -> Optional[Tuple[str, float, float]]:
    """
    Choose offloading target to minimize serving time with TDMA awareness.
    
    Evaluates three options:
    1. Local processing (no offloading)
    2. Base station offloading (uses TDMA)
    3. UAV offloading (uses TDMA)
    
    Filters by:
    - Channel reliability >= P_min
    - Total time (including TDMA wait) <= deadline
    
    Chooses option with minimum total serving time.
    
    Args:
        task: Task object
        iot_device: IoTDevice object
        uav: UAV object
        bs: BaseStation object
        time_idx: Current time index
        tdma_queue: TDMAQueue for transmission scheduling
        params: OffloadingParams object with system parameters
        device: 'cuda' or 'cpu'
        verbose: If True, print detailed decision breakdown
    
    Returns:
        decision: Tuple of (target, total_time, channel_prob) or None if infeasible
                 target is 'local', 'bs', or 'uav'
    """
    options = []
    
    if verbose:
        print(f"\n         [DECISION] Task {task.task_id}: slack={task.slack:.3f}s, size={task.length_bits/1e6:.2f}Mb")
    
    # Estimate TDMA waiting time for offloaded options
    t_wait = estimate_tdma_wait_time(tdma_queue, params.current_time)
    
    if verbose:
        print(f"         • TDMA wait time estimate: {t_wait:.3f}s")
    
    # ==================== Option 1: Local Processing ====================
    t_local = iot_device.compute_local_processing_time(task)
    p_local = 1.0  # Always reliable (no wireless channel)
    total_local = t_local  # No TDMA waiting for local
    
    if verbose:
        print(f"         • LOCAL: t_comp={t_local:.3f}s, total={total_local:.3f}s, p={p_local:.3f}, feasible={total_local <= task.slack}")
    
    if total_local <= task.slack:
        options.append(('local', total_local, p_local))
    
    # ==================== Option 2: Base Station Offloading ====================
    # Compute offload time (uplink + processing) using full BW_total
    t_bs = compute_total_offload_time(
        task, iot_device, bs, time_idx,
        H_M=params.H_M, H=bs.height, F=params.F,
        P_T=params.P_T, P_N=params.noise_var,
        bandwidth=params.BW_total  # Full bandwidth with TDMA
    )
    
    # Compute channel reliability
    iot_pos = iot_device.position.unsqueeze(1)  # (2, 1)
    bs_pos = bs.position.unsqueeze(1)  # (2, 1)
    p_r_bs = p_received(iot_pos, bs_pos, params.H_M, bs.height, params.F, 
                        params.P_T, device=device)
    
    # Compute channel success probability using dBm values
    # Convert SNR threshold from linear to dB for the function
    snr_thresh_dB = 10 * np.log10(params.snr_thresh)
    p_bs = channel_success_probability(p_r_bs, params.noise_var, snr_thresh_dB, device=device).item()
    
    if verbose:
        iot_pos_np = iot_device.position.cpu().numpy()
        bs_pos_np = bs.position.cpu().numpy()
        dist_bs = np.sqrt(np.sum((iot_pos_np - bs_pos_np)**2))
        print(f"         • BS: dist={dist_bs:.1f}m, t_offload={t_bs:.3f}s, total={t_bs + t_wait:.3f}s, p={p_bs:.3f}, feasible={p_bs >= params.P_min and (t_bs + t_wait) <= task.slack}")
    
    # Check reliability and deadline
    if p_bs >= params.P_min:
        total_bs = t_bs + t_wait  # Includes TDMA waiting
        if total_bs <= task.slack:
            options.append(('bs', total_bs, p_bs))
    
    # ==================== Option 3: UAV Offloading ====================
    # Only evaluate UAV option if UAV exists (not None for no_uav baseline)
    if uav is not None:
        # Compute offload time using full BW_total
        t_uav = compute_total_offload_time(
            task, iot_device, uav, time_idx,
            H_M=params.H_M, H=params.H, F=params.F,
            P_T=params.P_T, P_N=params.noise_var,
            bandwidth=params.BW_total  # Full bandwidth with TDMA
        )
        
        # Compute channel reliability
        uav_pos = uav.get_position(time_idx).unsqueeze(1)  # (2, 1)
        p_r_uav = p_received(iot_pos, uav_pos, params.H_M, params.H, params.F,
                             params.P_T, device=device)
        
        # Compute channel success probability using dBm values
        p_uav = channel_success_probability(p_r_uav, params.noise_var, snr_thresh_dB, device=device).item()
        
        if verbose:
            iot_pos_np = iot_device.position.cpu().numpy()
            uav_pos_np = uav.get_position(time_idx).cpu().numpy()
            dist_uav = np.sqrt(np.sum((iot_pos_np - uav_pos_np)**2))
            print(f"         • UAV: dist={dist_uav:.1f}m, t_offload={t_uav:.3f}s, total={t_uav + t_wait:.3f}s, p={p_uav:.3f}, feasible={p_uav >= params.P_min and (t_uav + t_wait) <= task.slack}")
        
        # Check reliability and deadline
        if p_uav >= params.P_min:
            total_uav = t_uav + t_wait  # Includes TDMA waiting
            if total_uav <= task.slack:
                options.append(('uav', total_uav, p_uav))
    
    # ==================== Choose Best Option ====================
    if not options:
        if verbose:
            print(f"         → DECISION: NONE (no feasible options)")
        return None  # No feasible option (all violate deadline or reliability)
    
    # Choose option with minimum total serving time
    decision = min(options, key=lambda x: x[1])
    
    if verbose:
        print(f"         → DECISION: {decision[0].upper()} (time={decision[1]:.3f}s, p={decision[2]:.3f})")
    
    return decision


def greedy_offloading_batch(tasks: list, iot_devices: list, uav, bs,
                           time_indices: torch.Tensor, params: OffloadingParams,
                           device='cuda') -> Dict[str, Any]:
    """
    Greedy offloading algorithm for a batch of tasks with TDMA scheduling.
    
    Processes tasks sequentially, making greedy decisions to minimize
    individual serving times while respecting TDMA constraints.
    
    OPTIMIZED: Pre-computes all IoT-to-BS and IoT-to-UAV distances for faster lookup.
    
    Args:
        tasks: List of Task objects
        iot_devices: List of IoTDevice objects (one per task)
        uav: UAV object
        bs: BaseStation object
        time_indices: Tensor of time indices
        params: OffloadingParams object
        device: 'cuda' or 'cpu'
    
    Returns:
        results: Dictionary containing:
            - decisions: List of (target, total_time, channel_prob) tuples
            - completions: List of completion indicators (0 or 1)
            - total_completed: Total number of completed tasks
            - tdma_queue_final: Final state of TDMA queue
    """
    tdma_queue = TDMAQueue(device=device)
    decisions = []
    completions = []
    
    # PRE-COMPUTE: All IoT positions and distances to BS (OPTIMIZATION)
    if iot_devices:
        iot_positions_tensor = torch.stack([iot.position for iot in iot_devices], dim=0)  # (N, 2)
        bs_pos_tensor = bs.position.unsqueeze(0)  # (1, 2)
        
        # Pre-compute all IoT-to-BS distances: (N,)
        precomputed_bs_distances = torch.cdist(iot_positions_tensor, bs_pos_tensor).squeeze(1)  # (N,)
        
        # Pre-compute all IoT-to-UAV distances for all timesteps: (N, T) - if UAV exists
        # UAV position is (2, T), transpose to (T, 2) for cdist
        if uav is not None:
            precomputed_uav_distances = torch.cdist(iot_positions_tensor, uav.position.T)  # (N, T)
        else:
            precomputed_uav_distances = None  # No UAV baseline
    else:
        precomputed_bs_distances = None
        precomputed_uav_distances = None
    
    for i, (task, iot_device) in enumerate(zip(tasks, iot_devices)):
        # Update current time
        time_idx = min(i, len(time_indices) - 1)
        params.current_time = time_indices[time_idx].item()
        
        # Clean up completed transmissions
        tdma_queue.remove_completed(params.current_time)
        
        # Get pre-computed distances for this task
        bs_dist = precomputed_bs_distances[i].item() if precomputed_bs_distances is not None else None
        uav_dist = precomputed_uav_distances[i, time_idx].item() if precomputed_uav_distances is not None else None
        
        # Make offloading decision with pre-computed distances
        # Verbose output for first 3 tasks only
        verbose = (i < 3)
        decision = make_offloading_decision(
            task, iot_device, uav, bs, time_idx,
            tdma_queue, params, device=device,
            precomputed_bs_dist=bs_dist,
            precomputed_uav_dist=uav_dist,
            verbose=verbose
        )
        
        if decision is None:
            # Task cannot be completed
            decisions.append(('none', float('inf'), 0.0))
            completions.append(0)
        else:
            target, total_time, channel_prob = decision
            decisions.append(decision)
            
            # Compute completion indicator
            completion = compute_task_completion(task, total_time, channel_prob, params.P_min)
            completions.append(completion)
            
            # Schedule in TDMA queue if offloaded
            if target != 'local':
                # Estimate uplink duration
                if target == 'bs':
                    t_offload = compute_total_offload_time(
                        task, iot_device, bs, time_idx,
                        params.H_M, bs.height, params.F, params.P_T,
                        params.noise_var, params.BW_total
                    )
                else:  # 'uav'
                    t_offload = compute_total_offload_time(
                        task, iot_device, uav, time_idx,
                        params.H_M, params.H, params.F, params.P_T,
                        params.noise_var, params.BW_total
                    )
                
                # Uplink is part of offload time; approximate as 60% for transmission
                uplink_duration = t_offload * 0.6
                
                from .tdma_scheduler import schedule_task_tdma
                schedule_task_tdma(i, target, uplink_duration, tdma_queue, params.current_time)
    
    results = {
        'decisions': decisions,
        'completions': completions,
        'total_completed': sum(completions),
        'completion_rate': sum(completions) / len(tasks) if tasks else 0.0,
        'tdma_queue_final': tdma_queue
    }
    
    return results