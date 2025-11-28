"""
Common utility functions for UAV network optimization
Author: Khalil El Kaaki & Joe Abi Samra
"""
import torch
import numpy as np

# Import channel reliability functions
from .channel_reliability import (
    rayleigh_channel_gain,
    compute_instantaneous_snr,
    channel_failure_probability,
    channel_success_probability,
    dual_channel_reliability,
    multi_channel_reliability,
    expected_channel_capacity_rayleigh,
    sample_rayleigh_channel_realization
)

def p_received(user_pos, uav_pos, H_M, H, F, P_T, device='cuda'):
    """
    Calculates the received power using the Okumura-Hata model.
    
    CRITICAL: Ensures proper unit conversions:
    - F input is in Hz, converted to MHz internally
    - Distances calculated in meters, converted to km for path loss formula
    
    Args:
        user_pos: Tensor of shape (2, M) - user positions (x, y) in METERS
        uav_pos: Tensor of shape (2, N) - UAV positions (x, y) in METERS
        H_M: Mobile height in METERS
        H: UAV height in METERS
        F: Frequency in HZ (will be converted to MHz internally)
        P_T: Transmit power in dBm
        device: 'cuda' or 'cpu'
    
    Returns:
        p_r: Tensor of shape (M, N) - received power in dBm
    """
    # Convert to tensors if needed and move to device
    if not isinstance(user_pos, torch.Tensor):
        user_pos = torch.tensor(user_pos, dtype=torch.float32, device=device)
    else:
        user_pos = user_pos.to(device)
    
    if not isinstance(uav_pos, torch.Tensor):
        uav_pos = torch.tensor(uav_pos, dtype=torch.float32, device=device)
    else:
        uav_pos = uav_pos.to(device)
    
    # Extract coordinates
    x_m = user_pos[0, :].unsqueeze(1)  # (M, 1)
    y_m = user_pos[1, :].unsqueeze(1)  # (M, 1)
    x_n = uav_pos[0, :].unsqueeze(0)   # (1, N)
    y_n = uav_pos[1, :].unsqueeze(0)   # (1, N)
    
    # Calculate 3D distances in METERS
    d_m = torch.sqrt((x_m - x_n)**2 + (y_m - y_n)**2 + H**2)  # (M, N) in meters
    
    # Unit conversions for Okumura-Hata model, model requires frequency in MHz and distance in km
    F_MHz = F / 1e6  # Hz -> MHz
    d_km = d_m / 1000.0  # meters -> km
    
    # Assertion for sanity check
    assert F_MHz > 0 and F_MHz < 1e6, f"Frequency {F_MHz} MHz out of reasonable range. Make sure F is in Hz."
    
    # Okumura-Hata path loss model (vectorized)
    # Reference: Okumura-Hata model for suburban/rural environments
    F_MHz_tensor = torch.tensor(F_MHz, device=device, dtype=torch.float32)
    H_tensor = torch.tensor(H, device=device, dtype=torch.float32)
    
    C_h = 0.8 + (1.1 * torch.log10(F_MHz_tensor) - 0.7) * H_M - 1.56 * torch.log10(F_MHz_tensor)
    L_u = 69.55 + 26.16 * torch.log10(F_MHz_tensor) - 13.82 * torch.log10(H_tensor) - C_h + \
          (44.9 - 6.55 * torch.log10(H_tensor)) * torch.log10(d_km)
    L = L_u - 4.78 * (torch.log10(F_MHz_tensor))**2 + 18.33 * torch.log10(F_MHz_tensor) - 40.94
    
    # Received power in dBm
    p_r = P_T - L  # (M, N)
    
    return p_r


def association(p_r):
    """
    Associates each user with the UAV providing maximum received power.
    One-hot encoding: each row has exactly one '1'.
    
    Args:
        p_r: Tensor of shape (M, N) - received power matrix
    
    Returns:
        a: Tensor of shape (M, N) - binary association matrix
    """
    device = p_r.device
    M, N = p_r.shape
    a = torch.zeros(M, N, device=device)
    
    # Find index of max value in each row
    index_max = torch.argmax(p_r, dim=1)  # (M,)
    
    # Set corresponding positions to 1
    a[torch.arange(M, device=device), index_max] = 1
    
    return a


def se(P_R, P_N, ASSOCIATION_MATRIX):
    """
    Calculates the spectral efficiency (bps/Hz).
    
    Args:
        P_R: Tensor of shape (M, N) - received power in dBm
        P_N: Scalar - noise power in dBm
        ASSOCIATION_MATRIX: Tensor of shape (M, N) - binary association matrix
    
    Returns:
        SE: Tensor of shape (M, N) - spectral efficiency in bps/Hz
    """
    device = P_R.device
    
    # Convert from dBm to linear (mW)
    P_r_lin = 10.0 ** (P_R / 10.0)  # (M, N)
    P_n_lin = 10.0 ** (P_N / 10.0)  # scalar
    
    # Calculate SNR
    SNR = P_r_lin / P_n_lin  # (M, N)
    
    # Shannon capacity formula
    SE = torch.log2(1 + SNR) * ASSOCIATION_MATRIX  # (M, N) bps/Hz, zeroed for non-associated links
    
    return SE


def bitrate(P_R, P_N, BW, ASSOCIATION_MATRIX):
    """
    Calculates bitrate using Shannon capacity.
    
    Args:
        P_R: Tensor of shape (M, N) - received power in dBm
        P_N: Scalar - noise power in dBm
        BW: Scalar or Tensor - bandwidth in Hz (can be per-user vector of shape (M,))
        ASSOCIATION_MATRIX: Tensor of shape (M, N) - binary association matrix
    
    Returns:
        br: Tensor of shape (M, N) - bitrate in bps
    """
    SE_matrix = se(P_R, P_N, ASSOCIATION_MATRIX)  # bps/Hz, (M, N)
    
    # Handle bandwidth: if it's a vector per user, reshape appropriately
    if isinstance(BW, torch.Tensor) and BW.dim() == 1:
        BW = BW.unsqueeze(1)  # (M, 1)
    
    br = BW * SE_matrix  # bps
    
    return br


def qos_constraint(br, Rmin):
    """
    QoS constraint for optimization: ensures all users achieve minimum rate.
    
    Args:
        br: Tensor of shape (M,) - bitrates per user
        Rmin: Scalar - minimum required rate
    
    Returns:
        c: Tensor of inequality constraints (c <= 0)
        ceq: Empty list (no equality constraints)
    """
    c = Rmin - br  # Should be <= 0, i.e., br >= Rmin
    ceq = []
    
    return c, ceq


def init_bandwidth(user_pos, uav_pos, assoc, Rmin, BW_total, H_M, H, F, P_T, P_N, device='cuda'):
    """
    Initialize bandwidth allocation based on spectral efficiency.
    
    Args:
        user_pos: Tensor (2, M) - user positions
        uav_pos: Tensor (2, N) - UAV positions
        assoc: Tensor (M, N) - association matrix
        Rmin: Scalar - minimum rate requirement
        BW_total: Scalar - total available bandwidth
        H_M, H, F, P_T, P_N: Physical parameters
        device: 'cuda' or 'cpu'
    
    Returns:
        bw_needed: Tensor (M,) - bandwidth needed per user
        feasible: Boolean - whether allocation is feasible
    """
    M = user_pos.shape[1]
    
    # Calculate received power and spectral efficiency
    Pr = p_received(user_pos, uav_pos, H_M, H, F, P_T, device=device)
    P_n_lin = 10.0 ** (P_N / 10.0)
    SE_matrix = torch.log2(1 + 10.0**(Pr/10.0) / P_n_lin)  # bits/s/Hz, (M, N)
    
    # Find associated UAV for each user
    user_indices = torch.arange(M, device=device)
    uav_indices = torch.argmax(assoc, dim=1)
    
    # Extract SE values for associated pairs
    se_values = SE_matrix[user_indices, uav_indices]
    
    # Calculate bandwidth requirements
    bw_req = Rmin / se_values
    
    # Check for invalid values
    bw_req = torch.where(torch.isfinite(bw_req) & (bw_req >= 0), bw_req, torch.zeros_like(bw_req))
    
    feasible = (torch.sum(bw_req) <= BW_total)
    
    return bw_req, feasible


def throughput(OFFLOADING, DATA, bandwidth, USER_POS, uav_pos, H_M, H, F, P_T, P_N, C_M, F_N, F_M, device='cuda'):
    """
    Calculate throughput considering offloading decisions.
    
    Args:
        OFFLOADING: Offloading decision (0 or 1)
        DATA: Data size
        bandwidth: Bandwidth allocation
        USER_POS: User positions
        uav_pos: UAV positions
        H_M, H, F, P_T, P_N: Physical parameters
        C_M: Computational complexity
        F_N: Edge server frequency
        F_M: Mobile device frequency
        device: 'cuda' or 'cpu'
    
    Returns:
        th: Throughput
    """
    P_R = p_received(USER_POS, uav_pos, H_M, H, F, P_T, device=device)
    ASSOCIATION_MATRIX = association(P_R)
    
    SE_val = se(P_R, P_N, ASSOCIATION_MATRIX)
    T_ul_m = DATA / (bandwidth * SE_val)
    T_comp_m = C_M / F_N
    T_local_m = C_M / F_M
    OFFLOADING_COMPLEMENT = 1 - OFFLOADING
    
    th = (DATA * OFFLOADING * 1/(2 * T_ul_m + T_comp_m)) + \
         (DATA * OFFLOADING_COMPLEMENT * 1/T_local_m)
    
    return th


def compute_mec_throughput(R_m, o_m, D_m, C_m, f_UAV, f_user, device='cuda'):
    """
    Computes MEC throughput per user using fractional offloading model.
    
    Fractional offloading formula:
    - T_ul(o) = o_m * D_m / R_m (uplink transmission time)
    - T_comp(o) = o_m * C_m / f_UAV (computation time at UAV)
    - T_dl(o) = o_m * D_m / R_m (downlink transmission time, symmetric)
    - T_local = C_m / f_user (local computation time)
    - Th_m(o) = o_m * D_m / (T_ul + T_comp + T_dl) + (1 - o_m) * D_m / T_local
    
    Args:
        R_m: Tensor (M,) - per-user uplink/downlink data rate (bps)
        o_m: Tensor (M,) - offloading fraction [0, 1]
        D_m: Scalar or Tensor (M,) - task data size (bits)
        C_m: Scalar or Tensor (M,) - computational complexity (CPU cycles)
        f_UAV: Scalar - UAV CPU frequency (Hz)
        f_user: Scalar - user device CPU frequency (Hz)
        device: 'cuda' or 'cpu'
    
    Returns:
        Th_m: Tensor (M,) - MEC throughput per user (bps)
    """
    # Ensure tensors
    if not isinstance(R_m, torch.Tensor):
        R_m = torch.tensor(R_m, dtype=torch.float32, device=device)
    if not isinstance(o_m, torch.Tensor):
        o_m = torch.tensor(o_m, dtype=torch.float32, device=device)
    
    # Clamp R_m to avoid division by zero
    R_m = torch.maximum(R_m, torch.tensor(1e-6, device=device))
    
    # Offloading times (in seconds)
    T_ul = o_m * D_m / R_m  # Uplink time
    T_comp = o_m * C_m / f_UAV  # Computation time at UAV
    T_dl = o_m * D_m / R_m  # Downlink time (symmetric UL/DL)
    T_offload_total = T_ul + T_comp + T_dl
    
    # Local computation time (in seconds)
    T_local = C_m / f_user
    
    # MEC throughput (bps)
    # Offloaded portion: o_m * D_m / T_offload_total
    # Local portion: (1 - o_m) * D_m / T_local
    Th_offload = torch.where(o_m > 0, o_m * D_m / torch.maximum(T_offload_total, torch.tensor(1e-9, device=device)), torch.zeros_like(o_m))
    Th_local = (1 - o_m) * D_m / T_local
    Th_m = Th_offload + Th_local
    
    # Clamp to avoid numerical issues
    Th_m = torch.maximum(Th_m, torch.tensor(1e-9, device=device))
    
    return Th_m

def compute_total_offload_time(task, iot_device, uav, time_idx, H_M, H, F, P_T, P_N, bandwidth):
    """
    Compute total time for offloading a task to UAV with TDMA.
    
    With TDMA, the task gets full bandwidth during its time slot.
    Total time = uplink transmission + UAV processing
    
    Args:
        task: Task object with length_bits and required_cycles
        iot_device: IoTDevice object with position
        uav: UAV object with position and cpu_frequency (or BaseStation)
        time_idx: Time index for UAV position lookup
        H_M: Mobile height in meters
        H: UAV height in meters
        F: Frequency in Hz
        P_T: Transmit power in dBm
        P_N: Noise variance in dBm (for Rayleigh fading model)
        bandwidth: Full bandwidth in Hz (BW_total in TDMA)
    
    Returns:
        total_time: Total offloading time in seconds (uplink + compute)
    """
    device = iot_device.device
    
    # Get IoT device and UAV/BS positions
    iot_pos = iot_device.position.unsqueeze(1)  # (2, 1)
    
    # Handle both UAV and BaseStation
    if hasattr(uav, 'get_position'):
        # UAV with time-varying position
        server_pos = uav.get_position(time_idx).unsqueeze(1)  # (2, 1)
        server_cpu = uav.cpu_frequency
    else:
        # BaseStation with static position
        server_pos = uav.position.unsqueeze(1)  # (2, 1)
        server_cpu = uav.cpu_frequency
    
    # Compute received power (M=1, N=1 case)
    p_r = p_received(iot_pos, server_pos, H_M, H, F, P_T, device=device)  # (1, 1)
    
    # Compute spectral efficiency
    P_r_lin = 10.0 ** (p_r / 10.0)
    P_n_lin = 10.0 ** (P_N / 10.0)
    SE = torch.log2(1 + P_r_lin / P_n_lin)  # bps/Hz
    
    # Compute uplink data rate (uses full bandwidth with TDMA)
    data_rate = bandwidth * SE.item()  # bps
    
    # Uplink transmission time
    t_uplink = task.length_bits / data_rate  # seconds
    
    # Server processing time
    t_compute = task.required_cycles / server_cpu  # seconds
    
    # Total offloading time (downlink negligible)
    total_time = t_uplink + t_compute
    
    return total_time


def compute_offload_decision(task, iot_device, uav, time_idx, current_time, H_M, H, F, P_T, P_N, bandwidth):
    """
    Compare local vs offload processing and decide.
    
    Args:
        task: Task object
        iot_device: IoTDevice object
        uav: UAV object
        time_idx: Time index for UAV position
        current_time: Current simulation time
        H_M: Mobile height in meters
        H: UAV height in meters
        F: Frequency in Hz
        P_T: Transmit power in dBm
        P_N: Noise power in dBm
        bandwidth: Allocated bandwidth in Hz
    
    Returns:
        decision: 'offload' or 'local'
        t_local: Local processing time
        t_offload: Offloading time
    """
    # Compute local processing time
    t_local = iot_device.compute_local_processing_time(task)
    
    # Compute offloading time
    t_offload = compute_total_offload_time(task, iot_device, uav, time_idx, H_M, H, F, P_T, P_N, bandwidth)
    
    # Make decision based on which is faster
    decision = 'offload' if t_offload < t_local else 'local'
    
    return decision, t_local, t_offload