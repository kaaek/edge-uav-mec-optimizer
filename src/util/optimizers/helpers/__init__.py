"""
Helper functions for optimization constraints and objectives
Author: Khalil El Kaaki & Joe Abi Samra
"""

import torch
import numpy as np
from ...common import p_received, association, bitrate, se, compute_mec_throughput


def bitrate_safe(P_r, P_N, B, A):
    """
    Safe bitrate calculation with minimum clamping.
    
    Args:
        P_r: Received power matrix
        P_N: Noise power
        B: Bandwidth allocation
        A: Association matrix
    
    Returns:
        R_safe: Clamped bitrate values
    """
    R_raw = bitrate(P_r, P_N, B, A)
    R_safe = torch.maximum(R_raw, torch.tensor(1e-9, device=R_raw.device))
    return R_safe


def rate_fn(x, N, SIDE, BW_total, user_pos, H_M, H, F, P_T, P_N, D_m, C_m, f_UAV, f_user, device='cuda'):
    """
    MEC-aware rate function for joint optimization.
    
    Decision vector layout: x = [uav_pos_flat (2*N), b_m (M), o_m (M)]
    
    Args:
        x: Decision vector [uav_pos_normalized; bandwidth (Hz); offloading_fraction]
        N: Number of UAVs
        SIDE: Side length of area
        BW_total: Total bandwidth (Hz)
        user_pos: User positions (2, M)
        H_M, H, F, P_T, P_N: Physical parameters
        D_m: Task data size (bits)
        C_m: Computational complexity (CPU cycles)
        f_UAV: UAV CPU frequency (Hz)
        f_user: User CPU frequency (Hz)
        device: 'cuda' or 'cpu'
    
    Returns:
        Th_m: Tensor (M,) - MEC throughput per user (bps)
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32, device=device)
    
    M = user_pos.shape[1]
    
    # Extract and rescale UAV positions (first 2*N entries)
    uav_xy_norm = x[:2*N].reshape(2, N)
    uav_xy = uav_xy_norm * SIDE
    
    # Extract bandwidth allocation (next M entries, in Hz)
    b_m = x[2*N:2*N+M]
    
    # Extract offloading fractions (last M entries, in [0,1])
    o_m = x[2*N+M:2*N+2*M]
    
    # Compute received power matrix (M, N) in dBm
    P_r = p_received(user_pos, uav_xy, H_M, H, F, P_T, device=device)
    
    # Compute association (M, N) - binary one-hot
    A = association(P_r)
    idx = torch.argmax(A, dim=1)  # (M,) - associated UAV index per user
    
    # Compute spectral efficiency matrix (M, N) in bps/Hz
    P_r_lin = 10.0 ** (P_r / 10.0)  # Convert dBm to mW
    P_n_lin = 10.0 ** (P_N / 10.0)
    SE_matrix = torch.log2(1 + P_r_lin / P_n_lin)  # (M, N)
    
    # Compute per-user data rate R_m (bps): R_m = b_m * SE(m, idx(m))
    # Use gather to select SE values for associated UAVs
    SE_selected = torch.gather(SE_matrix, 1, idx.unsqueeze(1)).squeeze(1)  # (M,)
    R_m = b_m * SE_selected  # (M,) in bps
    
    # Compute MEC throughput using fractional offloading model
    Th_m = compute_mec_throughput(R_m, o_m, D_m, C_m, f_UAV, f_user, device=device)
    
    return Th_m


def nonlcon(x, user_pos, H_M, H, F, P_T, P_N, BW, Rmin, device='cuda'):
    """
    Nonlinear constraint function for UAV position optimization.
    Ensures all users meet minimum QoS requirements.
    
    Args:
        x: UAV positions flattened (2*N,)
        user_pos: User positions (2, M)
        H_M, H, F, P_T, P_N: Physical parameters
        BW: Bandwidth
        Rmin: Minimum rate requirement
        device: 'cuda' or 'cpu'
    
    Returns:
        c: Inequality constraints (c <= 0)
        ceq: Equality constraints (empty)
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32, device=device)
    
    # Reshape UAV coordinates
    N = len(x) // 2
    uav_pos = x.reshape(2, N)
    
    # Calculate received power
    p_r_raw = p_received(user_pos, uav_pos, H_M, H, F, P_T, device=device)
    
    # Convert to linear power safely
    if not torch.is_complex(p_r_raw):
        if torch.any(p_r_raw <= 0):
            # dBm to Watts
            p_r_lin = 10.0 ** ((p_r_raw - 30) / 10.0)
        else:
            p_r_lin = p_r_raw
    else:
        p_r_lin = torch.abs(p_r_raw) ** 2
    
    # Association
    A = association(p_r_lin)
    Pr_sel = torch.sum(p_r_lin * A, dim=1)
    Pr_sel = torch.maximum(Pr_sel, torch.tensor(torch.finfo(torch.float32).tiny, device=device))
    
    # Noise and bandwidth
    Pn_W = 10.0 ** ((P_N - 30) / 10.0)
    BW_Hz = BW * 1e6 if BW < 1e6 else BW  # Convert MHz to Hz if needed
    K_users = user_pos.shape[1]
    b_k = (BW_Hz / K_users) * torch.ones(K_users, device=device)
    
    # SNR and rate
    SNR = Pr_sel / max(Pn_W, torch.finfo(torch.float32).tiny)
    r = b_k * torch.log2(1 + SNR)
    
    # QoS constraint: r >= Rmin => Rmin - r <= 0
    c = Rmin - r
    ceq = []
    
    # Convert to numpy for scipy
    if isinstance(c, torch.Tensor):
        c = c.cpu().detach().numpy()
    
    return c, ceq


def nonlcon_joint(x, N, M, user_pos, H_M, H, F, P_T, P_N, BW_total, Rmin, SIDE, D_m, C_m, f_UAV, f_user, device='cuda'):
    """
    MEC-aware nonlinear constraint function for joint optimization.
    
    Decision vector layout: x = [uav_pos_flat (2*N), b_m (M), o_m (M)]
    
    Constraints:
    1. QoS: Th_m >= Rmin for all users (M constraints)
    2. Bandwidth budget: sum(b_m) <= BW_total (1 constraint)
    3. CPU budget per UAV: sum_{m assigned to n} o_m * C_m / f_UAV <= 1 (N constraints)
    
    Args:
        x: Decision vector [uav_pos_normalized; bandwidth (Hz); offloading_fraction]
        N: Number of UAVs
        M: Number of users
        user_pos: User positions (2, M)
        H_M, H, F, P_T, P_N: Physical parameters
        BW_total: Total bandwidth (Hz)
        Rmin: Minimum throughput requirement (bps)
        SIDE: Side length of area
        D_m: Task data size (bits)
        C_m: Computational complexity (CPU cycles)
        f_UAV: UAV CPU frequency (Hz)
        f_user: User CPU frequency (Hz)
        device: 'cuda' or 'cpu'
    
    Returns:
        c: Inequality constraints (c <= 0)
        ceq: Equality constraints (empty)
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32, device=device)
    
    # Extract decision variables
    uav_xy_norm = x[:2*N].reshape(2, N)
    b_m = x[2*N:2*N+M]  # Bandwidth in Hz
    o_m = x[2*N+M:2*N+2*M]  # Offloading fractions
    
    # Rescale UAV positions
    uav_xy = uav_xy_norm * SIDE
    
    # Compute received power and association
    P_r = p_received(user_pos, uav_xy, H_M, H, F, P_T, device=device)
    A = association(P_r)
    idx = torch.argmax(A, dim=1)  # (M,) - associated UAV index
    
    # Compute spectral efficiency and data rates
    P_r_lin = 10.0 ** (P_r / 10.0)
    P_n_lin = 10.0 ** (P_N / 10.0)
    SE_matrix = torch.log2(1 + P_r_lin / P_n_lin)
    SE_selected = torch.gather(SE_matrix, 1, idx.unsqueeze(1)).squeeze(1)
    R_m = b_m * SE_selected  # (M,) in bps
    
    # Compute MEC throughput
    Th_m = compute_mec_throughput(R_m, o_m, D_m, C_m, f_UAV, f_user, device=device)
    
    # Constraint 1: QoS - Th_m >= Rmin => Rmin - Th_m <= 0
    c_qos = Rmin - Th_m  # (M,)
    
    # Constraint 2: Bandwidth budget - sum(b_m) <= BW_total
    c_bw = torch.sum(b_m) - BW_total  # (1,)
    
    # Constraint 3: CPU budget per UAV - for each UAV n, sum_{m: idx(m)==n} o_m * C_m / f_UAV <= 1
    c_cpu = []
    for n in range(N):
        # Find users assigned to UAV n
        assigned_mask = (idx == n)  # (M,) boolean
        # Compute CPU load for UAV n (in time units per slot)
        cpu_load_n = torch.sum((o_m * C_m / f_UAV) * assigned_mask.float())
        # Constraint: cpu_load_n <= 1
        c_cpu.append(cpu_load_n - 1.0)
    c_cpu = torch.stack(c_cpu)  # (N,)
    
    # Stack all constraints
    c = torch.cat([c_qos, c_bw.unsqueeze(0), c_cpu])
    ceq = []
    
    # Convert to numpy for scipy
    if isinstance(c, torch.Tensor):
        c = c.cpu().detach().numpy()
    
    return c, ceq
