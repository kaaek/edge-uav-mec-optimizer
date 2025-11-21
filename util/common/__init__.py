"""
Common utility functions for UAV network optimization
Author: Khalil El Kaaki & Joe Abi Samra
Date: 23/10/2025
Translated to Python with GPU support using PyTorch
"""

import torch
import numpy as np


def p_received(user_pos, uav_pos, H_M, H, F, P_T, device='cuda'):
    """
    Calculates the received power using the Okumura-Hata model.
    
    Args:
        user_pos: Tensor of shape (2, M) - user positions (x, y)
        uav_pos: Tensor of shape (2, N) - UAV positions (x, y)
        H_M: Mobile height in meters
        H: UAV height in meters
        F: Frequency in Hz
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
    
    # Calculate 3D distances (MxN matrix)
    d = torch.sqrt((x_m - x_n)**2 + (y_m - y_n)**2 + H**2)
    
    # Okumura-Hata path loss model (vectorized)
    F_MHz = F / 1e6  # Convert to MHz for the model
    C_h = 0.8 + (1.1 * torch.log10(torch.tensor(F_MHz, device=device)) - 0.7) * H_M - 1.56 * torch.log10(torch.tensor(F_MHz, device=device))
    L_u = 69.55 + 26.16 * torch.log10(torch.tensor(F_MHz, device=device)) - 13.82 * torch.log10(torch.tensor(H, device=device)) - C_h + \
          (44.9 - 6.55 * torch.log10(torch.tensor(H, device=device))) * torch.log10(d)
    L = L_u - 4.78 * (torch.log10(torch.tensor(F_MHz, device=device)))**2 + 18.33 * torch.log10(torch.tensor(F_MHz, device=device)) - 40.94
    
    # Received power in dBm
    p_r = P_T - L
    
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
    SE = torch.log2(1 + SNR) * ASSOCIATION_MATRIX  # (M, N)
    
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
