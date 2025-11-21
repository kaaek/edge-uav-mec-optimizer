"""
Helper functions for optimization constraints and objectives
Author: Khalil El Kaaki & Joe Abi Samra
Date: 25/10/2025
Translated to Python with GPU support using PyTorch
"""

import torch
import numpy as np
from ...common import p_received, association, bitrate, se


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


def rate_fn(x, N, SIDE, BW_total, user_pos, H_M, H, F, P_T, P_N, device='cuda'):
    """
    Rate function for joint optimization.
    
    Args:
        x: Decision vector [uav_pos_normalized; bandwidth_normalized]
        N: Number of UAVs
        SIDE: Side length of area
        BW_total: Total bandwidth
        user_pos: User positions
        H_M, H, F, P_T, P_N: Physical parameters
        device: 'cuda' or 'cpu'
    
    Returns:
        R: Per-user rates
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32, device=device)
    
    # Extract and rescale UAV positions
    uav_xy_norm = x[:2*N].reshape(2, N)
    uav_xy = uav_xy_norm * SIDE
    
    # Extract and rescale bandwidth
    b_norm = x[2*N:]
    B = b_norm * BW_total
    
    # Calculate received power and association
    P_r = p_received(user_pos, uav_xy, H_M, H, F, P_T, device=device)
    A = association(P_r)
    
    # Calculate rates
    R = bitrate_safe(P_r, P_N, B, A)
    R = torch.max(R, dim=1)[0]  # Take max along each row (user)
    
    return R


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


def nonlcon_joint(x, N, M, user_pos, H_M, H, F, P_T, P_N, BW_total, Rmin, SIDE, device='cuda'):
    """
    Nonlinear constraint function for joint UAV position and bandwidth optimization.
    
    Args:
        x: Decision vector [uav_pos_normalized; bandwidth_normalized]
        N: Number of UAVs
        M: Number of users
        user_pos: User positions
        H_M, H, F, P_T, P_N: Physical parameters
        BW_total: Total bandwidth
        Rmin: Minimum rate
        SIDE: Side length
        device: 'cuda' or 'cpu'
    
    Returns:
        c: Inequality constraints
        ceq: Equality constraints (empty)
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32, device=device)
    
    # Extract normalized variables
    uav_xy_norm = x[:2*N].reshape(2, N)
    b_norm = x[2*N:]
    
    # Rescale to physical units
    uav_xy = uav_xy_norm * SIDE
    B = b_norm * BW_total
    
    # Received power, association, and rate
    P_r = p_received(user_pos, uav_xy, H_M, H, F, P_T, device=device)
    A = association(P_r)
    R = torch.sum(bitrate(P_r, P_N, B, A), dim=1)
    
    # Constraints
    c_qos = Rmin - R  # QoS constraint: R >= Rmin
    c_bw_norm = torch.sum(b_norm) - 1  # Bandwidth normalization constraint
    c = torch.cat([c_qos.flatten(), c_bw_norm.unsqueeze(0)])
    ceq = []
    
    # Convert to numpy for scipy
    if isinstance(c, torch.Tensor):
        c = c.cpu().detach().numpy()
    
    return c, ceq
