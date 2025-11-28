"""
Optimization algorithms for UAV network
Author: Khalil El Kaaki & Joe Abi Samra
Date: November 2025
"""

import torch
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
from ..common import p_received, association, bitrate, init_bandwidth, qos_constraint
from .helpers import nonlcon, nonlcon_joint, rate_fn, bitrate_safe


# def optimize_uav_positions(N, AREA, uav_pos, user_pos, H_M, H, F, P_T, P_N, BW, Rmin, device='cuda'):
#     """
#     Optimizes UAV positions using constrained optimization with proportional fairness.
    
#     Args:
#         N: Number of UAVs
#         AREA: Coverage area
#         uav_pos: Initial UAV positions (2, N) or (N, 2)
#         user_pos: User positions (2, M)
#         H_M, H, F, P_T, P_N: Physical parameters
#         BW: Total bandwidth
#         Rmin: Minimum rate requirement
#         device: 'cuda' or 'cpu'
    
#     Returns:
#         uav_pos_opt: Optimized UAV positions (N, 2)
#     """
#     # Convert to tensors
#     if not isinstance(user_pos, torch.Tensor):
#         user_pos = torch.tensor(user_pos, dtype=torch.float32, device=device)
#     else:
#         user_pos = user_pos.to(device)
    
#     if not isinstance(uav_pos, torch.Tensor):
#         uav_pos = torch.tensor(uav_pos, dtype=torch.float32, device=device)
#     else:
#         uav_pos = uav_pos.to(device)
    
#     # Ensure uav_pos is (2, N)
#     if uav_pos.shape[0] != 2:
#         uav_pos = uav_pos.T
    
#     M = user_pos.shape[1]
#     side = np.ceil(np.sqrt(AREA))
    
#     # Bounds
#     lb = np.zeros(2 * N)
#     ub = np.ones(2 * N) * side
    
#     # Flatten initial UAV positions
#     uav_pos_flat = uav_pos.T.reshape(-1).cpu().numpy()  # (2*N,)
    
#     # Define objective function
#     alpha = 0.7
    
#     def objective(x):
#         """Negative of hybrid objective: throughput + fairness"""
#         x_torch = torch.tensor(x, dtype=torch.float32, device=device)
#         uav_pos_reshaped = x_torch.reshape(2, N)
        
#         p_r = p_received(user_pos, uav_pos_reshaped, H_M, H, F, P_T, device=device)
#         a = association(p_r)
#         br = torch.sum(bitrate(p_r, P_N, BW/M, a), dim=1)
#         br_safe = torch.maximum(br, torch.tensor(1e-9, device=device))
        
#         obj_value = -(alpha * torch.sum(br_safe) + (1 - alpha) * torch.sum(torch.log(br_safe)))
#         return obj_value.cpu().detach().numpy()
    
#     # Define constraint function wrapper
#     def constraint_func(x):
#         c, _ = nonlcon(x, user_pos, H_M, H, F, P_T, P_N, BW, Rmin, device=device)
#         return c
    
#     # Optimization options
#     options = {
#         'maxiter': 50,
#         'ftol': 1e-6,
#         'disp': False
#     }
    
#     # Use scipy.optimize.minimize with SLSQP
#     from scipy.optimize import Bounds, NonlinearConstraint
    
#     bounds = Bounds(lb, ub)
    
#     # Nonlinear constraint: c(x) <= 0
#     nlc = NonlinearConstraint(constraint_func, -np.inf, 0)
    
#     result = minimize(objective, uav_pos_flat, method='SLSQP', 
#                      bounds=bounds, constraints=nlc, options=options)
    
#     x_opt = result.x
    
#     # Reshape to (N, 2)
#     uav_pos_opt = x_opt.reshape(N, 2)
    
#     print('Optimized UAV positions (meters):')
#     print(uav_pos_opt)
    
#     return uav_pos_opt


# def optimize_bandwidth_allocation(M, BW_total, user_pos, opt_uav_pos, H_M, H, F, P_T, P_N, Rmin, device='cuda'):
#     """
#     Optimizes bandwidth allocation for given UAV positions.
    
#     Args:
#         M: Number of users
#         BW_total: Total bandwidth
#         user_pos: User positions (2, M)
#         opt_uav_pos: UAV positions (2, N) or (N, 2)
#         H_M, H, F, P_T, P_N: Physical parameters
#         Rmin: Minimum rate requirement
#         device: 'cuda' or 'cpu'
    
#     Returns:
#         B_opt: Optimal bandwidth allocation (M,)
#         br_opt: Resulting bitrates (M,)
#         sum_br_opt_mbps: Total throughput in Mbps
#     """
#     # Convert to tensors
#     if not isinstance(user_pos, torch.Tensor):
#         user_pos = torch.tensor(user_pos, dtype=torch.float32, device=device)
#     else:
#         user_pos = user_pos.to(device)
    
#     if not isinstance(opt_uav_pos, torch.Tensor):
#         opt_uav_pos = torch.tensor(opt_uav_pos, dtype=torch.float32, device=device)
#     else:
#         opt_uav_pos = opt_uav_pos.to(device)
    
#     # Ensure opt_uav_pos is (2, N)
#     if opt_uav_pos.shape[0] != 2:
#         opt_uav_pos = opt_uav_pos.T
    
#     # Initial bandwidth allocation (uniform)
#     B0 = np.ones(M) * (BW_total / M)
    
#     # Bounds
#     lb = np.zeros(M)
#     ub = np.ones(M) * BW_total
    
#     # Calculate received power and association (fixed)
#     p_r = p_received(user_pos, opt_uav_pos, H_M, H, F, P_T, device=device)
#     a = association(p_r)
    
#     # Objective function
#     alpha = 0.7
    
#     def objective(B):
#         """Negative of hybrid objective"""
#         B_torch = torch.tensor(B, dtype=torch.float32, device=device)
#         br = torch.sum(bitrate(p_r, P_N, B_torch, a), dim=1)
#         br_safe = torch.maximum(br, torch.tensor(1e-9, device=device))
        
#         obj_value = -(alpha * torch.sum(br_safe) + (1 - alpha) * torch.sum(torch.log(br_safe)))
#         return obj_value.cpu().detach().numpy()
    
#     # QoS constraint function
#     def qos_constraint_func(B):
#         B_torch = torch.tensor(B, dtype=torch.float32, device=device)
#         br = torch.sum(bitrate(p_r, P_N, B_torch, a), dim=1)
#         c = Rmin - br
#         return c.cpu().detach().numpy()
    
#     # Linear constraint: sum(B) <= BW_total
#     from scipy.optimize import LinearConstraint
#     A_eq = np.ones((1, M))
#     linear_constraint = LinearConstraint(A_eq, -np.inf, BW_total)
    
#     # Nonlinear QoS constraint
#     nlc = NonlinearConstraint(qos_constraint_func, -np.inf, 0)
    
#     # Optimize
#     from scipy.optimize import Bounds
#     bounds = Bounds(lb, ub)
    
#     options = {'maxiter': 50, 'ftol': 1e-6, 'disp': False}
    
#     result = minimize(objective, B0, method='SLSQP', 
#                      bounds=bounds, constraints=[linear_constraint, nlc], 
#                      options=options)
    
#     B_opt = result.x
    
#     # Calculate resulting bitrates
#     B_opt_torch = torch.tensor(B_opt, dtype=torch.float32, device=device)
#     br_opt = torch.sum(bitrate(p_r, P_N, B_opt_torch, a), dim=1)
#     sum_br_opt_mbps = torch.sum(br_opt).cpu().item() / 1e6
    
#     return B_opt, br_opt.cpu().numpy(), sum_br_opt_mbps


def optimize_network(M, N, INITIAL_UAV_POS, BW_total, AREA, user_pos, H_M, H, F, P_T, P_N, Rmin, D_m, C_m, f_UAV, f_user, device='cuda'):
    """
    MEC-aware joint optimization of UAV positions, bandwidth allocation, and offloading decisions.
    
    Decision vector layout: x = [uav_pos_flat (2*N), b_m (M), o_m (M)]
    
    Args:
        M: Number of users
        N: Number of UAVs
        INITIAL_UAV_POS: Initial UAV positions (2, N)
        BW_total: Total bandwidth (Hz)
        AREA: Coverage area
        user_pos: User positions (2, M)
        H_M, H, F, P_T, P_N: Physical parameters
        Rmin: Minimum throughput requirement (bps)
        D_m: Task data size (bits)
        C_m: Computational complexity (CPU cycles)
        f_UAV: UAV CPU frequency (Hz)
        f_user: User CPU frequency (Hz)
        device: 'cuda' or 'cpu'
    
    Returns:
        uav_pos_opt: Optimized UAV positions (2, N)
        Bandwidth_opt: Optimized bandwidth allocation (M,) in Hz
        Offload_opt: Optimized offloading fractions (M,)
        Throughput_opt: Resulting MEC throughputs (M,) in bps
        sumrate_mbps: Total throughput in Mbps
    """
    # Convert to tensors
    if not isinstance(user_pos, torch.Tensor):
        user_pos = torch.tensor(user_pos, dtype=torch.float32, device=device)
    else:
        user_pos = user_pos.to(device)
    
    if not isinstance(INITIAL_UAV_POS, torch.Tensor):
        INITIAL_UAV_POS = torch.tensor(INITIAL_UAV_POS, dtype=torch.float32, device=device)
    else:
        INITIAL_UAV_POS = INITIAL_UAV_POS.to(device)
    
    SIDE = np.ceil(np.sqrt(AREA))
    
    # Normalize UAV positions (first 2*N entries of decision vector)
    UAV_POS_FLAT_norm = (INITIAL_UAV_POS / SIDE).reshape(-1).cpu().numpy()
    
    # Initialize bandwidth allocation (next M entries, in Hz)
    p_r = p_received(user_pos, INITIAL_UAV_POS, H_M, H, F, P_T, device=device)
    A = association(p_r)
    bw_req_user, feasible = init_bandwidth(user_pos, INITIAL_UAV_POS, A, 
                                            Rmin, BW_total, H_M, H, F, P_T, P_N, device=device)
    
    bw_req_user_np = bw_req_user.cpu().numpy()
    bw_req_user_np[~np.isfinite(bw_req_user_np) | (bw_req_user_np < 0)] = 0
    
    # Scale bandwidth to use most of the budget
    # Start with uniform allocation, then let optimizer refine
    b_init = np.ones(M) * (BW_total / M)  # Uniform allocation
    
    # Initialize offloading fractions (last M entries, in [0,1])
    # Greedy heuristic: compute expected time for offload vs local
    # For simplicity, compute the ratio and use sigmoid to bound to [0,1]
    # If T_local > T_offload_expected, prefer offloading
    # Start with 1.0 (full offload) to let optimizer decide
    o_init = np.ones(M) * 1.0  # Start with full offloading
    
    # Decision vector: [UAV positions (normalized), bandwidth (Hz), offloading fractions]
    decision_vector = np.concatenate([UAV_POS_FLAT_norm, b_init, o_init])
    
    # Bounds
    # UAV positions: normalized to [0, 1]
    # Bandwidth: [0, BW_total] Hz
    # Offloading: [0, 1]
    LOWER_BOUND = np.concatenate([
        np.zeros(2*N),        # UAV positions
        np.zeros(M),          # Bandwidth
        np.zeros(M)           # Offloading
    ])
    UPPER_BOUND = np.concatenate([
        np.ones(2*N),         # UAV positions (normalized)
        np.ones(M) * BW_total,  # Bandwidth
        np.ones(M)            # Offloading
    ])
    
    # Sanity check
    assert len(decision_vector) == len(LOWER_BOUND) == len(UPPER_BOUND), \
        f"Decision vector size mismatch: {len(decision_vector)} vs {len(LOWER_BOUND)}"
    assert len(decision_vector) == 2*N + 2*M, \
        f"Expected {2*N + 2*M} decision variables, got {len(decision_vector)}"
    
    # Objective function: maximize sum(log(Th_m)) => minimize -sum(log(Th_m))
    def objective(x):
        """Negative log-sum of MEC throughputs for proportional fairness"""
        Th_m = rate_fn(x, N, SIDE, BW_total, user_pos, H_M, H, F, P_T, P_N, 
                       D_m, C_m, f_UAV, f_user, device=device)
        # Clamp to avoid log(0)
        Th_m_safe = torch.maximum(Th_m, torch.tensor(1e-9, device=device))
        obj = -torch.sum(torch.log(Th_m_safe))
        return obj.cpu().detach().numpy()
    
    # Constraint function
    def constraint_func(x):
        c, _ = nonlcon_joint(x, N, M, user_pos, H_M, H, F, P_T, P_N, BW_total, Rmin, SIDE,
                            D_m, C_m, f_UAV, f_user, device=device)
        return c
    
    # Optimize
    from scipy.optimize import Bounds, NonlinearConstraint
    bounds = Bounds(LOWER_BOUND, UPPER_BOUND)
    nlc = NonlinearConstraint(constraint_func, -np.inf, 0)
    
    # Early stopping: track convergence
    prev_obj = [np.inf]
    iter_count = [0]
    
    def callback(xk):
        """Early stopping if objective stops improving"""
        iter_count[0] += 1
        current_obj = objective(xk)
        
        # Compute relative improvement
        if np.isfinite(prev_obj[0]) and abs(prev_obj[0]) > 1e-9:
            improvement = abs(prev_obj[0] - current_obj) / abs(prev_obj[0])
        else:
            improvement = 1.0  # First iteration or invalid, continue
        
        prev_obj[0] = current_obj
        
        # Stop if improvement < 0.01% for numerical efficiency
        if improvement < 1e-4 and iter_count[0] > 3:
            return True  # Stop optimization
        return False
    
    # Relaxed tolerance for faster convergence, disable verbose output
    options = {'maxiter': 50, 'ftol': 1e-4, 'disp': False}
    
    result = minimize(objective, decision_vector, method='SLSQP',
                     bounds=bounds, constraints=nlc, options=options, callback=callback)
    
    x_opt = result.x
    
    # Extract results
    uav_pos_opt = torch.tensor(x_opt[:2*N].reshape(2, N), dtype=torch.float32, device=device) * SIDE
    Bandwidth_opt = x_opt[2*N:2*N+M]  # Hz
    Offload_opt = x_opt[2*N+M:2*N+2*M]  # Fractions
    
    # Calculate final MEC throughputs
    Throughput_opt = rate_fn(x_opt, N, SIDE, BW_total, user_pos, H_M, H, F, P_T, P_N,
                              D_m, C_m, f_UAV, f_user, device=device)
    sumrate_mbps = torch.sum(Throughput_opt).cpu().item() / 1e6
    
    # Runtime checks and diagnostics
    print("\n" + "="*70)
    print("MEC Optimization Results:")
    print("="*70)
    print(f"  Min Throughput: {torch.min(Throughput_opt).cpu().item()/1e6:.6f} Mbps")
    print(f"  Max Throughput: {torch.max(Throughput_opt).cpu().item()/1e6:.6f} Mbps")
    print(f"  Sum Throughput: {sumrate_mbps:.6f} Mbps")
    print(f"  Total Bandwidth Used: {np.sum(Bandwidth_opt)/1e6:.2f} MHz (Budget: {BW_total/1e6:.2f} MHz)")
    print(f"  Avg Offloading Fraction: {np.mean(Offload_opt):.3f}")
    
    # Check CPU constraints
    A = association(p_received(user_pos, uav_pos_opt, H_M, H, F, P_T, device=device))
    idx = torch.argmax(A, dim=1).cpu().numpy()
    o_m_torch = torch.tensor(Offload_opt, device=device)
    for n in range(N):
        assigned_mask = (idx == n)
        cpu_load_n = np.sum((Offload_opt * C_m / f_UAV)[assigned_mask])
        print(f"  UAV {n} CPU Load: {cpu_load_n:.3f} (Limit: 1.0)")
    print("="*70)
    
    return uav_pos_opt.cpu().numpy(), Bandwidth_opt, Offload_opt, Throughput_opt.cpu().numpy(), sumrate_mbps
