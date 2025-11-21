"""
Main benchmarking script for UAV-Ground Association Optimizer
Author: Khalil El Kaaki & Joe Abi Samra
Date: 23/10/2025
Translated to Python with GPU support

This script performs parameter sweeps to evaluate different UAV positioning algorithms.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from util.constants import constants
from util.benchmark_vals import benchmark_vals
from util.clustering import k_means_uav, hierarchical_uav
from util.optimizers import optimize_network
from util.plotter import plot_sweep

# Check for CUDA availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available. Running on CPU.")
    print("To enable GPU acceleration, install PyTorch with CUDA:")
    print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

# Get constants
M, N, AREA, H, H_M, F, K, GAMMA, D_0, P_T, P_N, MAX_ITER, TOL, BW_total, R_MIN, SIDE, TRIALS = constants()

# Get benchmark values
N_vals, M_vals, BW_vals, P_t_vals, Rmin_vals, Area_vals = benchmark_vals()

print("=" * 70)
print("UAV-Ground Association Optimizer - Benchmark Suite")
print("=" * 70)
print(f"System Parameters:")
print(f"  Users (M): {M}")
print(f"  UAVs (N): {N}")
print(f"  Area: {AREA} m²")
print(f"  UAV Height: {H} m")
print(f"  Frequency: {F/1e6} MHz")
print(f"  Transmit Power: {P_T} dBm")
print(f"  Noise Power: {P_N} dBm")
print(f"  Total Bandwidth: {BW_total/1e6} MHz")
print(f"  Minimum Rate: {R_MIN/1e6} Mbps")
print(f"  Trials: {TRIALS}")
print("=" * 70)

# Precompute user positions for all trials
print("\nPrecomputing user positions for all trials...")
user_pos_trials = []
for j in range(TRIALS):
    torch.manual_seed(j)
    np.random.seed(j)
    user_pos = SIDE * torch.rand(2, M, device=device)
    user_pos_trials.append(user_pos)

# Common arrays for storing trial results
trials_kmeans_ref = np.zeros(TRIALS)
trials_kmeans_opt = np.zeros(TRIALS)
trials_hier_ref = np.zeros(TRIALS)
trials_hier_opt = np.zeros(TRIALS)

# # ========================================================================
# # N Sweep (Number of UAVs)
# # ========================================================================
# print("\n" + "=" * 70)
# print("SWEEP 1: Number of UAVs (N)")
# print("=" * 70)

# sumrate_kmeans_ref_arr = np.zeros(len(N_vals))
# sumrate_kmeans_opt_arr = np.zeros(len(N_vals))
# sumrate_hier_ref_arr = np.zeros(len(N_vals))
# sumrate_hier_opt_arr = np.zeros(len(N_vals))

# start_time_total = time.time()

# for i, n in enumerate(N_vals):
#     print(f"\nN = {n} ({i+1}/{len(N_vals)})")
#     start_time = time.time()
    
#     for j in range(TRIALS):
#         if (j + 1) % 10 == 0:
#             print(f"  Trial {j+1}/{TRIALS}...", end='\r')
        
#         user_pos = user_pos_trials[j]
        
#         # K-means Clustering
#         uav_pos_kmeans, _, sumrate = k_means_uav(user_pos, M, n, AREA, H_M, H, F, P_T, P_N, MAX_ITER, TOL, BW_total, device=device)
#         trials_kmeans_ref[j] = sumrate.cpu().item() if isinstance(sumrate, torch.Tensor) else sumrate
        
#         # Optimize from k-means initial position
#         uav_pos_opt_kmeans, _, _, sumrate = optimize_network(M, n, uav_pos_kmeans, BW_total, AREA, user_pos, H_M, H, F, P_T, P_N, R_MIN, device=device)
#         trials_kmeans_opt[j] = sumrate
        
#         # Hierarchical Clustering
#         uav_pos_hier, _, sumrate = hierarchical_uav(user_pos, n, H_M, H, F, P_T, P_N, BW_total, device=device)
#         trials_hier_ref[j] = sumrate.cpu().item() if isinstance(sumrate, torch.Tensor) else sumrate
        
#         # Optimize from hierarchical initial position
#         uav_pos_opt_hier, _, _, sumrate = optimize_network(M, n, uav_pos_hier, BW_total, AREA, user_pos, H_M, H, F, P_T, P_N, R_MIN, device=device)
#         trials_hier_opt[j] = sumrate
    
#     # Compute averages
#     sumrate_kmeans_ref_arr[i] = np.mean(trials_kmeans_ref)
#     sumrate_kmeans_opt_arr[i] = np.mean(trials_kmeans_opt)
#     sumrate_hier_ref_arr[i] = np.mean(trials_hier_ref)
#     sumrate_hier_opt_arr[i] = np.mean(trials_hier_opt)
    
#     elapsed = time.time() - start_time
#     print(f"  N={n} completed in {elapsed:.1f}s")
#     print(f"    K-means Ref: {sumrate_kmeans_ref_arr[i]:.2f} Mbps")
#     print(f"    K-means Opt: {sumrate_kmeans_opt_arr[i]:.2f} Mbps")
#     print(f"    Hier Ref: {sumrate_hier_ref_arr[i]:.2f} Mbps")
#     print(f"    Hier Opt: {sumrate_hier_opt_arr[i]:.2f} Mbps")
    
#     # Reset trial arrays
#     trials_kmeans_ref = np.zeros(TRIALS)
#     trials_kmeans_opt = np.zeros(TRIALS)
#     trials_hier_ref = np.zeros(TRIALS)
#     trials_hier_opt = np.zeros(TRIALS)

# total_elapsed = time.time() - start_time_total
# print(f"\nN Sweep completed in {total_elapsed:.1f}s")

# # Plot N sweep results
# fig = plot_sweep(N_vals, sumrate_kmeans_ref_arr, sumrate_kmeans_opt_arr, 
#                  sumrate_hier_ref_arr, sumrate_hier_opt_arr, 
#                  'Number of UAVs (N)', 
#                  'Variation of the Sum Rate Relative to the Number of UAVs')
# plt.savefig('n_sweep_results.png', dpi=300, bbox_inches='tight')
# print("Saved plot: n_sweep_results.png")

# # ========================================================================
# # Uncomment sections below to run additional sweeps
# # ========================================================================

# # M Sweep (Number of Users)
# print("\n" + "=" * 70)
# print("SWEEP 2: Number of Users (M)")
# print("=" * 70)

# sumrate_kmeans_ref_arr = np.zeros(len(M_vals))
# sumrate_kmeans_opt_arr = np.zeros(len(M_vals))
# sumrate_hier_ref_arr = np.zeros(len(M_vals))
# sumrate_hier_opt_arr = np.zeros(len(M_vals))

# for i, m in enumerate(M_vals):
#     print(f"\nM = {m} ({i+1}/{len(M_vals)})")
    
#     # Generate new user positions for this M
#     user_pos_trials_m = []
#     for j in range(TRIALS):
#         torch.manual_seed(j)
#         np.random.seed(j)
#         user_pos = SIDE * torch.rand(2, m, device=device)
#         user_pos_trials_m.append(user_pos)
    
#     for j in range(TRIALS):
#         if (j + 1) % 10 == 0:
#             print(f"  Trial {j+1}/{TRIALS}...", end='\r')
        
#         user_pos = user_pos_trials_m[j]
        
#         # K-means
#         uav_pos_kmeans, _, sumrate = k_means_uav(user_pos, m, N, AREA, H_M, H, F, P_T, P_N, MAX_ITER, TOL, BW_total, device=device)
#         trials_kmeans_ref[j] = sumrate.cpu().item() if isinstance(sumrate, torch.Tensor) else sumrate
        
#         uav_pos_opt_kmeans, _, _, sumrate = optimize_network(m, N, uav_pos_kmeans, BW_total, AREA, user_pos, H_M, H, F, P_T, P_N, R_MIN, device=device)
#         trials_kmeans_opt[j] = sumrate
        
#         # Hierarchical
#         uav_pos_hier, _, sumrate = hierarchical_uav(user_pos, N, H_M, H, F, P_T, P_N, BW_total, device=device)
#         trials_hier_ref[j] = sumrate.cpu().item() if isinstance(sumrate, torch.Tensor) else sumrate
        
#         uav_pos_opt_hier, _, _, sumrate = optimize_network(m, N, uav_pos_hier, BW_total, AREA, user_pos, H_M, H, F, P_T, P_N, R_MIN, device=device)
#         trials_hier_opt[j] = sumrate
    
#     sumrate_kmeans_ref_arr[i] = np.mean(trials_kmeans_ref)
#     sumrate_kmeans_opt_arr[i] = np.mean(trials_kmeans_opt)
#     sumrate_hier_ref_arr[i] = np.mean(trials_hier_ref)
#     sumrate_hier_opt_arr[i] = np.mean(trials_hier_opt)
    
#     print(f"  M={m} - K-means Opt: {sumrate_kmeans_opt_arr[i]:.2f} Mbps, Hier Opt: {sumrate_hier_opt_arr[i]:.2f} Mbps")
    
#     trials_kmeans_ref = np.zeros(TRIALS)
#     trials_kmeans_opt = np.zeros(TRIALS)
#     trials_hier_ref = np.zeros(TRIALS)
#     trials_hier_opt = np.zeros(TRIALS)

# fig = plot_sweep(M_vals, sumrate_kmeans_ref_arr, sumrate_kmeans_opt_arr, 
#                  sumrate_hier_ref_arr, sumrate_hier_opt_arr, 
#                  'Number of Users (M)', 
#                  'Variation of the Sum Rate Relative to the Number of Users')
# plt.savefig('m_sweep_results.png', dpi=300, bbox_inches='tight')

# # ========================================================================
# # BW Sweep (Bandwidth)
# # ========================================================================
# print("\n" + "=" * 70)
# print("SWEEP 3: Total Bandwidth (BW)")
# print("=" * 70)

# sumrate_kmeans_ref_arr = np.zeros(len(BW_vals))
# sumrate_kmeans_opt_arr = np.zeros(len(BW_vals))
# sumrate_hier_ref_arr = np.zeros(len(BW_vals))
# sumrate_hier_opt_arr = np.zeros(len(BW_vals))

# for i, bw in enumerate(BW_vals):
#     print(f"\nBW = {bw/1e6:.1f} MHz ({i+1}/{len(BW_vals)})")
    
#     for j in range(TRIALS):
#         if (j + 1) % 10 == 0:
#             print(f"  Trial {j+1}/{TRIALS}...", end='\r')
        
#         user_pos = user_pos_trials[j]
        
#         # K-means
#         uav_pos_kmeans, _, sumrate = k_means_uav(user_pos, M, N, AREA, H_M, H, F, P_T, P_N, MAX_ITER, TOL, bw, device=device)
#         trials_kmeans_ref[j] = sumrate.cpu().item() if isinstance(sumrate, torch.Tensor) else sumrate
        
#         uav_pos_opt_kmeans, _, _, sumrate = optimize_network(M, N, uav_pos_kmeans, bw, AREA, user_pos, H_M, H, F, P_T, P_N, R_MIN, device=device)
#         trials_kmeans_opt[j] = sumrate
        
#         # Hierarchical
#         uav_pos_hier, _, sumrate = hierarchical_uav(user_pos, N, H_M, H, F, P_T, P_N, bw, device=device)
#         trials_hier_ref[j] = sumrate.cpu().item() if isinstance(sumrate, torch.Tensor) else sumrate
        
#         uav_pos_opt_hier, _, _, sumrate = optimize_network(M, N, uav_pos_hier, bw, AREA, user_pos, H_M, H, F, P_T, P_N, R_MIN, device=device)
#         trials_hier_opt[j] = sumrate
    
#     sumrate_kmeans_ref_arr[i] = np.mean(trials_kmeans_ref)
#     sumrate_kmeans_opt_arr[i] = np.mean(trials_kmeans_opt)
#     sumrate_hier_ref_arr[i] = np.mean(trials_hier_ref)
#     sumrate_hier_opt_arr[i] = np.mean(trials_hier_opt)
    
#     print(f"  BW={bw/1e6:.1f}MHz - K-means Opt: {sumrate_kmeans_opt_arr[i]:.2f} Mbps, Hier Opt: {sumrate_hier_opt_arr[i]:.2f} Mbps")
    
#     trials_kmeans_ref = np.zeros(TRIALS)
#     trials_kmeans_opt = np.zeros(TRIALS)
#     trials_hier_ref = np.zeros(TRIALS)
#     trials_hier_opt = np.zeros(TRIALS)

# fig = plot_sweep(BW_vals/1e6, sumrate_kmeans_ref_arr, sumrate_kmeans_opt_arr, 
#                  sumrate_hier_ref_arr, sumrate_hier_opt_arr, 
#                  'Total Bandwidth (MHz)', 
#                  'Variation of the Sum Rate Relative to the Total Bandwidth')
# plt.savefig('bw_sweep_results.png', dpi=300, bbox_inches='tight')
# print("Saved plot: bw_sweep_results.png")

# # ========================================================================
# # P_t Sweep (Transmit Power)
# # ========================================================================
# print("\n" + "=" * 70)
# print("SWEEP 4: Transmit Power (P_T)")
# print("=" * 70)

# sumrate_kmeans_ref_arr = np.zeros(len(P_t_vals))
# sumrate_kmeans_opt_arr = np.zeros(len(P_t_vals))
# sumrate_hier_ref_arr = np.zeros(len(P_t_vals))
# sumrate_hier_opt_arr = np.zeros(len(P_t_vals))

# for i, p_t in enumerate(P_t_vals):
#     print(f"\nP_T = {p_t} dBm ({i+1}/{len(P_t_vals)})")
    
#     for j in range(TRIALS):
#         if (j + 1) % 10 == 0:
#             print(f"  Trial {j+1}/{TRIALS}...", end='\r')
        
#         user_pos = user_pos_trials[j]
        
#         # K-means
#         uav_pos_kmeans, _, sumrate = k_means_uav(user_pos, M, N, AREA, H_M, H, F, p_t, P_N, MAX_ITER, TOL, BW_total, device=device)
#         trials_kmeans_ref[j] = sumrate.cpu().item() if isinstance(sumrate, torch.Tensor) else sumrate
        
#         uav_pos_opt_kmeans, _, _, sumrate = optimize_network(M, N, uav_pos_kmeans, BW_total, AREA, user_pos, H_M, H, F, p_t, P_N, R_MIN, device=device)
#         trials_kmeans_opt[j] = sumrate
        
#         # Hierarchical
#         uav_pos_hier, _, sumrate = hierarchical_uav(user_pos, N, H_M, H, F, p_t, P_N, BW_total, device=device)
#         trials_hier_ref[j] = sumrate.cpu().item() if isinstance(sumrate, torch.Tensor) else sumrate
        
#         uav_pos_opt_hier, _, _, sumrate = optimize_network(M, N, uav_pos_hier, BW_total, AREA, user_pos, H_M, H, F, p_t, P_N, R_MIN, device=device)
#         trials_hier_opt[j] = sumrate
    
#     sumrate_kmeans_ref_arr[i] = np.mean(trials_kmeans_ref)
#     sumrate_kmeans_opt_arr[i] = np.mean(trials_kmeans_opt)
#     sumrate_hier_ref_arr[i] = np.mean(trials_hier_ref)
#     sumrate_hier_opt_arr[i] = np.mean(trials_hier_opt)
    
#     print(f"  P_T={p_t}dBm - K-means Opt: {sumrate_kmeans_opt_arr[i]:.2f} Mbps, Hier Opt: {sumrate_hier_opt_arr[i]:.2f} Mbps")
    
#     trials_kmeans_ref = np.zeros(TRIALS)
#     trials_kmeans_opt = np.zeros(TRIALS)
#     trials_hier_ref = np.zeros(TRIALS)
#     trials_hier_opt = np.zeros(TRIALS)

# fig = plot_sweep(P_t_vals, sumrate_kmeans_ref_arr, sumrate_kmeans_opt_arr, 
#                  sumrate_hier_ref_arr, sumrate_hier_opt_arr, 
#                  'Transmit Power (dBm)', 
#                  'Variation of the Sum Rate Relative to the Transmit Power')
# plt.savefig('pt_sweep_results.png', dpi=300, bbox_inches='tight')
# print("Saved plot: pt_sweep_results.png")

# # ========================================================================
# # R_min Sweep (QoS Requirement)
# # ========================================================================
# print("\n" + "=" * 70)
# print("SWEEP 5: Minimum QoS (R_MIN)")
# print("=" * 70)

# sumrate_kmeans_ref_arr = np.zeros(len(Rmin_vals))
# sumrate_kmeans_opt_arr = np.zeros(len(Rmin_vals))
# sumrate_hier_ref_arr = np.zeros(len(Rmin_vals))
# sumrate_hier_opt_arr = np.zeros(len(Rmin_vals))

# for i, rmin in enumerate(Rmin_vals):
#     print(f"\nR_MIN = {rmin/1e3:.1f} kbps ({i+1}/{len(Rmin_vals)})")
    
#     for j in range(TRIALS):
#         if (j + 1) % 10 == 0:
#             print(f"  Trial {j+1}/{TRIALS}...", end='\r')
        
#         user_pos = user_pos_trials[j]
        
#         # K-means
#         uav_pos_kmeans, _, sumrate = k_means_uav(user_pos, M, N, AREA, H_M, H, F, P_T, P_N, MAX_ITER, TOL, BW_total, device=device)
#         trials_kmeans_ref[j] = sumrate.cpu().item() if isinstance(sumrate, torch.Tensor) else sumrate
        
#         uav_pos_opt_kmeans, _, _, sumrate = optimize_network(M, N, uav_pos_kmeans, BW_total, AREA, user_pos, H_M, H, F, P_T, P_N, rmin, device=device)
#         trials_kmeans_opt[j] = sumrate
        
#         # Hierarchical
#         uav_pos_hier, _, sumrate = hierarchical_uav(user_pos, N, H_M, H, F, P_T, P_N, BW_total, device=device)
#         trials_hier_ref[j] = sumrate.cpu().item() if isinstance(sumrate, torch.Tensor) else sumrate
        
#         uav_pos_opt_hier, _, _, sumrate = optimize_network(M, N, uav_pos_hier, BW_total, AREA, user_pos, H_M, H, F, P_T, P_N, rmin, device=device)
#         trials_hier_opt[j] = sumrate
    
#     sumrate_kmeans_ref_arr[i] = np.mean(trials_kmeans_ref)
#     sumrate_kmeans_opt_arr[i] = np.mean(trials_kmeans_opt)
#     sumrate_hier_ref_arr[i] = np.mean(trials_hier_ref)
#     sumrate_hier_opt_arr[i] = np.mean(trials_hier_opt)
    
#     print(f"  R_MIN={rmin/1e3:.1f}kbps - K-means Opt: {sumrate_kmeans_opt_arr[i]:.2f} Mbps, Hier Opt: {sumrate_hier_opt_arr[i]:.2f} Mbps")
    
#     trials_kmeans_ref = np.zeros(TRIALS)
#     trials_kmeans_opt = np.zeros(TRIALS)
#     trials_hier_ref = np.zeros(TRIALS)
#     trials_hier_opt = np.zeros(TRIALS)

# fig = plot_sweep(Rmin_vals/1e3, sumrate_kmeans_ref_arr, sumrate_kmeans_opt_arr, 
#                  sumrate_hier_ref_arr, sumrate_hier_opt_arr, 
#                  'Minimum QoS (kbps)', 
#                  'Variation of the Sum Rate Relative to the Minimum QoS')
# plt.savefig('rmin_sweep_results.png', dpi=300, bbox_inches='tight')
# print("Saved plot: rmin_sweep_results.png")

# # ========================================================================
# # Area Sweep
# # ========================================================================
# print("\n" + "=" * 70)
# print("SWEEP 6: Coverage Area")
# print("=" * 70)

# sumrate_kmeans_ref_arr = np.zeros(len(Area_vals))
# sumrate_kmeans_opt_arr = np.zeros(len(Area_vals))
# sumrate_hier_ref_arr = np.zeros(len(Area_vals))
# sumrate_hier_opt_arr = np.zeros(len(Area_vals))

# for i, area in enumerate(Area_vals):
#     print(f"\nArea = {area/1e6:.1f} km² ({i+1}/{len(Area_vals)})")
#     side_area = np.sqrt(area)
    
#     # Generate new user positions for this area
#     user_pos_trials_area = []
#     for j in range(TRIALS):
#         torch.manual_seed(j)
#         np.random.seed(j)
#         user_pos = side_area * torch.rand(2, M, device=device)
#         user_pos_trials_area.append(user_pos)
    
#     for j in range(TRIALS):
#         if (j + 1) % 10 == 0:
#             print(f"  Trial {j+1}/{TRIALS}...", end='\r')
        
#         user_pos = user_pos_trials_area[j]
        
#         # K-means
#         uav_pos_kmeans, _, sumrate = k_means_uav(user_pos, M, N, area, H_M, H, F, P_T, P_N, MAX_ITER, TOL, BW_total, device=device)
#         trials_kmeans_ref[j] = sumrate.cpu().item() if isinstance(sumrate, torch.Tensor) else sumrate
        
#         uav_pos_opt_kmeans, _, _, sumrate = optimize_network(M, N, uav_pos_kmeans, BW_total, area, user_pos, H_M, H, F, P_T, P_N, R_MIN, device=device)
#         trials_kmeans_opt[j] = sumrate
        
#         # Hierarchical
#         uav_pos_hier, _, sumrate = hierarchical_uav(user_pos, N, H_M, H, F, P_T, P_N, BW_total, device=device)
#         trials_hier_ref[j] = sumrate.cpu().item() if isinstance(sumrate, torch.Tensor) else sumrate
        
#         uav_pos_opt_hier, _, _, sumrate = optimize_network(M, N, uav_pos_hier, BW_total, area, user_pos, H_M, H, F, P_T, P_N, R_MIN, device=device)
#         trials_hier_opt[j] = sumrate
    
#     sumrate_kmeans_ref_arr[i] = np.mean(trials_kmeans_ref)
#     sumrate_kmeans_opt_arr[i] = np.mean(trials_kmeans_opt)
#     sumrate_hier_ref_arr[i] = np.mean(trials_hier_ref)
#     sumrate_hier_opt_arr[i] = np.mean(trials_hier_opt)
    
#     print(f"  Area={area/1e6:.1f}km² - K-means Opt: {sumrate_kmeans_opt_arr[i]:.2f} Mbps, Hier Opt: {sumrate_hier_opt_arr[i]:.2f} Mbps")
    
#     trials_kmeans_ref = np.zeros(TRIALS)
#     trials_kmeans_opt = np.zeros(TRIALS)
#     trials_hier_ref = np.zeros(TRIALS)
#     trials_hier_opt = np.zeros(TRIALS)

# fig = plot_sweep(Area_vals/1e6, sumrate_kmeans_ref_arr, sumrate_kmeans_opt_arr, 
#                  sumrate_hier_ref_arr, sumrate_hier_opt_arr, 
#                  'Area (km²)', 
#                  'Variation of the Sum Rate Relative to the Area')
# plt.savefig('area_sweep_results.png', dpi=300, bbox_inches='tight')

# Do not change below this line
# =================================================================
print("Saved plot: area_sweep_results.png")

print("\n" + "=" * 70)
print("Benchmark suite completed!")
print("=" * 70)
print("\nTo run sweeps, uncomment the desired sections in main.py:")
print("  - N Sweep (Number of UAVs)")
print("  - M Sweep (Number of Users)")
print("  - BW Sweep (Bandwidth)")
print("  - P_t Sweep (Transmit Power)")
print("  - R_min Sweep (QoS Requirement)")
print("  - Area Sweep (Coverage Area)")
plt.show()