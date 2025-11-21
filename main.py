"""
Main benchmarking script for Edge UAV-MEC Optimizer. This script performs parameter sweeps to evaluate different UAV positioning algorithms.
Author: Khalil El Kaaki & Joe Abi Samra
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from util.constants import constants
from util.benchmark_vals import benchmark_vals
from util.clustering import k_means_uav, hierarchical_uav
from util.optimizers import optimize_network
from util.optimizers.batch_optimizer import batch_optimize_trials
from util.plotter import plot_sweep

# Check for CUDA availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available. Running on CPU.")
    print("To enable GPU acceleration, install PyTorch with CUDA:")
    print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130")

# Get constants
M, N, AREA, H, H_M, F, K, GAMMA, D_0, P_T, P_N, MAX_ITER, TOL, BW_total, R_MIN, SIDE, TRIALS, D_m, C_m, f_UAV, f_user = constants()

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

# # ========================================================================
# # N Sweep (Number of UAVs)
# # ========================================================================
print("\n" + "=" * 70)
print("SWEEP 1: Number of UAVs (N)")
print("=" * 70)

sumrate_kmeans_ref_arr = np.zeros(len(N_vals))
sumrate_kmeans_opt_arr = np.zeros(len(N_vals))
sumrate_hier_ref_arr = np.zeros(len(N_vals))
sumrate_hier_opt_arr = np.zeros(len(N_vals))

start_time_total = time.time()

for i, n in enumerate(N_vals):
    print(f"\nN = {n} ({i+1}/{len(N_vals)})")
    start_time = time.time()
    
    # Run K-means trials in parallel
    kmeans_baseline, kmeans_optimized = batch_optimize_trials(
        user_pos_trials=user_pos_trials,
        M=M,
        N=n,
        AREA=AREA,
        H_M=H_M,
        H=H,
        F=F,
        P_T=P_T,
        P_N=P_N,
        MAX_ITER=MAX_ITER,
        TOL=TOL,
        BW_total=BW_total,
        R_MIN=R_MIN,
        D_m=D_m,
        C_m=C_m,
        f_UAV=f_UAV,
        f_user=f_user,
        device=device,
        num_workers=2,
        clustering_method='kmeans'
    )
    
    # Run hierarchical trials in parallel
    hier_baseline, hier_optimized = batch_optimize_trials(
        user_pos_trials=user_pos_trials,
        M=M,
        N=n,
        AREA=AREA,
        H_M=H_M,
        H=H,
        F=F,
        P_T=P_T,
        P_N=P_N,
        MAX_ITER=MAX_ITER,
        TOL=TOL,
        BW_total=BW_total,
        R_MIN=R_MIN,
        D_m=D_m,
        C_m=C_m,
        f_UAV=f_UAV,
        f_user=f_user,
        device=device,
        num_workers=2,
        clustering_method='hierarchical'
    )
    
    # Extract results (compute means from lists)
    sumrate_kmeans_ref_arr[i] = np.mean(kmeans_baseline)
    sumrate_kmeans_opt_arr[i] = np.mean(kmeans_optimized)
    sumrate_hier_ref_arr[i] = np.mean(hier_baseline)
    sumrate_hier_opt_arr[i] = np.mean(hier_optimized)
    
    elapsed = time.time() - start_time
    print(f"  N={n} completed in {elapsed:.1f}s")
    print(f"    K-means Ref: {sumrate_kmeans_ref_arr[i]:.2f} Mbps")
    print(f"    K-means Opt: {sumrate_kmeans_opt_arr[i]:.2f} Mbps")
    print(f"    Hier Ref: {sumrate_hier_ref_arr[i]:.2f} Mbps")
    print(f"    Hier Opt: {sumrate_hier_opt_arr[i]:.2f} Mbps")

total_elapsed = time.time() - start_time_total
print(f"\nN Sweep completed in {total_elapsed:.1f}s")

# Plot N sweep results
fig = plot_sweep(N_vals, sumrate_kmeans_ref_arr, sumrate_kmeans_opt_arr, 
                 sumrate_hier_ref_arr, sumrate_hier_opt_arr, 
                 'Number of UAVs (N)', 
                 'Variation of the Sum Rate Relative to the Number of UAVs')
plt.savefig('n_sweep_results.png', dpi=300, bbox_inches='tight')
print("Saved plot: n_sweep_results.png")

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

# start_time_total = time.time()

# for i, m in enumerate(M_vals):
#     print(f"\nM = {m} ({i+1}/{len(M_vals)})")
#     start_time = time.time()
    
#     # Generate new user positions for this M
#     user_pos_trials_m = []
#     for j in range(TRIALS):
#         torch.manual_seed(j)
#         np.random.seed(j)
#         user_pos = SIDE * torch.rand(2, m, device=device)
#         user_pos_trials_m.append(user_pos)
    
#     # Run K-means trials in parallel
#     kmeans_baseline, kmeans_optimized = batch_optimize_trials(
#         user_pos_trials=user_pos_trials_m,
#         M=m,
#         N=N,
#         AREA=AREA,
#         H_M=H_M,
#         H=H,
#         F=F,
#         P_T=P_T,
#         P_N=P_N,
#         MAX_ITER=MAX_ITER,
#         TOL=TOL,
#         BW_total=BW_total,
#         R_MIN=R_MIN,
#         D_m=D_m,
#         C_m=C_m,
#         f_UAV=f_UAV,
#         f_user=f_user,
#         device=device,
#         num_workers=2,
#         clustering_method='kmeans'
#     )
    
#     # Run hierarchical trials in parallel
#     hier_baseline, hier_optimized = batch_optimize_trials(
#         user_pos_trials=user_pos_trials_m,
#         M=m,
#         N=N,
#         AREA=AREA,
#         H_M=H_M,
#         H=H,
#         F=F,
#         P_T=P_T,
#         P_N=P_N,
#         MAX_ITER=MAX_ITER,
#         TOL=TOL,
#         BW_total=BW_total,
#         R_MIN=R_MIN,
#         D_m=D_m,
#         C_m=C_m,
#         f_UAV=f_UAV,
#         f_user=f_user,
#         device=device,
#         num_workers=2,
#         clustering_method='hierarchical'
#     )
    
#     # Extract results
#     sumrate_kmeans_ref_arr[i] = np.mean(kmeans_baseline)
#     sumrate_kmeans_opt_arr[i] = np.mean(kmeans_optimized)
#     sumrate_hier_ref_arr[i] = np.mean(hier_baseline)
#     sumrate_hier_opt_arr[i] = np.mean(hier_optimized)
    
#     elapsed = time.time() - start_time
#     print(f"  M={m} completed in {elapsed:.1f}s")
#     print(f"    K-means Ref: {sumrate_kmeans_ref_arr[i]:.2f} Mbps")
#     print(f"    K-means Opt: {sumrate_kmeans_opt_arr[i]:.2f} Mbps")
#     print(f"    Hier Ref: {sumrate_hier_ref_arr[i]:.2f} Mbps")
#     print(f"    Hier Opt: {sumrate_hier_opt_arr[i]:.2f} Mbps")

# total_elapsed = time.time() - start_time_total
# print(f"\nM Sweep completed in {total_elapsed:.1f}s")

# fig = plot_sweep(M_vals, sumrate_kmeans_ref_arr, sumrate_kmeans_opt_arr, 
#                  sumrate_hier_ref_arr, sumrate_hier_opt_arr, 
#                  'Number of Users (M)', 
#                  'Variation of the Sum Rate Relative to the Number of Users')
# plt.savefig('m_sweep_results.png', dpi=300, bbox_inches='tight')
# print("Saved plot: m_sweep_results.png")

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

# start_time_total = time.time()

# for i, bw in enumerate(BW_vals):
#     print(f"\nBW = {bw/1e6:.1f} MHz ({i+1}/{len(BW_vals)})")
#     start_time = time.time()
    
#     # Run K-means trials in parallel
#     kmeans_baseline, kmeans_optimized = batch_optimize_trials(
#         user_pos_trials=user_pos_trials,
#         M=M,
#         N=N,
#         AREA=AREA,
#         H_M=H_M,
#         H=H,
#         F=F,
#         P_T=P_T,
#         P_N=P_N,
#         MAX_ITER=MAX_ITER,
#         TOL=TOL,
#         BW_total=bw,
#         R_MIN=R_MIN,
#         D_m=D_m,
#         C_m=C_m,
#         f_UAV=f_UAV,
#         f_user=f_user,
#         device=device,
#         num_workers=2,
#         clustering_method='kmeans'
#     )
    
#     # Run hierarchical trials in parallel
#     hier_baseline, hier_optimized = batch_optimize_trials(
#         user_pos_trials=user_pos_trials,
#         M=M,
#         N=N,
#         AREA=AREA,
#         H_M=H_M,
#         H=H,
#         F=F,
#         P_T=P_T,
#         P_N=P_N,
#         MAX_ITER=MAX_ITER,
#         TOL=TOL,
#         BW_total=bw,
#         R_MIN=R_MIN,
#         D_m=D_m,
#         C_m=C_m,
#         f_UAV=f_UAV,
#         f_user=f_user,
#         device=device,
#         num_workers=2,
#         clustering_method='hierarchical'
#     )
    
#     # Extract results
#     sumrate_kmeans_ref_arr[i] = np.mean(kmeans_baseline)
#     sumrate_kmeans_opt_arr[i] = np.mean(kmeans_optimized)
#     sumrate_hier_ref_arr[i] = np.mean(hier_baseline)
#     sumrate_hier_opt_arr[i] = np.mean(hier_optimized)
    
#     elapsed = time.time() - start_time
#     print(f"  BW={bw/1e6:.1f}MHz completed in {elapsed:.1f}s")
#     print(f"    K-means Ref: {sumrate_kmeans_ref_arr[i]:.2f} Mbps")
#     print(f"    K-means Opt: {sumrate_kmeans_opt_arr[i]:.2f} Mbps")
#     print(f"    Hier Ref: {sumrate_hier_ref_arr[i]:.2f} Mbps")
#     print(f"    Hier Opt: {sumrate_hier_opt_arr[i]:.2f} Mbps")

# total_elapsed = time.time() - start_time_total
# print(f"\nBW Sweep completed in {total_elapsed:.1f}s")

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

# start_time_total = time.time()

# for i, p_t in enumerate(P_t_vals):
#     print(f"\nP_T = {p_t} dBm ({i+1}/{len(P_t_vals)})")
#     start_time = time.time()
    
#     # Run K-means trials in parallel
#     kmeans_baseline, kmeans_optimized = batch_optimize_trials(
#         user_pos_trials=user_pos_trials,
#         M=M,
#         N=N,
#         AREA=AREA,
#         H_M=H_M,
#         H=H,
#         F=F,
#         P_T=p_t,
#         P_N=P_N,
#         MAX_ITER=MAX_ITER,
#         TOL=TOL,
#         BW_total=BW_total,
#         R_MIN=R_MIN,
#         D_m=D_m,
#         C_m=C_m,
#         f_UAV=f_UAV,
#         f_user=f_user,
#         device=device,
#         num_workers=2,
#         clustering_method='kmeans'
#     )
    
#     # Run hierarchical trials in parallel
#     hier_baseline, hier_optimized = batch_optimize_trials(
#         user_pos_trials=user_pos_trials,
#         M=M,
#         N=N,
#         AREA=AREA,
#         H_M=H_M,
#         H=H,
#         F=F,
#         P_T=p_t,
#         P_N=P_N,
#         MAX_ITER=MAX_ITER,
#         TOL=TOL,
#         BW_total=BW_total,
#         R_MIN=R_MIN,
#         D_m=D_m,
#         C_m=C_m,
#         f_UAV=f_UAV,
#         f_user=f_user,
#         device=device,
#         num_workers=2,
#         clustering_method='hierarchical'
#     )
    
#     # Extract results
#     sumrate_kmeans_ref_arr[i] = np.mean(kmeans_baseline)
#     sumrate_kmeans_opt_arr[i] = np.mean(kmeans_optimized)
#     sumrate_hier_ref_arr[i] = np.mean(hier_baseline)
#     sumrate_hier_opt_arr[i] = np.mean(hier_optimized)
    
#     elapsed = time.time() - start_time
#     print(f"  P_T={p_t}dBm completed in {elapsed:.1f}s")
#     print(f"    K-means Ref: {sumrate_kmeans_ref_arr[i]:.2f} Mbps")
#     print(f"    K-means Opt: {sumrate_kmeans_opt_arr[i]:.2f} Mbps")
#     print(f"    Hier Ref: {sumrate_hier_ref_arr[i]:.2f} Mbps")
#     print(f"    Hier Opt: {sumrate_hier_opt_arr[i]:.2f} Mbps")

# total_elapsed = time.time() - start_time_total
# print(f"\nP_T Sweep completed in {total_elapsed:.1f}s")

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

# start_time_total = time.time()

# for i, rmin in enumerate(Rmin_vals):
#     print(f"\nR_MIN = {rmin/1e3:.1f} kbps ({i+1}/{len(Rmin_vals)})")
#     start_time = time.time()
    
#     # Run K-means trials in parallel
#     kmeans_baseline, kmeans_optimized = batch_optimize_trials(
#         user_pos_trials=user_pos_trials,
#         M=M,
#         N=N,
#         AREA=AREA,
#         H_M=H_M,
#         H=H,
#         F=F,
#         P_T=P_T,
#         P_N=P_N,
#         MAX_ITER=MAX_ITER,
#         TOL=TOL,
#         BW_total=BW_total,
#         R_MIN=rmin,
#         D_m=D_m,
#         C_m=C_m,
#         f_UAV=f_UAV,
#         f_user=f_user,
#         device=device,
#         num_workers=2,
#         clustering_method='kmeans'
#     )
    
#     # Run hierarchical trials in parallel
#     hier_baseline, hier_optimized = batch_optimize_trials(
#         user_pos_trials=user_pos_trials,
#         M=M,
#         N=N,
#         AREA=AREA,
#         H_M=H_M,
#         H=H,
#         F=F,
#         P_T=P_T,
#         P_N=P_N,
#         MAX_ITER=MAX_ITER,
#         TOL=TOL,
#         BW_total=BW_total,
#         R_MIN=rmin,
#         D_m=D_m,
#         C_m=C_m,
#         f_UAV=f_UAV,
#         f_user=f_user,
#         device=device,
#         num_workers=2,
#         clustering_method='hierarchical'
#     )
    
#     # Extract results
#     sumrate_kmeans_ref_arr[i] = np.mean(kmeans_baseline)
#     sumrate_kmeans_opt_arr[i] = np.mean(kmeans_optimized)
#     sumrate_hier_ref_arr[i] = np.mean(hier_baseline)
#     sumrate_hier_opt_arr[i] = np.mean(hier_optimized)
    
#     elapsed = time.time() - start_time
#     print(f"  R_MIN={rmin/1e3:.1f}kbps completed in {elapsed:.1f}s")
#     print(f"    K-means Ref: {sumrate_kmeans_ref_arr[i]:.2f} Mbps")
#     print(f"    K-means Opt: {sumrate_kmeans_opt_arr[i]:.2f} Mbps")
#     print(f"    Hier Ref: {sumrate_hier_ref_arr[i]:.2f} Mbps")
#     print(f"    Hier Opt: {sumrate_hier_opt_arr[i]:.2f} Mbps")

# total_elapsed = time.time() - start_time_total
# print(f"\nR_MIN Sweep completed in {total_elapsed:.1f}s")

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

# start_time_total = time.time()

# for i, area in enumerate(Area_vals):
#     print(f"\nArea = {area/1e6:.1f} km² ({i+1}/{len(Area_vals)})")
#     start_time = time.time()
#     side_area = np.sqrt(area)
    
#     # Generate new user positions for this area
#     user_pos_trials_area = []
#     for j in range(TRIALS):
#         torch.manual_seed(j)
#         np.random.seed(j)
#         user_pos = side_area * torch.rand(2, M, device=device)
#         user_pos_trials_area.append(user_pos)
    
#     # Run K-means trials in parallel
#     kmeans_baseline, kmeans_optimized = batch_optimize_trials(
#         user_pos_trials=user_pos_trials_area,
#         M=M,
#         N=N,
#         AREA=area,
#         H_M=H_M,
#         H=H,
#         F=F,
#         P_T=P_T,
#         P_N=P_N,
#         MAX_ITER=MAX_ITER,
#         TOL=TOL,
#         BW_total=BW_total,
#         R_MIN=R_MIN,
#         D_m=D_m,
#         C_m=C_m,
#         f_UAV=f_UAV,
#         f_user=f_user,
#         device=device,
#         num_workers=2,
#         clustering_method='kmeans'
#     )
    
#     # Run hierarchical trials in parallel
#     hier_baseline, hier_optimized = batch_optimize_trials(
#         user_pos_trials=user_pos_trials_area,
#         M=M,
#         N=N,
#         AREA=area,
#         H_M=H_M,
#         H=H,
#         F=F,
#         P_T=P_T,
#         P_N=P_N,
#         MAX_ITER=MAX_ITER,
#         TOL=TOL,
#         BW_total=BW_total,
#         R_MIN=R_MIN,
#         D_m=D_m,
#         C_m=C_m,
#         f_UAV=f_UAV,
#         f_user=f_user,
#         device=device,
#         num_workers=2,
#         clustering_method='hierarchical'
#     )
    
#     # Extract results
#     sumrate_kmeans_ref_arr[i] = np.mean(kmeans_baseline)
#     sumrate_kmeans_opt_arr[i] = np.mean(kmeans_optimized)
#     sumrate_hier_ref_arr[i] = np.mean(hier_baseline)
#     sumrate_hier_opt_arr[i] = np.mean(hier_optimized)
    
#     elapsed = time.time() - start_time
#     print(f"  Area={area/1e6:.1f}km² completed in {elapsed:.1f}s")
#     print(f"    K-means Ref: {sumrate_kmeans_ref_arr[i]:.2f} Mbps")
#     print(f"    K-means Opt: {sumrate_kmeans_opt_arr[i]:.2f} Mbps")
#     print(f"    Hier Ref: {sumrate_hier_ref_arr[i]:.2f} Mbps")
#     print(f"    Hier Opt: {sumrate_hier_opt_arr[i]:.2f} Mbps")

# total_elapsed = time.time() - start_time_total
# print(f"\nArea Sweep completed in {total_elapsed:.1f}s")

# fig = plot_sweep(Area_vals/1e6, sumrate_kmeans_ref_arr, sumrate_kmeans_opt_arr, 
#                  sumrate_hier_ref_arr, sumrate_hier_opt_arr, 
#                  'Area (km²)', 
#                  'Variation of the Sum Rate Relative to the Area')
# plt.savefig('area_sweep_results.png', dpi=300, bbox_inches='tight')
# print("Saved plot: area_sweep_results.png")

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