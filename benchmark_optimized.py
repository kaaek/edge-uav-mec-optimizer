"""
GPU-Optimized Benchmarking Script
Author: GitHub Copilot
Date: January 2025

This script uses optimized runtime settings:
- Reduced optimizer iterations with early stopping
- Relaxed convergence tolerance (1e-4 instead of 1e-6)
- Parallel trial processing (2 threads)
- Silent optimization output
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from util.constants import constants
from util.benchmark_vals import benchmark_vals
from util.optimizers.batch_optimizer import batch_optimize_trials
from util.plotter import plot_sweep

# Check for CUDA availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("WARNING: CUDA not available. Running on CPU will be slow.")

# Get constants
M, N, AREA, H, H_M, F, K, GAMMA, D_0, P_T, P_N, MAX_ITER, TOL, BW_total, R_MIN, SIDE, TRIALS, D_m, C_m, f_UAV, f_user = constants()

# Get benchmark values
N_vals, M_vals, BW_vals, P_t_vals, Rmin_vals, Area_vals = benchmark_vals()

print("=" * 70)
print("GPU-Optimized UAV-MEC Benchmark Suite")
print("=" * 70)
print(f"System Parameters:")
print(f"  Users (M): {M}")
print(f"  UAVs (N): {N}")
print(f"  Area: {AREA/1e6:.1f} km²")
print(f"  Total Bandwidth: {BW_total/1e6:.1f} MHz")
print(f"  Trials per parameter: {TRIALS}")
print(f"  Parallel workers: 2 (recommended for GPU)")
print("=" * 70)

# Precompute user positions for all trials
print("\nPrecomputing user positions...")
user_pos_trials = []
for j in range(TRIALS):
    torch.manual_seed(j)
    np.random.seed(j)
    user_pos = SIDE * torch.rand(2, M, device=device)
    user_pos_trials.append(user_pos)
print(f"Generated {TRIALS} random user distributions")

# ========================================================================
# N Sweep (Number of UAVs) - OPTIMIZED
# ========================================================================
print("\n" + "=" * 70)
print("SWEEP 1: Number of UAVs (N) - GPU Optimized")
print("=" * 70)

sumrate_kmeans_ref_arr = np.zeros(len(N_vals))
sumrate_kmeans_opt_arr = np.zeros(len(N_vals))
sumrate_hier_ref_arr = np.zeros(len(N_vals))
sumrate_hier_opt_arr = np.zeros(len(N_vals))

start_time_total = time.time()

for i, n in enumerate(N_vals):
    print(f"\nN = {n} ({i+1}/{len(N_vals)})")
    start_time = time.time()
    
    # K-means with parallel trials
    kmeans_base, kmeans_opt = batch_optimize_trials(
        user_pos_trials, M, n, AREA, H_M, H, F, P_T, P_N, MAX_ITER, TOL,
        BW_total, R_MIN, D_m, C_m, f_UAV, f_user,
        clustering_method='kmeans', device=device, num_workers=2
    )
    
    # Hierarchical with parallel trials
    hier_base, hier_opt = batch_optimize_trials(
        user_pos_trials, M, n, AREA, H_M, H, F, P_T, P_N, MAX_ITER, TOL,
        BW_total, R_MIN, D_m, C_m, f_UAV, f_user,
        clustering_method='hierarchical', device=device, num_workers=2
    )
    
    # Compute averages
    sumrate_kmeans_ref_arr[i] = np.mean(kmeans_base)
    sumrate_kmeans_opt_arr[i] = np.mean(kmeans_opt)
    sumrate_hier_ref_arr[i] = np.mean(hier_base)
    sumrate_hier_opt_arr[i] = np.mean(hier_opt)
    
    elapsed = time.time() - start_time
    print(f"  Completed in {elapsed:.1f}s ({TRIALS/elapsed:.1f} trials/sec)")
    print(f"    K-means: {sumrate_kmeans_ref_arr[i]:.2f} → {sumrate_kmeans_opt_arr[i]:.2f} Mbps")
    print(f"    Hierarchical: {sumrate_hier_ref_arr[i]:.2f} → {sumrate_hier_opt_arr[i]:.2f} Mbps")

total_elapsed = time.time() - start_time_total
print(f"\nN Sweep completed in {total_elapsed:.1f}s")
print(f"Average: {total_elapsed/len(N_vals):.1f}s per parameter value")
print(f"Throughput: {len(N_vals)*TRIALS*2/total_elapsed:.1f} optimizations/sec")

# Plot N sweep results
fig = plot_sweep(N_vals, sumrate_kmeans_ref_arr, sumrate_kmeans_opt_arr, 
                 sumrate_hier_ref_arr, sumrate_hier_opt_arr, 
                 'Number of UAVs (N)', 
                 'MEC Throughput vs Number of UAVs')
plt.savefig('n_sweep_optimized.png', dpi=300, bbox_inches='tight')
print("\nSaved plot: n_sweep_optimized.png")

print("\n" + "=" * 70)
print("Benchmark completed!")
print("=" * 70)
print(f"\nPerformance Summary:")
print(f"  Total time: {total_elapsed:.1f}s")
print(f"  Total optimizations: {len(N_vals) * TRIALS * 2} (K-means + Hierarchical)")
print(f"  Average per optimization: {total_elapsed/(len(N_vals)*TRIALS*2):.2f}s")
print(f"  Device: {device}")

if device == 'cuda':
    print(f"\nGPU Memory Usage:")
    print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    print(f"  Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")

plt.show()
