"""
VRAM usage test script - runs a single trial to measure GPU memory consumption.
Run this with nvidia-smi monitoring to see peak VRAM usage.
"""
import torch
import numpy as np
from util.constants import constants
from util.clustering import k_means_uav, hierarchical_uav
from util.optimizers import optimize_network

# Get constants
M, N, AREA, H, H_M, F, K, GAMMA, D_0, P_T, P_N, MAX_ITER, TOL, BW_total, R_MIN, SIDE, TRIALS, D_m, C_m, f_UAV, f_user = constants()

# Check CUDA
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print()

# Generate single trial user positions
torch.manual_seed(42)
np.random.seed(42)
user_pos = SIDE * torch.rand(2, M, device=device)

print("=" * 70)
print("Running K-means baseline clustering...")
print("=" * 70)

# Reset VRAM stats
if device == 'cuda':
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

# K-means clustering
uav_pos_kmeans, _, sumrate_kmeans = k_means_uav(
    user_pos, M, N, AREA, H_M, H, F, P_T, P_N, MAX_ITER, TOL, BW_total,
    D_m, C_m, f_UAV, f_user, device=device
)

if device == 'cuda':
    kmeans_vram = torch.cuda.max_memory_allocated() / 1024**2
    print(f"K-means peak VRAM: {kmeans_vram:.2f} MB")
    print(f"K-means throughput: {sumrate_kmeans:.2f} Mbps")
    print()

print("=" * 70)
print("Running optimization from K-means initial position...")
print("=" * 70)

# Reset VRAM stats
if device == 'cuda':
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

# Optimization
uav_pos_opt, _, _, _, sumrate_opt = optimize_network(
    M, N, uav_pos_kmeans, BW_total, AREA, user_pos, H_M, H, F, P_T, P_N, R_MIN,
    D_m, C_m, f_UAV, f_user, device=device
)

if device == 'cuda':
    opt_vram = torch.cuda.max_memory_allocated() / 1024**2
    print(f"Optimization peak VRAM: {opt_vram:.2f} MB")
    print(f"Optimized throughput: {sumrate_opt:.2f} Mbps")
    print()

print("=" * 70)
print("Running Hierarchical baseline clustering...")
print("=" * 70)

# Reset VRAM stats
if device == 'cuda':
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

# Hierarchical clustering
uav_pos_hier, _, sumrate_hier = hierarchical_uav(
    user_pos, N, H_M, H, F, P_T, P_N, BW_total,
    D_m, C_m, f_UAV, f_user, device=device
)

if device == 'cuda':
    hier_vram = torch.cuda.max_memory_allocated() / 1024**2
    print(f"Hierarchical peak VRAM: {hier_vram:.2f} MB")
    print(f"Hierarchical throughput: {sumrate_hier:.2f} Mbps")
    print()

print("=" * 70)
print("VRAM Usage Summary")
print("=" * 70)
if device == 'cuda':
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**2
    max_single_trial = max(kmeans_vram, opt_vram, hier_vram)
    
    print(f"Total GPU VRAM: {total_vram:.2f} MB")
    print(f"K-means peak: {kmeans_vram:.2f} MB")
    print(f"Optimization peak: {opt_vram:.2f} MB")
    print(f"Hierarchical peak: {hier_vram:.2f} MB")
    print(f"Max per trial: {max_single_trial:.2f} MB")
    print()
    print(f"Available VRAM: {total_vram:.2f} MB")
    print(f"Recommended parallel trials (80% VRAM usage): {int((total_vram * 0.8) / max_single_trial)}")
    print(f"Maximum parallel trials (95% VRAM usage): {int((total_vram * 0.95) / max_single_trial)}")
    print()
    print("Note: Run 'nvidia-smi' in another terminal to monitor real-time VRAM usage")
else:
    print("CUDA not available - cannot measure VRAM usage")
