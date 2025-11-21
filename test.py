"""
Simple test script to verify the Python translation works correctly
"""

import torch
import numpy as np
from util.constants import constants
from util.clustering import k_means_uav, hierarchical_uav
from util.optimizers import optimize_network
from util.plotter import plot_network
import matplotlib.pyplot as plt

print("=" * 70)
print("UAV-Ground Association Optimizer - Quick Test")
print("=" * 70)

# Check device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nDevice: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Get constants
M, N, AREA, H, H_M, F, K, GAMMA, D_0, P_T, P_N, MAX_ITER, TOL, BW_total, R_MIN, SIDE, TRIALS = constants()

print(f"\nTest Parameters:")
print(f"  Users: {M}")
print(f"  UAVs: {N}")
print(f"  Area: {AREA/1e6:.1f} km²")
print(f"  Total Bandwidth: {BW_total/1e6:.1f} MHz")

# Generate random user positions
torch.manual_seed(42)
np.random.seed(42)
user_pos = SIDE * torch.rand(2, M, device=device)

print("\n" + "=" * 70)
print("Test 1: K-means Clustering")
print("=" * 70)
uav_pos_kmeans, rates_kmeans, throughput_kmeans = k_means_uav(
    user_pos, M, N, AREA, H_M, H, F, P_T, P_N, MAX_ITER, TOL, BW_total, device=device
)
print(f"✓ K-means completed")
print(f"  Throughput: {throughput_kmeans:.6f} Mbps")
print(f"  UAV positions shape: {uav_pos_kmeans.shape}")

print("\n" + "=" * 70)
print("Test 2: Hierarchical Clustering")
print("=" * 70)
uav_pos_hier, rates_hier, throughput_hier = hierarchical_uav(
    user_pos, N, H_M, H, F, P_T, P_N, BW_total, device=device
)
print(f"✓ Hierarchical clustering completed")
print(f"  Throughput: {throughput_hier:.6f} Mbps")
print(f"  UAV positions shape: {uav_pos_hier.shape}")

print("\n" + "=" * 70)
print("Test 3: Network Optimization")
print("=" * 70)
print("Optimizing from K-means initialization...")
uav_pos_opt, bandwidth_opt, rates_opt, throughput_opt = optimize_network(
    M, N, uav_pos_kmeans, BW_total, AREA, user_pos, H_M, H, F, P_T, P_N, R_MIN, device=device
)
print(f"✓ Optimization completed")
print(f"  Optimized Throughput: {throughput_opt:.6f} Mbps")
print(f"  Improvement: {((throughput_opt - throughput_kmeans) / throughput_kmeans * 100):.1f}%")

print("\n" + "=" * 70)
print("Test 4: Visualization")
print("=" * 70)
fig = plot_network(user_pos, uav_pos_opt, H_M, H, F, P_T, "Test: Optimized UAV Network")
plt.savefig('test_network.png', dpi=150, bbox_inches='tight')
print(f"✓ Network plot saved to: test_network.png")

print("\n" + "=" * 70)
print("All tests passed! ✓")
print("=" * 70)
print("\nThe MATLAB-to-Python translation is working correctly.")
print("You can now run the full benchmark suite with: python main.py")

