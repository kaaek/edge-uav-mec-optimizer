"""
CPU parallel benchmark - tests different num_workers to find optimal parallelization.
Measures actual runtime with different worker counts to determine CPU bottleneck.
"""
import torch
import numpy as np
import time
import psutil
from util.constants import constants
from util.optimizers.batch_optimizer import batch_optimize_trials

# Get constants
M, N, AREA, H, H_M, F, K, GAMMA, D_0, P_T, P_N, MAX_ITER, TOL, BW_total, R_MIN, SIDE, TRIALS, D_m, C_m, f_UAV, f_user = constants()

# Check CUDA
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("=" * 70)
print("CPU Parallelization Benchmark")
print("=" * 70)
print(f"Device: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Get CPU info
cpu_count_physical = psutil.cpu_count(logical=False)
cpu_count_logical = psutil.cpu_count(logical=True)
print(f"Physical CPU cores: {cpu_count_physical}")
print(f"Logical CPU cores (threads): {cpu_count_logical}")
print(f"Available RAM: {psutil.virtual_memory().available / 1024**3:.2f} GB")
print("=" * 70)

# Generate test trials (small number for quick benchmark)
num_test_trials = 10
print(f"\nGenerating {num_test_trials} test trials...")
user_pos_trials = []
for j in range(num_test_trials):
    torch.manual_seed(j)
    np.random.seed(j)
    user_pos = SIDE * torch.rand(2, M, device=device)
    user_pos_trials.append(user_pos)

# Test different worker counts
worker_counts = [1, 2, 4, 6, 8, 12, 16]
# Filter out worker counts that exceed logical cores
worker_counts = [w for w in worker_counts if w <= cpu_count_logical]

results = {}

print(f"\nRunning benchmark with {num_test_trials} trials each...")
print("=" * 70)

for num_workers in worker_counts:
    print(f"\nTesting with {num_workers} worker(s)...")
    
    # Measure CPU usage before
    cpu_before = psutil.cpu_percent(interval=0.1)
    
    # Run K-means trials
    start_time = time.time()
    baseline, optimized = batch_optimize_trials(
        user_pos_trials=user_pos_trials,
        M=M,
        N=N,
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
        num_workers=num_workers,
        clustering_method='kmeans'
    )
    elapsed = time.time() - start_time
    
    # Measure CPU usage after
    cpu_after = psutil.cpu_percent(interval=0.1)
    
    # Calculate metrics
    trials_per_second = num_test_trials / elapsed
    speedup = results.get(1, {}).get('elapsed', elapsed) / elapsed if 1 in results else 1.0
    efficiency = speedup / num_workers * 100
    
    results[num_workers] = {
        'elapsed': elapsed,
        'trials_per_second': trials_per_second,
        'speedup': speedup,
        'efficiency': efficiency,
        'cpu_usage': cpu_after
    }
    
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Throughput: {trials_per_second:.2f} trials/sec")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Efficiency: {efficiency:.1f}%")
    print(f"  CPU usage: {cpu_after:.1f}%")

# Summary
print("\n" + "=" * 70)
print("Benchmark Summary")
print("=" * 70)
print(f"{'Workers':<10} {'Time (s)':<12} {'Speedup':<10} {'Efficiency':<12} {'Trials/sec':<12}")
print("-" * 70)

for num_workers in worker_counts:
    r = results[num_workers]
    print(f"{num_workers:<10} {r['elapsed']:<12.2f} {r['speedup']:<10.2f}x {r['efficiency']:<12.1f}% {r['trials_per_second']:<12.2f}")

# Find optimal
best_workers = max(results.keys(), key=lambda w: results[w]['trials_per_second'])
best_efficiency = max([w for w in results.keys() if results[w]['efficiency'] >= 80], 
                     key=lambda w: results[w]['speedup'], default=2)

print("\n" + "=" * 70)
print("Recommendations")
print("=" * 70)
print(f"Fastest configuration: {best_workers} workers ({results[best_workers]['trials_per_second']:.2f} trials/sec)")
print(f"Best efficiency (≥80%): {best_efficiency} workers ({results[best_efficiency]['speedup']:.2f}x speedup)")
print(f"\nFor main.py benchmarks (50 trials per parameter):")
print(f"  Recommended: num_workers={best_efficiency}")
print(f"  Maximum performance: num_workers={best_workers}")
print("\nNote: Higher worker counts may cause diminishing returns due to:")
print("  - Thread management overhead")
print("  - Memory bandwidth saturation")
print("  - CPU cache contention")
