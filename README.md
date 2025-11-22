# UAV-Ground Association Optimizer with Multi-Access Edge Computing (MEC)

**Authors:** Khalil El Kaaki & Joe Abi Samra  

## Overview

This is a GPU-accelerated Python implementation of UAV (Unmanned Aerial Vehicle) positioning and resource allocation optimization for Multi-Access Edge Computing (MEC) networks. The system jointly optimizes UAV positions, bandwidth allocation, and task offloading decisions to maximize network throughput while ensuring Quality of Service (QoS) constraints.

### Key Features

- **GPU Acceleration:** PyTorch-based tensor operations with CUDA support
- **MEC Support:** Fractional task offloading with computation at UAVs or locally
- **Joint Optimization:** Simultaneous optimization of positions, bandwidth, and offloading
- **Multiple Algorithms:** K-means clustering, hierarchical clustering, and gradient-based optimization
- **Runtime Optimizations:** Parallel trial processing, early stopping, adaptive convergence
- **Comprehensive Benchmarking:** Parameter sweeps with 50 Monte Carlo trials
- **CPU/GPU Flexibility:** Automatic fallback to CPU when CUDA unavailable

## Installation

### Prerequisites

- **Python 3.8 or higher**
- **CUDA-capable NVIDIA GPU** (optional but highly recommended for performance)
- **CUDA Toolkit 11.0+** (for GPU acceleration)

### CUDA Toolkit Installation

To enable GPU acceleration, install the NVIDIA CUDA Toolkit:

1. **Download CUDA Toolkit** from NVIDIA:
   - Visit: https://developer.nvidia.com/cuda-downloads
   - Select your operating system and follow installation instructions
   - Recommended version: CUDA 11.8 or newer (compatible with PyTorch)

2. **Verify CUDA Installation:**
   ```bash
   nvcc --version
   nvidia-smi
   ```

   You should see your CUDA version and GPU information.

### Python Environment Setup

1. **Clone or download this repository:**
   ```bash
   cd edge-uav-mec-optimizer
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install PyTorch with CUDA support:**
   
   For **CUDA 11.8**:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

   For **CUDA 12.1+**:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

   For **CPU-only** (no GPU):
   ```bash
   pip install torch torchvision torchaudio
   ```

4. **Verify GPU Detection:**
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
   ```

   Example output (if GPU available):
   ```
   CUDA available: True
   GPU: NVIDIA GeForce RTX 3050 Laptop GPU
   ```

### Additional Tools (Optional)

For performance monitoring and CPU parallelization benchmarking:
```bash
pip install psutil
```

## Project Structure

```
edge-uav-mec-optimizer/
├── main.py                         # Main benchmarking script (parameter sweeps)
├── test.py                         # Quick verification script
├── vram_test.py                    # GPU VRAM usage profiler
├── cpu_parallel_test.py            # CPU parallelization benchmark
├── requirements.txt                # Python dependencies
├── README.md                       # This comprehensive documentation
└── util/
    ├── constants.py                # System parameters (wireless + MEC)
    ├── benchmark_vals.py           # Parameter sweep values
    ├── common/
    │   └── __init__.py            # Core functions (propagation, association, MEC throughput)
    ├── clustering/
    │   └── __init__.py            # UAV placement algorithms (K-means, Hierarchical)
    ├── optimizers/
    │   ├── __init__.py            # Joint optimization (positions + BW + offloading)
    │   ├── batch_optimizer.py     # Parallel trial processing
    │   └── helpers/
    │       └── __init__.py        # Objective functions and constraints
    └── plotter/
        └── __init__.py            # Visualization tools
```

## Multi-Access Edge Computing (MEC) Problem Formulation

### Decision Variables

The optimization jointly determines:

```
x = [uav_positions(2×N), bandwidth(M), offloading_fractions(M)]
```

- **UAV Positions** (2N variables): `(x_n, y_n)` coordinates for each UAV `n ∈ {1,...,N}`
- **Bandwidth Allocation** (M variables): `b_m` Hz assigned to user `m ∈ {1,...,M}`
- **Offloading Fractions** (M variables): `o_m ∈ [0,1]` fraction of task offloaded to UAV

**Total:** `2N + 2M` decision variables

### MEC Throughput Model

For each user `m`, the effective task completion throughput is:

```
Th_m = o_m × D_m / (T_ul + T_comp + T_dl) + (1 - o_m) × D_m / T_local
```

Where:
- **T_ul = D_m / R_m**: Uplink transmission time (seconds)
- **T_comp = C_m / f_UAV**: Computation time at UAV (seconds)  
- **T_dl = D_m / R_m**: Downlink transmission time (symmetric link assumed)
- **T_local = C_m / f_user**: Local computation time (seconds)
- **R_m = b_m × SE_m**: Wireless data rate (bps), where SE_m is spectral efficiency

**Interpretation:** Fractional offloading splits the task between UAV processing (fraction `o_m`) and local processing (fraction `1 - o_m`).

### Objective Function

Maximize proportional fairness:

```
maximize  Σ_{m=1}^M log(Th_m)
```

This ensures fairness across users while maximizing total throughput.

### Constraints

1. **QoS Constraints** (M constraints):
   ```
   Th_m ≥ R_min    ∀m ∈ {1,...,M}
   ```
   Each user must achieve minimum throughput.

2. **Bandwidth Budget** (1 constraint):
   ```
   Σ_{m=1}^M b_m ≤ BW_total
   ```
   Total allocated bandwidth cannot exceed available spectrum.

3. **UAV CPU Capacity** (N constraints):
   ```
   Σ_{m∈A_n} (o_m × C_m / f_UAV) ≤ 1    ∀n ∈ {1,...,N}
   ```
   Where `A_n` is the set of users associated with UAV `n`. Each UAV's CPU load cannot exceed 100%.

### Variable Bounds

- **UAV positions**: `(x_n, y_n) ∈ [0, SIDE]²` (coverage area)
- **Bandwidth**: `b_m ∈ [0, BW_total]` Hz
- **Offloading**: `o_m ∈ [0, 1]` (fractional offloading allowed)

## Usage

### Running the Main Benchmark

```bash
python main.py
```

This will:
1. Detect available GPU (CUDA) or fall back to CPU
2. Run N-sweep (varying number of UAVs) with 50 Monte Carlo trials
3. Compare K-means vs Hierarchical clustering (baseline and optimized)
4. Generate comparison plots
5. Save results to `n_sweep_results.png`

**Note:** GPU acceleration provides 10-100× speedup. CPU execution takes significantly longer (~24 minutes vs 2-3 minutes on GPU for default settings).

### Example: Custom Simulation

```python
import torch
from util.constants import constants
from util.clustering import k_means_uav, hierarchical_uav
from util.optimizers import optimize_network
from util.plotter import plot_network

# Get system parameters (including MEC parameters)
M, N, AREA, H, H_M, F, K, GAMMA, D_0, P_T, P_N, MAX_ITER, TOL, \
    BW_total, R_MIN, SIDE, TRIALS, D_m, C_m, f_UAV, f_user = constants()

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Generate random user positions
user_pos = SIDE * torch.rand(2, M, device=device)

# Method 1: K-means clustering
uav_pos_kmeans, rates_kmeans, throughput_kmeans = k_means_uav(
    user_pos, M, N, AREA, H_M, H, F, P_T, P_N, MAX_ITER, TOL, BW_total, device=device
)
print(f"K-means throughput: {throughput_kmeans:.6f} Mbps")

# Method 2: MEC-aware optimization from k-means initialization
uav_pos_opt, bandwidth_opt, offload_opt, throughput_opt_arr, throughput_opt = optimize_network(
    M, N, uav_pos_kmeans, BW_total, AREA, user_pos, H_M, H, F, P_T, P_N, R_MIN,
    D_m, C_m, f_UAV, f_user,  # MEC parameters
    device=device
)
print(f"MEC-optimized throughput: {throughput_opt:.6f} Mbps")
print(f"Average offloading fraction: {offload_opt.mean():.3f}")

# Visualize network
plot_network(user_pos, uav_pos_opt, H_M, H, F, P_T, "Optimized UAV-MEC Network")
```

## Core Modules

### Common Utilities (`util/common/`)

- **`p_received(user_pos, uav_pos, H_M, H, F, P_T, device='cuda')`**
  - Calculates received power using Okumura-Hata propagation model
  - Returns: (M, N) tensor of received power in dBm

- **`association(p_r)`**
  - Associates each user with the UAV providing maximum received power
  - Returns: (M, N) binary one-hot association matrix

- **`bitrate(P_R, P_N, BW, ASSOCIATION_MATRIX)`**
  - Calculates achievable bitrates using Shannon capacity
  - Returns: (M, N) tensor of bitrates in bps

- **`compute_mec_throughput(R_m, o_m, D_m, C_m, f_UAV, f_user, device='cuda')`**
  - Computes MEC throughput with fractional offloading
  - Th_m = o_m × D_m / (T_ul + T_comp + T_dl) + (1 - o_m) × D_m / T_local
  - Returns: (M,) tensor of MEC throughputs in bps

- **`se(P_R, P_N, ASSOCIATION_MATRIX)`**
  - Calculates spectral efficiency (bps/Hz)
  - Returns: (M, N) tensor

### Clustering Algorithms (`util/clustering/`)

- **`k_means(user_pos, N, AREA, MAX_ITER, TOL, device='cuda')`**
  - GPU-accelerated K-means clustering for UAV positioning
  - Returns: (2, N) tensor of UAV positions

- **`k_means_uav(user_pos, M, N, AREA, H_M, H, F, P_T, P_N, MAX_ITER, TOL, BW, device='cuda')`**
  - K-means with rate calculation
  - Returns: uav_pos, rates, throughput

- **`hierarchical_uav(user_pos, N, H_M, H, F, P_T, P_N, BW, device='cuda')`**
  - Hierarchical clustering (Ward's method)
  - Returns: uav_pos, rates, throughput

### Optimization (`util/optimizers/`)

- **`optimize_network(M, N, INITIAL_UAV_POS, BW_total, AREA, user_pos, H_M, H, F, P_T, P_N, Rmin, D_m, C_m, f_UAV, f_user, device='cuda')`**
  - Joint optimization of UAV positions, bandwidth allocation, and task offloading (MEC-aware)
  - Uses proportional fairness objective: maximize Σ log(Th_m)
  - Constraints: QoS (Th_m ≥ Rmin), bandwidth budget, CPU capacity per UAV
  - Returns: optimized UAV positions, bandwidth allocation, offloading fractions, throughputs, sum throughput

### Plotting (`util/plotter/`)

- **`plot_network(user_pos, uav_pos, H_M, H, F, P_T, fig_title)`**
  - Visualizes network layout with association lines

- **`plot_sweep(x, sumrate_kmeans_ref, sumrate_kmeans_opt, sumrate_hier_ref, sumrate_hier_opt, xlabel, title)`**
  - Plots benchmark sweep results

## System Parameters

Default parameters (defined in `util/constants.py`):

### Wireless System

| Parameter | Value | Description |
|-----------|-------|-------------|
| M | 50 | Number of ground users |
| N | 5 | Number of UAVs |
| AREA | 9×10⁶ m² | Coverage area (3000×3000 m) |
| H | 100 m | UAV height |
| H_M | 1.5 m | Mobile user height |
| F | 500 MHz | Carrier frequency |
| P_T | 30 dBm | Transmit power |
| P_N | -91 dBm | Noise power |
| BW_total | 40 MHz | Total bandwidth |
| R_MIN | 0.2 Mbps | Minimum QoS throughput |
| TRIALS | 50 | Monte Carlo trials |

### MEC Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| D_m | 5×10⁶ bits | Task data size (5 Mbit) |
| C_m | 1×10⁹ cycles | Computational complexity (1 Gcycles) |
| f_UAV | 10×10⁹ Hz | UAV CPU frequency (10 GHz) |
| f_user | 1×10⁹ Hz | User CPU frequency (1 GHz) |

## Benchmark Sweeps

The main script supports sweeps over:
- **N** (Number of UAVs): [1, 2, 6, 10, 14, 18, 24, 30]
- **M** (Number of Users): [20, 50, 100, 200, 500, 700]
- **BW** (Bandwidth): [20, 40, 80, 160] MHz
- **P_T** (Transmit Power): [20, 30, 40] dBm
- **R_MIN** (QoS Requirement): [50, 200, 1000] kbps
- **AREA**: [1, 4, 9, 16] ×10⁶ m²

Uncomment the relevant sections in `main.py` to run the sweep you want.

## GPU Acceleration and Runtime Optimization

The implementation uses a **hybrid CPU-GPU architecture**:

```
[CPU] SciPy SLSQP Optimizer
  │
  ├─→ [CPU→GPU] Transfer decision vector x
  ├─→ [GPU] Compute MEC throughput Th_m = f(x)
  ├─→ [GPU] Evaluate constraints c(x)
  ├─→ [GPU→CPU] Return objective & constraint values
  └─→ [CPU] Update decision vector and repeat
```

**Why This Design?**
- **SciPy optimization** (SLSQP) runs on CPU (not GPU-accelerated).
- **Function evaluations** (throughput, constraints) use PyTorch tensors on GPU.
- **Result:** GPU accelerates the expensive function evaluations.

### Implemented Optimizations

#### 1. Early Stopping Callback
Terminates optimization when improvement drops below 0.01% after 3+ iterations

```python
def callback(xk):
    improvement = abs(prev_obj - current_obj) / max(abs(prev_obj), 1e-9)
    if improvement < 1e-4 and iter_count > 3:
        return True  # Stop early
```

#### 2. Relaxed Convergence Tolerance
**Change:** `ftol: 1e-6 → 1e-4` (0.01% tolerance sufficient for MEC throughput)

#### 3. Parallel Trial Processing
**File:** `util/optimizers/batch_optimizer.py`

```python
with ThreadPoolExecutor(max_workers=num_workers) as executor:
    futures = {executor.submit(process_trial, idx, user_pos): idx 
               for idx, user_pos in enumerate(user_pos_trials)}
```

**Why Threading Works:**
- SciPy releases Python's GIL during optimization
- Multiple CPU threads can run optimization simultaneously  
- PyTorch automatically synchronizes GPU access

**Recommended Workers:** 
- **2-4 workers**: Safe for most systems (GPU memory not a bottleneck)
- **6-8 workers**: If you have ≥6 CPU cores
- **The bottleneck** is the amount of CPU cores, not GPU VRAM (VRAM usage ~21 MB per trial)

#### 4. Silent Optimization Output
**Change:** `disp: True → False` in optimizer options

### Benchmarking Tools

**VRAM Usage Test:**
```bash
python vram_test.py
```
Measures peak GPU memory per operation and recommends parallel worker count.

**CPU Parallelization Test:**
```bash
python cpu_parallel_test.py
```
Benchmarks different `num_workers` values (1, 2, 4, 6, 8, 12, 16) to find optimal CPU parallelization. It is recommended you run this first and adjust the number of parallel workers accordingly.

**Real-Time GPU Monitoring (optional):**
```bash
nvidia-smi -l 1  # Updates every 1 second
```

### Performance Tuning

**To modify parallel workers in `main.py`:**
```python
# Default (conservative)
num_workers=2

# Higher performance (if you have 6+ CPU cores)
num_workers=4  # or 6, 8
```

## Differences from MATLAB Version

The original MATLAB implementation (see `UAV-Ground-Association-Optimizer/`) is now deprecated. This Python version offers:

**Technical Differences:**
- **Optimization:** Uses `scipy.optimize.minimize` (SLSQP) instead of MATLAB's `fmincon`
- **Clustering:** Uses `scipy.cluster.hierarchy` instead of MATLAB built-ins
- **Tensors:** PyTorch tensors (GPU) instead of MATLAB arrays (CPU)
- **Profiling:** Python profilers (`cProfile`, `line_profiler`) instead of MATLAB's profile viewer

## Baseline vs Optimized Comparison

### Baseline Algorithms

Both K-means and Hierarchical clustering baselines now compute **MEC throughput** (not just wireless capacity) using:

1. **UAV Positions:** Cluster centroids
2. **Bandwidth Allocation:** Uniform (`b_m = BW_total / M`)
3. **Offloading Decision:** Full offload (`o_m = 1.0`)

This ensures fair comparison with the optimizer, which also computes MEC throughput.

### When Optimizer Helps Most

The optimizer shows larger gains over baselines when:
- **Heterogeneous users:** Different QoS requirements → non-uniform bandwidth needed
- **CPU-constrained scenarios:** Limited UAV capacity → selective offloading critical
- **Poor initialization:** Random UAV placement → position optimization essential
- **Asymmetric distributions:** Non-uniform user locations → centroids suboptimal

For well-initialized problems with homogeneous users, the baseline clustering already finds near-optimal solutions.

## Troubleshooting

### GPU Issues

**CUDA not detected:**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
If `False`, verify:
1. NVIDIA GPU drivers installed
2. CUDA Toolkit installed (https://developer.nvidia.com/cuda-downloads)
3. PyTorch installed with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

**CUDA out of memory:**
- **Unlikely** (VRAM usage ~21 MB per trial)
- If occurs: Reduce `M` (number of users) or `N` (number of UAVs)
- Clear cache: `torch.cuda.empty_cache()`

**GPU not being utilized:**
- Check `main.py` output for "Using device: cuda"
- Monitor with `nvidia-smi` during execution
- Ensure `device='cuda'` in function calls

### Performance Issues

**Slow on CPU:**
- **Expected:** CPU is 10-100× slower than GPU
- **Solution:** Install CUDA toolkit and GPU-enabled PyTorch
- Alternatively: Reduce `TRIALS` from 50 to 10-20

**Slow even with GPU:**
- Run `cpu_parallel_test.py` to check optimal `num_workers`
- Default `num_workers=2` is conservative; try 4-6 if you have more CPU cores
- Verify GPU utilization with `nvidia-smi` (should be >50%)

### Installation Issues

**Import errors:**
```python
ModuleNotFoundError: No module named 'util.constants'
```
- Ensure all `__init__.py` files exist in subdirectories
- Run from project root: `cd edge-uav-mec-optimizer`

**PyTorch version mismatch:**
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Missing psutil:**
```bash
pip install psutil
```
Required only for `cpu_parallel_test.py`.

### Optimization Convergence

**Infeasible solution warnings:**
- Occurs when QoS constraints too strict for available resources
- **Fix:** Reduce `R_MIN`, increase `BW_total`, or increase `N` (more UAVs)

**Poor optimization results:**
- Verify initialization is reasonable (run baseline clustering first)
- Check constraint violations in output
- Try different random seeds for user positions

## Citation

If you use this code in your research, please cite:

```bibtex
@software{elkaaki2025uavmec,
  author = {El Kaaki, Khalil and Abi Samra, Joe},
  title = {Edge UAV-MEC Optimizer: GPU-Accelerated Multi-Access Edge Computing},
  year = {2025},
  month = {January},
  url = {https://github.com/kaaek/edge-uav-mec-optimizer}
}
```

## License

This project is provided as-is for educational and research purposes.

## Acknowledgments

- Original MATLAB implementation: October 2025
- MEC extension & Python GPU-accelerated version: November 2025

## Contact

For questions, issues, or contributions:
- **Khalil El Kaaki** - [GitHub: @kaaek](https://github.com/kaaek)
- **Joe Abi Samra**

**Repository:** https://github.com/kaaek/edge-uav-mec-optimizer
