# UAV-Ground Association Optimizer (Python with GPU Acceleration)

**Authors:** Khalil El Kaaki & Joe Abi Samra  
**Date:** October 2025  
**Translated to Python:** November 2025

## Overview

This is a GPU-accelerated Python implementation of UAV (Unmanned Aerial Vehicle) positioning and bandwidth allocation optimization for wireless communication networks. The system maximizes network throughput while ensuring Quality of Service (QoS) constraints for ground users.

**Extended for Multi-Access Edge Computing (MEC):** This implementation now supports task offloading scenarios where users can choose to execute computational tasks locally or offload them to UAVs. See `MEC_EXTENSION.md` for details.

**Key Features:**
- GPU acceleration using PyTorch for fast tensor operations
- Faithful translation from MATLAB preserving all original logic
- Support for both CUDA and CPU execution
- Multiple UAV placement strategies (K-means, Hierarchical clustering)
- Joint optimization of UAV positions, bandwidth allocation, and task offloading
- MEC-aware throughput modeling with fractional offloading
- Comprehensive benchmarking suite with parameter sweeps

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended for performance)
- CUDA Toolkit 11.0+ and cuDNN (if using GPU)

### Setup

1. Clone or download this repository:
```bash
cd edge-uav-mec-optimizer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

For GPU support, ensure you have PyTorch with CUDA installed:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Project Structure

```
edge-uav-mec-optimizer/
├── main.py                         # Main benchmarking script
├── test.py                         # Quick verification script
├── requirements.txt                # Python library dependencies
├── README.md                       # This file
├── MEC_EXTENSION.md                # MEC extension documentation
├── TRANSLATION_SUMMARY.md          # Translation notes
└── util/
    ├── constants.py                # System constants (includes MEC parameters)
    ├── benchmark_vals.py           # Benchmark parameter values
    ├── common/
    │   └── __init__.py            # Common utilities (p_received, association, bitrate, compute_mec_throughput, etc.)
    ├── clustering/
    │   └── __init__.py            # Clustering algorithms (k_means, hierarchical)
    ├── optimizers/
    │   ├── __init__.py            # Main optimizers (optimize_network with MEC support, etc.)
    │   └── helpers/
    │       └── __init__.py        # Helper functions (constraints, objectives)
    └── plotter/
        └── __init__.py            # Plotting functions
```

## Usage

### Running the Main Benchmark

```bash
python main.py
```

This will:
1. Detect available GPU (CUDA) or fall back to CPU (It is a long calculation if on CPU).
2. Run parameter sweeps (default: N sweep - number of UAVs)
3. Generate plots comparing different algorithms
4. Save results to PNG files

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

- **`optimize_uav_positions(N, AREA, uav_pos, user_pos, H_M, H, F, P_T, P_N, BW, Rmin, device='cuda')`**
  - Optimizes only UAV positions (fixed bandwidth)
  - Uses hybrid objective: α·sum(rates) + (1-α)·sum(log(rates))
  - Returns: optimized UAV positions

- **`optimize_bandwidth_allocation(M, BW_total, user_pos, opt_uav_pos, H_M, H, F, P_T, P_N, Rmin, device='cuda')`**
  - Optimizes only bandwidth allocation (fixed UAV positions)
  - Returns: bandwidth allocation, rates, throughput

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

## GPU Acceleration

This implementation uses PyTorch tensors for all matrix operations, enabling:
- **Automatic GPU acceleration** when CUDA is available
- **Vectorized operations** for efficient batch processing
- **Mixed CPU-GPU execution** (optimization on CPU, function evaluations on GPU)

Performance tips:
- Ensure user positions and UAV positions are kept on GPU throughout computations
- Use `device='cuda'` parameter in all function calls
- For very large problems, consider batch processing

## Differences from MATLAB Version
Originally, this project was implemented on MATLAB. That version is now deprecated to allow for GPU acceleration.
While the Python version is a faithful translation, there are minor differences:
- **Optimization**: Uses `scipy.optimize.minimize` instead of MATLAB's `fmincon`
- **Hierarchical clustering**: Uses `scipy.cluster.hierarchy` instead of MATLAB's built-in functions
- **Global search**: Currently uses local optimization with good initialization (can be extended)
- **Profiling**: Use Python profilers instead of MATLAB's profile viewer

## Performance Considerations

- GPU acceleration provides **10-100x speedup** for large problems (M > 100, N > 10)
- CPU fallback works well for small problems
- Optimization steps are CPU-bound (scipy), but function evaluations are GPU-accelerated
- Each trial in the N-sweep takes approximately 5-30 seconds on modern GPUs

## Troubleshooting

**CUDA out of memory:**
- Reduce batch size or number of users
- Use `torch.cuda.empty_cache()` between trials

**Slow performance on CPU:**
- Install PyTorch with MKL for faster CPU operations
- Reduce number of trials

**Import errors:**
- Ensure all `__init__.py` files are in place
- Check that module paths are correct

## Citation

If you use this code in your research, please cite:

```
El Kaaki, K., & Abi Samra, J. (2025). 
Edge UAV-MEC Optimizer: GPU-Accelerated Python Implementation.
```

## License

This project is provided as-is for educational and research purposes.

## Contact

For questions or issues, please contact the authors:
- Khalil El Kaaki
- Joe Abi Samra
