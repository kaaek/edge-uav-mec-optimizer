# UAV-Assisted MEC Task Offloading Optimizer

**Authors:** Khalil El Kaaki & Joe Abi Samra  
**Course:** EECE 454 - Computer Network Modelling and Optimization  
**Date:** December 2025
**Research Paper:** A comprehensive research paper detailing the methodology, results, and analysis is included in [Mock Research Paper.pdf](Mock%20Research%20Paper.pdf).

## Overview

This project implements and evaluates UAV-assisted Mobile Edge Computing (MEC) for deadline-constrained IoT task offloading. The system maximizes the number of successfully completed tasks by optimizing UAV trajectory in real-time while managing offloading decisions under wireless channel uncertainty and computational resource constraints.

The main thread is gradient-based trajectory optimization using differentiable task completion objectives with sigmoid deadline relaxations and softmax offloading probabilities, enabling end-to-end optimization through automatic differentiation.

### System Model
- **Single UAV**: Provides mobile edge computing with 5 GHz CPU, 50m altitude, 20 m/s max velocity
- **Base Station**: Fixed at (700, 700) m with 4 GHz CPU, serves as fallback computation resource
- **IoT Devices**: M = 5 devices (configurable 2-20) with 1 GHz local CPUs, distributed in 50-350 m area
- **Task Model**: Poisson arrivals (λ = 0.5 tasks/s/device), 1.5-7 Mb data size, 500-1500 cycles/bit, 0.7-2.0 s deadlines
- **Channel Model**: Okumura-Hata path loss with 95% reliability requirement (P_min), TDMA medium access (15 MHz)
- **Time Horizon**: 70 time steps over 20 seconds (Δt ≈ 0.286 s)

### Key Features
- **Four-way Performance Comparison:**
  1. No UAV baseline (local + BS only)
  2. Circular trajectory baseline (fixed path)
  3. Online MPC optimization (receding horizon H=15, 20 iterations/step)
  4. Offline batch optimization (25 iterations global planning)
  
- **GPU-Accelerated Optimization:** PyTorch-based gradient descent with CUDA support
- **Constraints:** Velocity limits, TDMA queue management, and probabilistic channel reliability
- **Comprehensive Evaluation:** Three parameter sweeps (task size, network scalability, bandwidth), 5 Monte Carlo trials each
- **Visualization:** Trajectory plots, four-way comparison charts, sweep analysis with error bars

## Mathematical Formulation

**Objective:** Maximize total completed tasks
```
maximize Σ_k I_k^complete
```
where `I_k^complete = 1` if task k meets its deadline with sufficient channel reliability.

**Decision Variables:**
- UAV trajectory: `x(t) ∈ ℝ²` for t = 1...T
- Binary offloading: `δ_k ∈ {local, BS, UAV}` for each task k

**Constraints:**
- **Velocity bounds:** `||x(t) - x(t-1)|| ≤ v_max · Δt`
- **Deadline:** `T_serve^k + T_wait^TDMA ≤ slack_k`
- **Channel reliability:** `P_success(SNR) ≥ P_min = 0.95`
- **CPU capacity:** Per-time step computational limits at UAV/BS
- **TDMA scheduling:** Non-overlapping channel access

**Solution Approach:** Gradient-based optimization with:
- Sigmoid relaxation: `σ(deadline_margin / τ)` for deadline constraints
- Softmax formulation: Differentiable offloading decisions
- Adam optimizer: Automatic differentiation through PyTorch

See `docs/optimization_formulation.md` for complete MINLP formulation.

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

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kaaek/edge-uav-mec-optimizer.git
   cd edge-uav-mec-optimizer
   git checkout phase-2-maximization-of-completed-tasks-using-a-probabilistic-channel-model
   ```

2. **Navigate to the source directory:**
   ```bash
   cd src
   ```

4. **Install PyTorch with CUDA support** (choose based on your CUDA version):
   ```bash
   # CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   # CUDA 12.1+
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   # CPU only (slower)
   pip install torch torchvision torchaudio
   ```

5. **Verify GPU Detection:**
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
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

## Repository Structure
```
edge-uav-mec-optimizer/
├── Mock Research Paper.pdf              # Research paper submission
├── README.md                            # This file
├── src/                                 # Source code directory
│   ├── main.py                          # Main entry point - runs all sweeps
│   ├── requirements.txt                 # Python dependencies
│   └── util/                            # Core utilities package
│       ├── classes/                     # Core data models
│       │   ├── uav.py                   # UAV with trajectory and CPU
│       │   ├── base_station.py          # Fixed BS with CPU
│       │   ├── iot_device.py            # Device with Poisson task generation
│       │   ├── task.py                  # Task (size, density, deadline, arrival)
│       │   ├── constants.py             # System-wide parameters
│       │   └── benchmark_vals.py        # Benchmark configuration values
│       ├── schedulers/                  # Offloading and channel access
│       │   ├── offloading_decision.py   # Greedy TDMA-aware scheduler
│       │   └── tdma_scheduler.py        # TDMA queue management
│       ├── optimizers/                  # Trajectory optimization
│       │   ├── online_trajectory_optimizer.py   # MPC receding horizon (20 iter/step)
│       │   ├── trajectory_optimizer.py          # Offline batch (25 iterations)
│       │   ├── batch_optimizer.py       # Batch optimization utilities
│       │   └── helpers/                 # Optimization helper functions
│       ├── common/                      # Channel models
│       │   └── channel_reliability.py   # Okumura-Hata + sigmoid reliability
│       ├── plotter/                     # Visualization utilities
## Quick Start

### Run Complete Evaluation
Execute all parameter sweeps with four-way comparison (260 total simulations):
```bash
cd src
python main.py
```

**Expected runtime:** 10-30 minutes (GPU) or 1-3 hours (CPU)

**Outputs:**
- Console: Real-time progress with completion rates per trial
- `results/`: 15 plots (3 sweeps × 5 plot types)
  - `*_four_way_comparison.png`: Main performance comparison
  - `*_no_uav.png`, `*_circular.png`, `*_online.png`, `*_offline.png`: Individual method details
- Terminal summary: Average completion rates and UAV deployment benefit

### Configuration

Edit `BASE_CONFIG` in `src/main.py`: UAV deployment benefit

### Configuration

Edit `BASE_CONFIG` in `main_task_offloading.py`:

```python
BASE_CONFIG = {
    # Network topology
    'num_devices': 5,              # IoT devices (sweep: 2-20)
    'device_area_min': 50,         # Device distribution area (m)
    'device_area_max': 350,
    
    # UAV parameters
    'uav_cpu_frequency': 5e9,      # 5 GHz
    'uav_max_velocity': 20.0,      # m/s
    'uav_height': 50.0,            # meters
    
    # Base station
    'bs_position': [700.0, 700.0], # Far from devices
    'bs_cpu_frequency': 4e9,       # 4 GHz
    
    # Communication
    'BW_total': 15e6,              # 15 MHz (sweep: 5-60 MHz)
    'P_min': 0.95,                 # 95% reliability requirement
    
    # Task characteristics
    'task_size_min': 1.5e6,        # 1.5 Mb (sweep: 1-9 Mb)
    'task_size_max': 7e6,          # 7 Mb
    'slack_min': 0.7,              # Deadline slack (s)
    'slack_max': 2.0,
    
    # Simulation
    'trials': 5,                   # Monte Carlo trials
    'T': 70,                       # Time steps
    'duration': 20.0,              # Mission duration (s)
}
```

### Parameter Sweeps

Three sweeps are configured in `SWEEP_CONFIGS`:

1. **Task Size Impact** (3 points)
   - Ranges: [(1, 3), (2, 6), (3, 9)] Mb
   - Tests: Communication bottleneck vs computation tradeoff

2. **Network Scalability** (5 points)
   - Devices: {2, 5, 10, 15, 20}
   - Tests: TDMA contention and UAV coverage limits

3. **Bandwidth Availability** (5 points)
   - Values: {5, 10, 20, 40, 60} MHz
   - Tests: Communication resource impact

**Total simulations:** 13 points × 4 methods × 5 trials = **260 runs**

## Algorithm Details

### Optimization Methods

| Method | Description | Parameters |
|--------|-------------|------------|
| **No UAV** | Baseline without UAV deployment - tasks choose between local execution and distant base station | N/A |
| **Circular** | UAV follows fixed circular path (radius 200m, center [100, 100]) without optimization | N/A |
| **Online MPC** | Receding horizon optimization: re-plans next H=15 steps at each timestep using gradient descent | 20 iter/step, lr=2.0 |
| **Offline Batch** | Global trajectory optimization across all T=70 steps before mission starts | 25 iter total, lr=0.5 |

### Greedy Offloading Scheduler

For each arriving task, the scheduler:

1. **Estimates serving time** for each option (local/BS/UAV):
   - Local: `T_comp = (size × density) / f_local`
   - BS/UAV: `T_wait^TDMA + T_uplink + T_comp`

2. **Checks reliability constraint:**
   - Computes SNR from distance → channel success probability
   - Rejects option if `P_success < P_min = 0.95`

3. **Verifies deadline:**
   - Checks if `T_serve ≤ slack_remaining`
   - Uses sigmoid relaxation during optimization: `σ(slack - T_serve)`

4. **Selects best feasible option:**
   - Greedy: Choose fastest option meeting constraints
   - During optimization: Softmax over options for differentiability

### Trajectory Optimization

**Online (MPC):**
- At timestep t, optimize positions for t+1...t+H
- Execute first position, advance to t+1, re-optimize
- Objective: Maximize expected completed tasks in horizon window
- Urgency weighting: Tasks near deadline get higher priority

**Offline (Batch):**
- Optimize entire trajectory x(1)...x(T) before mission
- Objective: Maximize total expected completed tasks
- Velocity penalty ensures smooth, feasible paths

Both use **Adam optimizer** with **automatic differentiation** through PyTorch.

## Performance Metrics

**Primary:** Task Completion Rate (%)
- Percentage of generated tasks meeting deadlines with required reliability

**Secondary:**
- UAV task allocation: Fraction offloaded to UAV vs BS vs local
- Trajectory efficiency: Average distance to active devices
- Optimization benefit: Performance gain over circular baseline
- UAV deployment value: Improvement over no-UAV baseline

**Statistical:** Mean ± std error across 5 Monte Carlo trials

**Key Findings:**
- UAV deployment provides **~15-20% absolute improvement** over no-UAV baseline
- Trajectory optimization adds **marginal ~0.5-1% gain** over circular path
- UAV wins ~70% of tasks due to proximity advantage (BS positioned far away)
- Performance degrades gracefully with increasing device count due to TDMA contention

## Citation

If you use this work in your research, please cite:

```bibtex
@software{elkaakiabisamra2025uavmec,
  author = {El Kaaki, Khalil and Abi Samra, Joe},
  title = {UAV-Assisted MEC Task Offloading Optimizer},
  year = {2025},
  month = {December},
  course = {EECE 454 - Computer Network Modelling and Optimization},
  institution = {American University of Beirut},
  url = {https://github.com/kaaek/edge-uav-mec-optimizer},
  branch = {man}
}
```

## License

Educational and research use only. Code provided as-is for academic purposes.

## Contact

- **Khalil El Kaaki** — [GitHub @kaaek](https://github.com/kaaek)
- **Joe Abi Samra** — [GitHub @ovbismark74](https://github.com/ovbismark74)

**Repository:** https://github.com/kaaek/edge-uav-mec-optimizer  
**Branch:** `main`

---

**Last Updated:** December 2025
