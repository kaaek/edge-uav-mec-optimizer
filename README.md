# UAV-Ground Association Optimizer with Multi-Access Edge Computing (MEC)

**Authors:** Khalil El Kaaki & Joe Abi Samra

## Overview

Phase 2 shifts the focus from proportional-fair throughput maximization (Phase 1) to maximizing the number of successfully completed IoT tasks under deadline (slack) and probabilistic wireless reliability constraints. Tasks arrive according to a Poisson process; each task has data size, computational density, and a deadline. A single UAV with an optimized trajectory (offline batch or online receding horizon) and a fixed ground Base Station compete with local processing as candidate execution locations. Communication uses TDMA (full bandwidth per active transmission) and a probabilistic channel success model.

The legacy Phase 1 modules (clustering, fractional offloading fairness optimizer) remain in the repository for comparison but are not invoked by the Phase 2 entry script.

### Key Features (Phase 2)
- GPU-accelerated PyTorch simulation (automatic CUDA fallback to CPU)
- Poisson-distributed task arrivals per IoT device
- TDMA uplink scheduling with dynamic waiting time estimation
- Greedy reliability- and deadline-aware offloading (local vs BS vs UAV)
- Channel reliability model (exponential success probability from SNR)
- Two trajectory optimization modes:
  - Offline batch gradient descent (global path planning)
  - Online receding horizon (MPC-style) adaptive optimization
- Parameter sweeps: UAV velocity, UAV CPU, bandwidth, task size range, cycles/bit range, number of devices
- Monte Carlo trials with fresh random IoT spatial layouts per trial
- Rich visualization: trajectory plots + sweep performance/error-bar charts

### Legacy (Phase 1) Capabilities
If you need the proportional fairness throughput optimization (joint UAV placement, bandwidth allocation, fractional offloading), the original formulation is retained under `util/clustering`, `util/optimizers`, and `util/common`. See archived description in earlier README revisions and `docs/optimization_formulation.md` for the new task completion MINLP formulation.

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

Install PyTorch (choose one):
```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1+
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# CPU only
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

## Repository Structure (Current Focus)
```
edge-uav-mec-optimizer/
├── main_task_offloading.py      # Phase 2 entry point (task completion sweeps)
├── docs/optimization_formulation.md  # Formal MINLP task completion model
├── util/
│   ├── classes/                 # Core data models (UAV, IoTDevice, Task, BaseStation, constants)
│   ├── schedulers/              # Greedy offloading + TDMA queue logic
│   ├── optimizers/              # Offline & online trajectory optimizers
│   ├── common/                  # Wireless propagation & reliability primitives
│   ├── clustering/              # Legacy Phase 1 UAV placement algorithms
│   └── plotter/                 # Plot helpers (trajectories, sweeps)
└── results/                     # Auto-saved figures (sweeps & sample trajectories)
```

## Phase 2 Problem (Simplified)
Objective: Maximize total number of completed tasks.

A task completes if (serving_time ≤ slack) AND (channel_success ≥ P_min).

Serving time components (offloaded): uplink transmission + remote computation (+ TDMA wait). Local tasks use only local computation time.

Reliability: `P_success = exp(-SNR_threshold / SNR_avg)` (Rayleigh average model) applied to offloaded options; local execution is assumed reliable.

See `docs/optimization_formulation.md` for full MINLP formulation & constraints (trajectory dynamics, TDMA non-overlap, velocity bounds, etc.).

## Usage (Phase 2)

Run all parameter sweeps (both online & offline trajectory optimization):
```bash
python main_task_offloading.py
```
Outputs:
- Console progress with per-trial debug info
- For each sweep: three plots (`*_online.png`, `*_offline.png`, `*_comparison.png`)
- Trajectory visualization for the first trial of the last sweep value

### Adjusting Configuration
Edit `BASE_CONFIG` in `main_task_offloading.py`:
- `uav_max_velocity`, `uav_cpu_frequency`
- `BW_total` (Hz), task size ranges (`task_size_min/max`), cycles per bit ranges
- `num_devices`, `trials` for Monte Carlo averaging
- Toggle optimization mode with `online_optimization` (script runs both by default in sweeps)

### Single Trial Example
To run one quick scenario, modify the `__main__` block (or create a small driver):
```python
from main_task_offloading import run_trial, BASE_CONFIG
completion_rate = run_trial(BASE_CONFIG, trial_seed=123, device='cuda', optimization_mode='online')
print(f"Completion Rate: {completion_rate*100:.1f}%")
```

### Trajectory Optimization Modes
- Offline: Gradient descent minimizes average distance to IoT cluster (proxy for channel quality) with velocity penalties.
- Online (Receding Horizon / MPC): Re-optimizes next H steps based on pending tasks and urgency (inverse remaining slack).

## Core Phase 2 Components

| Module | Purpose |
|--------|---------|
| `util/classes/task.py` | Task definition (size, density, generation time, slack). |
| `util/classes/iot_device.py` | IoT device with Poisson arrival generator & local compute model. |
| `util/schedulers/offloading_decision.py` | Greedy offloading & completion logic (deadline + reliability). |
| `util/schedulers/tdma_scheduler.py` | TDMA queue, waiting time estimation, scheduling primitives. |
| `util/optimizers/trajectory_optimizer.py` | Offline gradient or SCA trajectory optimization. |
| `util/optimizers/online_trajectory_optimizer.py` | Receding horizon (MPC) trajectory refinement. |
| `util/common/channel_reliability.py` | Channel success probability (Rayleigh-based). |

## Metrics Reported
- Task Completion Rate (%) with error bars (across trials)
- Online vs Offline average improvement per sweep
- Debug counts: tasks per device, per-trial completion breakdown

## Parameter Sweeps (Phase 2)
- `uav_max_velocity`: Mobility impact
- `uav_cpu_frequency`: Edge compute scaling
- `bandwidth (BW_total)`: Communication resource impact
- `task_size (min/max range)`: Data volume sensitivity
- `cycles_per_bit (min/max range)`: Computational density impact
- `num_devices`: System scalability

Each point averages `trials` independent layouts (new IoT spatial positions & Poisson realizations).

## Reliability & Deadlines
- Minimum reliability threshold: `P_min` (default 0.90–0.95 range supported)
- SNR threshold (linear) convertible to dB internally for probability model
- TDMA waiting time inserted prior to transmission start; if total time exceeds slack, task rejected.

## GPU Acceleration
All heavy tensor operations (distance, loss gradients, velocity penalties) leverage PyTorch. Fallback to CPU if CUDA unavailable. Expect 5–20× speedup for larger sweeps versus CPU-only.

## Legacy Throughput Optimizer (Phase 1)
The earlier proportional fairness (`Σ log(Th_m)`) formulation with fractional offloading, bandwidth allocation, and clustering-based initialization is still accessible for research comparison. Use previous commit history or restore `main.py` (not included in Phase 2 branch) to run those experiments.

## Troubleshooting (Phase 2)
| Issue | Likely Cause | Fix |
|-------|--------------|-----|
| `ModuleNotFoundError` | Wrong working directory | `cd edge-uav-mec-optimizer` |
| CUDA = False | Missing toolkit or wrong PyTorch wheel | Reinstall with proper `--index-url` |
| Slow sweeps | Too many trials / CPU-only | Reduce `trials` or install CUDA |
| Empty plots | No tasks generated (low λ) | Increase `iot_lambda_rate` in `BASE_CONFIG` |
| Low completion rate | Deadlines too strict or weak channels | Increase slack ranges / bandwidth / UAV velocity |

## Citation
If this Phase 2 task completion work is used, cite:
```bibtex
@software{elkaaki2025uavmec,
  author = {El Kaaki, Khalil and Abi Samra, Joe},
  title = {Edge UAV-MEC Optimizer: Task Completion & Trajectory Optimization},
  year = {2025},
  month = {November},
  url = {https://github.com/kaaek/edge-uav-mec-optimizer}
}
```

## License
Educational & research use only (as-is).

## Contact
- Khalil El Kaaki — [GitHub @kaaek](https://github.com/kaaek)
- Joe Abi Samra - [Github @ovbismark74](https://github.com/ovbismark74)

Branch: `phase-2-maximization-of-completed-tasks-using-a-probabilistic-channel-model`

**Repository:** https://github.com/kaaek/edge-uav-mec-optimizer
