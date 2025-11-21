# Changelog

All notable changes to the Edge UAV-MEC Optimizer project.

## [2.0.0] - 2025-01-XX - MEC Extension

### Added
- **Multi-Access Edge Computing (MEC) support**
  - Fractional task offloading model
  - Decision variable extended: now includes offloading fractions `o_m ∈ [0,1]`
  - MEC throughput calculation: `Th_m = o_m*D_m/(T_ul+T_comp+T_dl) + (1-o_m)*D_m/T_local`
  
- **New MEC Parameters** (`util/constants.py`)
  - `D_m = 5e6` bits: Task data size
  - `C_m = 1e9` cycles: Computational complexity
  - `f_UAV = 10e9` Hz: UAV CPU frequency
  - `f_user = 1e9` Hz: User device CPU frequency

- **New Function**: `compute_mec_throughput()` in `util/common/__init__.py`
  - Implements fractional offloading model
  - Computes upload time, computation time, download time, local processing time
  - Returns end-to-end task completion throughput

- **Enhanced Constraints** in `nonlcon_joint()`
  - QoS constraints: `Th_m >= R_min` (MEC throughput, not just wireless rate)
  - Bandwidth budget: `sum(b_m) <= BW_total`
  - CPU capacity: `sum(o_m * C_m / f_UAV) <= 1` per UAV (new)

- **Documentation**
  - `MEC_EXTENSION.md`: Complete MEC problem formulation and implementation guide
  - Updated `README.md` with MEC usage examples
  - This `CHANGELOG.md` file

### Changed
- **`optimize_network()` signature** now includes MEC parameters
  - Before: `(M, N, ..., Rmin, device)`
  - After: `(M, N, ..., Rmin, D_m, C_m, f_UAV, f_user, device)`
  - Returns 5 values: `(uav_pos, bandwidth, offloading, throughput, sumrate)`

- **`rate_fn()` in helpers** now computes MEC throughputs instead of wireless rates
  - Decision vector extended from `2*N + M` to `2*N + 2*M` entries
  - Unpacks: `[uav_pos(2*N), b_m(M), o_m(M)]`
  - Returns: MEC throughputs `Th_m` via `compute_mec_throughput()`

- **`p_received()` in common** - fixed unit conversions
  - Frequency: `F_MHz = F / 1e6` (Hz → MHz)
  - Distance: `d_km = d_m / 1000` (meters → km)
  - Added assertions to validate input ranges

- **Initialization strategy** in `optimize_network()`
  - Bandwidth: Uniform allocation (`BW_total / M` per user) instead of conservative estimate
  - Offloading: Start at full offload (`o_m = 1.0`) to allow optimizer to explore
  - Increased max iterations from 20 to 100 for better convergence

- **`main.py` and `test.py`** updated to:
  - Import all 21 constants (was 17)
  - Pass MEC parameters to `optimize_network()` calls
  - Unpack 5 return values (was 4)

### Fixed
- Unit conversion bugs in Okumura-Hata propagation model
- Bandwidth initialization too conservative (now uses full budget)
- Offloading initialization (was arbitrary 0.5, now 1.0 for better exploration)

---

## [1.0.0] - 2025-01-XX - Initial Python Translation

### Added
- Complete Python translation from MATLAB UAV-Ground Association Optimizer
- GPU acceleration using PyTorch tensors
- K-means clustering for UAV positioning (`util/clustering/`)
- Hierarchical clustering for UAV positioning (`util/clustering/`)
- Joint optimization of UAV positions and bandwidth allocation (`util/optimizers/`)
- Okumura-Hata propagation model (`p_received()`)
- Shannon capacity model (`bitrate()`, `se()`)
- User-UAV association (`association()`)
- Proportional fairness objective (log-sum rate)
- QoS constraints (minimum rate per user)
- Bandwidth budget constraint
- Parameter sweep framework (`main.py`)
  - N sweep (Number of UAVs)
  - M sweep (Number of users)
  - BW sweep (Total bandwidth)
  - P_T sweep (Transmit power)
  - R_MIN sweep (QoS requirement)
  - AREA sweep (Coverage area)
- Network visualization (`plot_network()`)
- Sweep result plotting (`plot_sweep()`)
- Quick test script (`test.py`) with 4-bar performance comparison chart
- Documentation (`README.md`, `TRANSLATION_SUMMARY.md`)

### Technical Details
- **Language**: Python 3.8+
- **Deep Learning Framework**: PyTorch >= 2.0.0
- **Optimization**: SciPy SLSQP (Sequential Least Squares Programming)
- **Clustering**: SciPy hierarchical clustering + custom K-means
- **Device support**: CUDA (GPU) or CPU

### Dependencies
```
torch>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
```

### Performance
- 10-100× speedup on GPU vs CPU for large problems (M > 100, N > 10)
- Faithful reproduction of MATLAB logic
- All parameter sweeps verified against original implementation

---

## Release Notes

### v2.0.0 - MEC Extension
This major release extends the UAV network optimizer to support Multi-Access Edge Computing scenarios. Users can now offload computational tasks to UAVs, and the optimizer jointly decides:
1. Where to place UAVs (positions)
2. How to allocate bandwidth (resource allocation)
3. How much of each task to offload (offloading decisions)

The MEC model accounts for:
- Upload transmission time
- Computation time at UAV
- Download transmission time
- Local computation time

This enables realistic modeling of edge computing applications like:
- AR/VR offloading
- Video analytics
- Real-time object detection
- Collaborative robotics

**Breaking Changes:**
- `optimize_network()` signature changed (added 4 MEC parameters)
- Return values changed (4 → 5 values, added offloading fractions)
- `constants()` return changed (17 → 21 values)

**Migration Guide:**
```python
# Old (v1.0.0)
M, N, ..., TRIALS = constants()  # 17 values
uav_pos, bw, rates, sumrate = optimize_network(M, N, ..., Rmin, device)

# New (v2.0.0)
M, N, ..., TRIALS, D_m, C_m, f_UAV, f_user = constants()  # 21 values
uav_pos, bw, offload, rates, sumrate = optimize_network(
    M, N, ..., Rmin, D_m, C_m, f_UAV, f_user, device
)
```

### v1.0.0 - Initial Release
First stable release of the Python GPU-accelerated UAV network optimizer. Faithful translation from MATLAB with full feature parity.

---

## Future Roadmap

### Planned Features
- [ ] Heterogeneous tasks (different D_m, C_m per user)
- [ ] UAV power consumption modeling
- [ ] Energy-aware optimization
- [ ] Multi-hop offloading (UAV-to-UAV)
- [ ] User mobility support
- [ ] Stochastic task arrival models
- [ ] Deep learning-based initialization
- [ ] Distributed optimization algorithms

### Performance Improvements
- [ ] Mixed-precision computation (FP16/FP32)
- [ ] Batch optimization for multiple scenarios
- [ ] Asynchronous GPU execution
- [ ] Sparse constraint handling

### Usability
- [ ] Web-based visualization dashboard
- [ ] Configuration file support (YAML/JSON)
- [ ] Pre-trained models for common scenarios
- [ ] Interactive Jupyter notebook tutorials

---

**Contributors:**
- Khalil El Kaaki (Original MATLAB implementation, Python translation)
- Joe Abi Samra (Original MATLAB implementation, Python translation)

**License:** Educational/Research Use

**Citation:**
```bibtex
@software{edge_uav_mec_optimizer_2025,
  author = {El Kaaki, Khalil and Abi Samra, Joe},
  title = {Edge UAV-MEC Optimizer: GPU-Accelerated Python Implementation},
  year = {2025},
  version = {2.0.0}
}
```
