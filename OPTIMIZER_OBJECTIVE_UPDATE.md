# Optimizer Objective Function Update

## Latest Updates (December 2, 2025)

### Critical Fixes Implemented

**All 6 identified issues have been resolved:**

1. ✅ **Use Okumura-Hata Path Loss Model** - Replaced simplified `SNR ∝ 1/dist²` with actual `p_received()` function
2. ✅ **Model All Three Offloading Options** - Optimizer now considers Local, BS, and UAV offloading
3. ✅ **Include TDMA Wait Time** - Added estimated queueing delay (100ms average)
4. ✅ **Increased Iterations** - Offline: 50 iterations (was 20), Online: 40 iterations (was 20)
5. ✅ **Tuned Hyperparameters** - Adjusted learning rates and sigmoid steepness factors
6. ✅ **Added Debugging Output** - Shows predicted offloading choices during optimization

---

## Changes Made

Both trajectory optimizers have been updated to **maximize the expected number of completed tasks** with realistic channel models and offloading decision logic.

---

## New Implementation Details

### Objective Function: Maximize Expected Completed Tasks

Both optimizers now optimize:

```
maximize Σ_k E[completion_k]

where E[completion_k] = max(
    P(local_completion_k),
    P(BS_completion_k),
    P(UAV_completion_k)
)
```

### Task Completion Probability (All Three Options)

For each task k, the optimizer evaluates:

#### **Option 1: Local Processing**
```python
t_local = (task.cycles) / iot_cpu_freq
p_local = 1.0  # Always reliable (no channel)
local_score = sigmoid(5.0 × (slack - t_local)) × 1.0
```

#### **Option 2: Base Station Offloading**
```python
# Use Okumura-Hata path loss
p_r_bs = p_received(iot_pos, bs_pos, H_M, H_bs, F, P_T)
snr_bs = 10^(p_r_bs/10) / 10^(noise_var/10)
data_rate_bs = BW_total × log2(1 + snr_bs)

# Serving time
t_bs = task.bits/data_rate_bs + task.cycles/bs_cpu + TDMA_wait

# Reliability and completion
p_reliable_bs = sigmoid(2.0 × (snr_bs - snr_thresh))
bs_score = sigmoid(5.0 × (slack - t_bs)) × p_reliable_bs
```

#### **Option 3: UAV Offloading**
```python
# Use Okumura-Hata path loss
p_r_uav = p_received(iot_pos, uav_pos, H_M, H_uav, F, P_T)
snr_uav = 10^(p_r_uav/10) / 10^(noise_var/10)
data_rate_uav = BW_total × log2(1 + snr_uav)

# Serving time
t_uav = task.bits/data_rate_uav + task.cycles/uav_cpu + TDMA_wait

# Reliability and completion
p_reliable_uav = sigmoid(2.0 × (snr_uav - snr_thresh))
uav_score = sigmoid(5.0 × (slack - t_uav)) × p_reliable_uav
```

#### **Soft Decision Mechanism**
```python
scores = [local_score, bs_score, uav_score]
weights = softmax(10.0 × scores)  # Temperature=10 for sharp decisions
expected_completion = Σ(weights[i] × scores[i])
```

This creates a **differentiable approximation** of the greedy scheduler's decision logic.

---

## Key Improvements Over Previous Version

### 1. **Realistic Path Loss Model**

**Before:**
```python
path_loss_factor = 1.0 / (dist_3d^2 + 1.0)  # Simplified
estimated_snr = 20.0 × path_loss_factor
```

**After:**
```python
# Uses Okumura-Hata model with frequency, height, environment
p_r = p_received(iot_pos, uav_pos, H_M, H, F, P_T)
snr = 10^(p_r/10) / 10^(noise_var/10)
```

**Impact:** Optimizer now optimizes for the **same physics** the scheduler uses.

### 2. **All Offloading Options Modeled**

**Before:**
- Assumed all tasks offload to UAV only

**After:**
- Local processing (fastest CPU, no transmission)
- BS offloading (static position, powerful CPU)
- UAV offloading (mobile, moderate CPU)

**Impact:** Optimizer predicts which option scheduler will actually choose.

### 3. **TDMA Wait Time Included**

**Before:**
```python
serving_time = transmission_time + computation_time
```

**After:**
```python
serving_time = transmission_time + computation_time + avg_tdma_wait
```

**Impact:** More accurate serving time estimation (±100ms).

### 4. **Better Convergence**

**Before:**
- Offline: 20 iterations, lr=1.0
- Online: 20 iterations, lr=5.0

**After:**
- Offline: 50 iterations, lr=0.5
- Online: 40 iterations, lr=2.0

**Impact:** More stable convergence, less overshooting.

### 5. **Tuned Sigmoid Functions**

**Before:**
```python
completion_prob = sigmoid(10.0 × slack_margin)
reliability_prob = sigmoid(-0.1 × (dist - 30))
```

**After:**
```python
completion_prob = sigmoid(5.0 × slack_margin)  # Smoother
reliability_prob = sigmoid(2.0 × (snr - snr_thresh))  # SNR-based
```

**Impact:** Better gradient flow, more realistic constraints.

### 6. **Debugging Output**

**New Output:**
```
[OPTIMIZER] Iter  20/50: expected_completions=8.45/10 (84.5%), 
           vel_penalty=0.0234, loss=-8.4266
           Predicted choices: Local=2, BS=3, UAV=5
```

**Impact:** Can verify optimizer is making sensible offloading decisions.

---

## Optimization Details

### Offline Trajectory Optimizer

**File**: `util/optimizers/trajectory_optimizer.py`

**Changes**:
- Lines 13-15: Added imports for `p_received` and `channel_success_probability`
- Lines 70-80: Updated docstring to reflect new objective
- Lines 124-240: Complete rewrite of objective function
  - Evaluates all three offloading options
  - Uses Okumura-Hata path loss via `p_received()`
  - Computes SNR from received power in dBm
  - Includes TDMA wait time estimate
  - Uses softmax for differentiable decision
  - Tracks predicted choices for debugging

**Hyperparameters**:
- Iterations: 50 (increased from 20)
- Learning rate: 0.5 (reduced from 1.0)
- Slack sigmoid steepness: 5.0 (reduced from 10.0)
- SNR sigmoid steepness: 2.0 (new)
- Softmax temperature: 10.0 (for sharp decisions)

### Online Trajectory Optimizer (Receding Horizon)

**File**: `util/optimizers/online_trajectory_optimizer.py`

**Changes**:
- Lines 13-15: Added imports for channel calculations
- Lines 17-25: Updated class docstring
- Lines 28-50: Added UAV parameters (cpu_freq, height, params) to optimizer
- Lines 145-210: Complete rewrite of horizon optimization
  - Evaluates all three offloading options per task
  - Uses Okumura-Hata path loss via `p_received()`
  - Includes TDMA wait time
  - Uses softmax for differentiable decision

**Hyperparameters**:
- Iterations per step: 40 (increased from 20)
- Learning rate: 2.0 (reduced from 5.0)
- Slack sigmoid steepness: 5.0
- SNR sigmoid steepness: 2.0
- Softmax temperature: 10.0

---

## Loss Function

Both optimizers use:

```python
loss = -expected_completions + velocity_penalty
```

**Components**:
- `expected_completions`: Sum of soft completion scores across all tasks
  - Each score is weighted combination of {local, BS, UAV} options
  - Ranges from 0 (no completions) to N (all complete)
- `velocity_penalty`: 100× sum of velocity violations (hard constraint)

We minimize the negative of expected completions, which is equivalent to maximizing completions.

---

## Differentiability

All components remain differentiable:
- ✅ Okumura-Hata path loss: `log10(distance)`
- ✅ Power conversion: `10^(p_r/10)`
- ✅ SNR calculation: Division of linear powers
- ✅ Data rate: `log2(1 + SNR)`
- ✅ Serving time: Division with epsilon
- ✅ Completion probability: `sigmoid(slack_margin)`
- ✅ Reliability: `sigmoid(snr_margin)`
- ✅ Soft decision: `softmax(scores)`

This enables gradient-based optimization using PyTorch autograd.

---

## Expected Behavior Changes

### Before Fixes:
- ❌ Optimized for wrong physics (simplified path loss)
- ❌ Assumed UAV-only offloading
- ❌ Ignored TDMA wait time
- ❌ May have converged to local minima (too few iterations)
- ❌ Circular baseline performed better

### After Fixes:
- ✅ Optimizes for correct physics (Okumura-Hata)
- ✅ Models realistic offloading decisions
- ✅ Accounts for queueing delays
- ✅ Better convergence with more iterations
- ✅ **Should outperform circular baseline**

---

## Verification Steps

1. **Run optimizer and check debug output**:
   ```
   Predicted choices: Local=X, BS=Y, UAV=Z
   ```
   - Should see mix of all three options
   - UAV choices should increase when UAV is closer

2. **Compare with scheduler decisions**:
   - Optimizer predictions should align with actual scheduler choices
   - Mismatch indicates further tuning needed

3. **Monitor convergence**:
   ```
   expected_completions: 5.2 → 6.8 → 7.9 → 8.5 → 8.7
   ```
   - Should increase over iterations
   - Should plateau near actual completion rate

4. **Compare trajectories**:
   - Optimized trajectories should cluster near devices with urgent tasks
   - Should balance between multiple devices based on slack times

---

## Files Modified

1. ✅ `util/optimizers/trajectory_optimizer.py` - Offline batch optimizer
2. ✅ `util/optimizers/online_trajectory_optimizer.py` - Online receding horizon optimizer
3. ✅ `main_task_offloading.py` - Updated iteration counts and learning rates

---

## Performance Expectations

With these fixes, you should see:

1. **Higher completion rates** - Optimized trajectories should beat circular baseline
2. **Realistic offloading mix** - Debug output should show Local/BS/UAV choices
3. **Better convergence** - Loss should decrease smoothly over iterations
4. **Sensible trajectories** - UAV should position near urgent/pending tasks

If circular baseline still performs better after these fixes, investigate:
- Task distribution patterns (clustered vs uniform)
- Slack time settings (too tight or too loose)
- Velocity constraints (too restrictive)
- Optimization hyperparameters (may need further tuning)

---

## Next Steps

1. Run a small test sweep to verify improvements
2. Check debug output to confirm realistic offloading predictions
3. Compare optimized vs circular trajectories visually
4. If needed, further tune sigmoid steepness factors
5. Consider adaptive learning rate scheduling
