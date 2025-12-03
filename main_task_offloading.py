"""
Main benchmarking script for UAV-MEC Task Offloading with TDMA
Compares Online (Receding Horizon) vs Offline (Batch Gradient) Trajectory Optimization

Parameter Sweeps:
- UAV maximum velocity
- UAV CPU frequency
- Communication bandwidth
- Task data size
- Task computational complexity (cycles/bit)
- Number of IoT devices (scalability)

Features:
- Poisson-distributed task arrivals
- New random IoT positions for each trial (proper Monte Carlo)
- Trajectory visualizations
- Statistical comparison with error bars

Author: Khalil El Kaaki & Joe Abi Samra
Date: December, 2025

python -m pyinstrument -r html -o profile.html ./main_task_offloading.py
or
python -m cProfile -o profile.prof main_task_offloading.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from util.classes import UAV, IoTDevice, Task, BaseStation
from util.schedulers import greedy_offloading_batch, OffloadingParams
from util.optimizers.trajectory_optimizer import optimize_uav_trajectory, compute_trajectory_velocity
from util.optimizers.online_trajectory_optimizer import optimize_trajectory_online


# For NVIDIA GPUs, use CUDA for performance. Otherwise, fall back to the CPU.
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("\n" + "=" * 80)
print("  UAV-MEC TASK OFFLOADING OPTIMIZER")
print("  Trajectory Optimization for Aerial Base Stations")
print("=" * 80)
if device == 'cuda':
    print(f"✓ Using CUDA GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠ CUDA not available - Running on CPU")
print("=" * 80)

# ========================================================================
# Base Configuration
# ========================================================================
BASE_CONFIG = {
    # Time
    'T': 70,
    'duration': 20.0,  # total duration in seconds 
    
    # UAV
    'uav_cpu_frequency': 5e9,   # 5 GHz
    'uav_max_velocity': 20.0,   # m/s
    'uav_height': 50.0,         # meters
    'uav_radius': 200.0,         # circular path radius
    'initial_position': [200.0, 200.0],
    
    # Base station
    'bs_position': [700.0, 700.0],  # Further away to make UAV proximity advantage matter
    'bs_cpu_frequency': 4e9,    # 4 GHz (lower than UAV to favor proximity)
    'bs_height': 30.0,          # meters
    
    # IoT device 
    'num_devices': 5,
    'iot_cpu_frequency': 1e9,   # 1 GHz
    'iot_lambda_rate': 0.5,     # tasks/second (Poisson arrival rate)
    'device_area_min': 50,   # meters
    'device_area_max': 350,  # meters
    
    # Task (generated via Poisson process)
    'task_size_min': 1.5e6,  # 1.5 Mb
    'task_size_max': 7e6,  # 7 Mb
    'cycles_per_bit_min': 500,
    'cycles_per_bit_max': 1500,
    'slack_min': 0.7,   # seconds (increased to reduce infeasible tasks)
    'slack_max': 2.0,   # seconds
    
    # System parameters
    'BW_total': 15e6,   # 15 MHz
    'H_M': 1.5,         # IoT device height (meters)
    'F': 2.4e9,         # 2.4 GHz carrier
    'P_T': 20.0,        # 20 dBm transmit power
    'noise_var': -80.0, # -80 dBm noise variance
    'snr_thresh': 15.0, # SNR threshold (linear)
    'P_min': 0.95,      # 95% reliability
    
    # Simulation
    'trials': 5,        # Monte Carlo trials (increased for statistical reliability)
    'seed_base': 50,
    
    # Trajectory optimization
    'optimize_trajectory': True,    # Enable UAV trajectory optimization
    'online_optimization': True,    # Use online receding horizon (vs offline batch)
    'use_circular_baseline': False, # Use circular trajectory (for baseline comparison)
    'planning_horizon': 15,         # Planning horizon for online optimization
}


# ========================================================================
# Sweep Configurations
# ========================================================================
SWEEP_CONFIGS = {
    # 'uav_max_velocity': {
    #     'values': [5.0, 10.0, 15.0, 20.0, 25.0, 30.0],  # m/s
    #     'param': 'uav_max_velocity',
    #     'xlabel': 'UAV Maximum Velocity (m/s)',
    #     'ylabel': 'Task Completion Rate (%)',
    #     'title': 'Impact of UAV Mobility on Task Completion Rate',
    # },
    # 'uav_cpu_frequency': {
    #     'values': [2e9, 3e9, 4e9, 5e9, 6e9, 8e9, 10e9],  # Hz
    #     'param': 'uav_cpu_frequency',
    #     'xlabel': 'UAV CPU Frequency (GHz)',
    #     'ylabel': 'Task Completion Rate (%)',
    #     'title': 'Impact of UAV Computing Capacity on Task Completion Rate',
    #     'scale': 1e9,
    # },
    # 'bandwidth': {
    #     'values': [5e6, 10e6, 20e6, 40e6, 60e6, 80e6, 100e6],  # Hz
    #     'param': 'BW_total',
    #     'xlabel': 'Total Bandwidth (MHz)',
    #     'ylabel': 'Task Completion Rate (%)',
    #     'title': 'Impact of Communication Bandwidth on Task Completion Rate',
    #     'scale': 1e6,
    # },
    'task_size': {
        'values': [(1e6, 3e6), (2e6, 6e6), (3e6, 9e6)],  # (min, max) bits
        'param': 'task_size',
        'xlabel': 'Average Task Size (Mb)',
        'ylabel': 'Task Completion Rate (%)',
        'title': 'Impact of Task Data Size on Task Completion Rate',
        'scale': 1e6,
    },
    # 'cycles_per_bit': {
    #     'values': [(100, 500), (300, 800), (500, 1500), (1000, 2000), (1500, 3000)],  # (min, max)
    #     'param': 'cycles_per_bit',
    #     'xlabel': 'Average Computational Density (Cycles/Bit)',
    #     'ylabel': 'Task Completion Rate (%)',
    #     'title': 'Impact of Task Computational Complexity on Completion Rate',
    # },
    'num_devices': {
        'values': [2, 5, 10, 15, 20],
        'param': 'num_devices',
        'xlabel': 'Number of IoT Devices',
        'ylabel': 'Task Completion Rate (%)',
        'title': 'System Scalability: Task Completion vs Network Size',
    },
    'bandwidth': {
        'values': [5e6, 10e6, 20e6, 40e6, 60e6],  # Hz
        'param': 'BW_total',
        'xlabel': 'Total Bandwidth (MHz)',
        'ylabel': 'Task Completion Rate (%)',
        'title': 'Impact of Communication Bandwidth on Task Completion Rate',
        'scale': 1e6,
    },
}

# ========================================================================
# Helper Functions
# ========================================================================

def create_scenario(config, trial_seed, device='cuda'):
    """Create UAV-MEC scenario with given configuration.
    
    Args:
        config: Configuration dictionary
        trial_seed: Seed for this specific trial (generates new random IoT positions)
        device: 'cuda' or 'cpu'
    """
    T = config['T']                 # The time vector
    duration = config['duration']   # The time step (analogous to delta t)
    time_indices = torch.linspace(0, duration, T, device=device) # array of [0, duration, 2*duration ... T]
    
    # UAV trajectory: circular path
    theta = torch.linspace(0, 2*np.pi, T, device=device)
    radius = config['uav_radius']
    center_x, center_y = 100.0, 100.0
    
    uav_x = center_x + radius * torch.cos(theta)
    uav_y = center_y + radius * torch.sin(theta)
    uav_position = torch.stack([uav_x, uav_y], dim=0)
    
    uav_vx = -radius * 2*np.pi/duration * torch.sin(theta)
    uav_vy = radius * 2*np.pi/duration * torch.cos(theta)
    uav_velocity = torch.stack([uav_vx, uav_vy], dim=0)
    
    # Base station (create first, before IoT devices)
    bs = BaseStation(
        position=config['bs_position'],
        cpu_frequency=config['bs_cpu_frequency'],
        time_indices=time_indices,
        height=config['bs_height'],
        device=device
    )
    
    # IoT devices - use trial_seed for new random positions each trial
    iot_devices = []
    np.random.seed(trial_seed)
    torch.manual_seed(trial_seed)
    for i in range(config['num_devices']):
        pos_x = np.random.uniform(BASE_CONFIG.get('device_area_min'), BASE_CONFIG.get('device_area_max'))
        pos_y = np.random.uniform(BASE_CONFIG.get('device_area_min'), BASE_CONFIG.get('device_area_max'))
        
        iot = IoTDevice(
            position=[pos_x, pos_y],
            cpu_frequency=config['iot_cpu_frequency'],
            lambda_rate=config['iot_lambda_rate'],
            device=device
        )
        iot_devices.append(iot)
    
    # System parameters
    params = OffloadingParams(
        BW_total=config['BW_total'],
        H_M=config['H_M'],
        H=config['uav_height'],
        F=config['F'],
        P_T=config['P_T'],
        noise_var=config['noise_var'],
        snr_thresh=config['snr_thresh'],
        P_min=config['P_min'],
        current_time=0.0
    )
    
    # Optimize UAV trajectory if requested
    initial_position = BASE_CONFIG.get('initial_position')
    
    if config.get('no_uav_baseline', False):
        # No UAV baseline - skip UAV entirely
        print("  → No UAV baseline (Local + BS only)")
        uav = None
        optimization_tasks = None
        return uav, bs, iot_devices, time_indices, params, optimization_tasks
    
    if config.get('use_circular_baseline', False):
        # Use circular trajectory as baseline
        print("  → Using circular trajectory baseline (no optimization)")
        # Circular trajectory already computed above - just use it
        # Compute velocity from circular positions
        uav_velocity = compute_trajectory_velocity(uav_position, time_indices)
        
    elif config.get('optimize_trajectory', True):
        # Generate tasks first (needed for trajectory optimization)
        tasks, task_owners = generate_tasks(iot_devices, config)
        
        print("  → Starting trajectory optimization...")
        if config.get('online_optimization', True):
            # Online receding horizon optimization
            print("     • Mode: Online (Receding Horizon)")
            uav_position = optimize_trajectory_online(
                iot_devices=iot_devices,
                tasks=tasks,
                bs=bs,
                time_indices=time_indices,
                uav_cpu_frequency=config['uav_cpu_frequency'],
                uav_max_velocity=config['uav_max_velocity'],
                uav_height=config['uav_height'],
                initial_position=initial_position,
                params=params,
                device=device,
                horizon=config.get('planning_horizon', 10)
            )
        else:
            # Offline batch optimization
            print("     • Mode: Offline (Batch Gradient Descent)")
            uav_position = optimize_uav_trajectory(
                iot_devices=iot_devices,
                tasks=tasks,
                bs=bs,
                time_indices=time_indices,
                uav_cpu_frequency=config['uav_cpu_frequency'],
                uav_max_velocity=config['uav_max_velocity'],
                uav_height=config['uav_height'],
                initial_position=initial_position,
                params=params,
                device=device,
                max_iter=25,
                learning_rate=0.5,
                method='gradient'
            )
        
        print("  ✓ Trajectory optimization completed\n")
        
        # Compute velocity from optimized positions
        uav_velocity = compute_trajectory_velocity(uav_position, time_indices)
    else:
        # Fallback: stationary UAV at center
        uav_position = torch.zeros(2, T, device=device)
        uav_position[0, :] = initial_position[0]
        uav_position[1, :] = initial_position[1]
        uav_velocity = torch.zeros(2, T, device=device)
    
    # Create UAV with optimized trajectory
    uav = UAV(
        time_indices=time_indices,
        cpu_frequency=config['uav_cpu_frequency'],
        max_velocity=config['uav_max_velocity'],
        initial_position=initial_position,
        initial_velocity=[0.0, 0.0],
        device=device
    )
    uav.position = uav_position
    uav.velocity = uav_velocity
    uav.height = config['uav_height']
    
    # Return tasks used for optimization (or None if circular baseline)
    if config.get('optimize_trajectory', True) and not config.get('use_circular_baseline', False):
        optimization_tasks = (tasks, task_owners)
    else:
        optimization_tasks = None
    
    return uav, bs, iot_devices, time_indices, params, optimization_tasks


def generate_tasks(iot_devices, config):
    """Generate tasks for IoT devices using Poisson process."""
    tasks = []
    task_owners = []
    
    duration = config['duration']
    
    for device_idx, iot in enumerate(iot_devices):
        # Generate task arrival times using Poisson process
        arrival_times = iot.generate_task_arrival_times(duration)
        
        for task_num, arrival_time in enumerate(arrival_times):
            # Random task parameters
            length_bits = np.random.uniform(
                config['task_size_min'],
                config['task_size_max']
            )
            computation_density = np.random.uniform(
                config['cycles_per_bit_min'],
                config['cycles_per_bit_max']
            )
            slack = np.random.uniform(config['slack_min'], config['slack_max'])
            
            task = Task(
                length_bits=length_bits,
                computation_density=computation_density,
                time_generated=arrival_time.item() if torch.is_tensor(arrival_time) else arrival_time,
                slack=slack,
                task_id=f"task_d{device_idx}_t{task_num}",
                device_id=device_idx  # Track which device generated this task
            )
            
            tasks.append(task)
            task_owners.append(device_idx)
    
    return tasks, task_owners


def visualize_trajectory(uav, bs, iot_devices, optimization_mode, sweep_name, param_value, save_name):
    """
    Visualize UAV trajectory with IoT devices and base station.
    
    Args:
        uav: UAV object with position trajectory
        bs: BaseStation object
        iot_devices: List of IoTDevice objects
        optimization_mode: 'online' or 'offline'
        sweep_name: Name of the sweep
        param_value: Value of the parameter being swept
        save_name: Filename for saving the plot
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # UAV trajectory
    uav_pos = uav.position.cpu().numpy()
    
    # Plot shaded path
    ax.plot(uav_pos[0, :], uav_pos[1, :], 'b-', alpha=0.6, linewidth=2.5, label='UAV Trajectory', zorder=3)
    ax.fill_between(uav_pos[0, :], uav_pos[1, :] - 2, uav_pos[1, :] + 2, 
                     alpha=0.2, color='blue', zorder=1)
    
    # UAV start position
    ax.scatter(uav_pos[0, 0], uav_pos[1, 0], c='blue', s=300, marker='^', 
               edgecolors='black', linewidths=2.5, label='UAV Start', zorder=5)
    
    # UAV end position
    ax.scatter(uav_pos[0, -1], uav_pos[1, -1], c='darkblue', s=300, marker='v', 
               edgecolors='black', linewidths=2.5, label='UAV End', zorder=5)
    
    # Base station
    bs_pos = bs.position.cpu().numpy()
    ax.scatter(bs_pos[0], bs_pos[1], c='red', s=400, marker='s',
               edgecolors='black', linewidths=2.5, label='Base Station', zorder=5)
    ax.text(bs_pos[0] + 5, bs_pos[1] + 5, 'BS', fontsize=12, fontweight='bold')
    
    # IoT devices
    for i, iot in enumerate(iot_devices):
        iot_pos = iot.position.cpu().numpy()
        ax.scatter(iot_pos[0], iot_pos[1], c='green', s=150, marker='o',
                   edgecolors='black', linewidths=1.5, label='IoT Device' if i == 0 else '', zorder=4)
        ax.text(iot_pos[0] + 3, iot_pos[1] + 3, f'IoT{i}', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('X Position (m)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y Position (m)', fontsize=14, fontweight='bold')
    
    title = f'UAV Trajectory Visualization\n{optimization_mode.capitalize()} Optimization - {sweep_name}'
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
    
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    # Save
    save_path = f'results/{save_name}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"     📊 Saved trajectory: {save_path}")
    plt.close(fig)


def run_trial(config, trial_seed, device='cuda', save_trajectory=False, 
              optimization_mode='online', sweep_name='', param_value=None):
    """Run a single trial with new random IoT device positions.
    
    Each trial generates:
    - New random IoT device positions
    - New Poisson-distributed task arrivals
    - New random task parameters
    
    Args:
        save_trajectory: If True, save trajectory visualization for this trial
        optimization_mode: 'online' or 'offline'
        sweep_name: Name of current sweep (for visualization)
        param_value: Current parameter value (for visualization)
    """
    # Set seeds for reproducibility
    np.random.seed(trial_seed)
    torch.manual_seed(trial_seed)
    
    # Create scenario with new random IoT positions for this trial
    uav, bs, iot_devices, time_indices, params, optimization_tasks = create_scenario(config, trial_seed, device)
    
    # ============================================================================
    # TRACE: Trajectory Metrics
    # ============================================================================
    if uav is not None:
        uav_pos_np = uav.position.cpu().numpy()  # (2, T)
        uav_vel_np = uav.velocity.cpu().numpy()  # (2, T)
        
        # Compute trajectory characteristics
        distances = np.sqrt(np.sum(np.diff(uav_pos_np, axis=1)**2, axis=0))
        total_distance = np.sum(distances)
        avg_distance_per_step = np.mean(distances)
        
        velocities = np.sqrt(np.sum(uav_vel_np**2, axis=0))
        avg_velocity = np.mean(velocities)
        max_velocity = np.max(velocities)
        
        # Compute average distance from UAV to all IoT devices over time
        iot_positions = torch.stack([iot.position for iot in iot_devices], dim=0)  # (N, 2)
        uav_positions_t = uav.position.T  # (T, 2)
        all_distances = torch.cdist(iot_positions, uav_positions_t)  # (N, T)
        avg_dist_to_devices = all_distances.mean().item()
        min_dist_to_devices = all_distances.min().item()
        
        print(f"\n     [TRACE] === TRAJECTORY METRICS ({optimization_mode.upper()}) ===")
        print(f"     • Total distance traveled: {total_distance:.2f} m")
        print(f"     • Avg distance per step: {avg_distance_per_step:.2f} m")
        print(f"     • Avg velocity: {avg_velocity:.2f} m/s (max: {max_velocity:.2f} m/s)")
        print(f"     • Avg distance to IoT devices: {avg_dist_to_devices:.2f} m")
        print(f"     • Min distance to any IoT device: {min_dist_to_devices:.2f} m")
    else:
        print(f"\n     [TRACE] === NO UAV DEPLOYMENT ===")
        print(f"     • Using only local execution and base station offloading")
    
    # Use tasks from optimization if available, otherwise generate new ones
    if optimization_tasks is not None:
        print(f"\n     [TRACE] Using tasks from optimization phase (avoiding mismatch)")
        tasks, task_owners = optimization_tasks
    else:
        # Generate tasks using Poisson process (for circular baseline or no_uav)
        tasks, task_owners = generate_tasks(iot_devices, config)
    
    task_iot_devices = [iot_devices[owner] for owner in task_owners]
    
    # Count tasks per device
    tasks_per_device = {i: task_owners.count(i) for i in range(len(iot_devices))}
    print(f"\n     [TRACE] === TASK GENERATION ===")
    print(f"     • Generated {len(tasks)} tasks (Poisson)")
    print(f"     • Distribution: {dict(tasks_per_device)}")
    
    # Compute task statistics
    task_sizes = [t.length_bits / 1e6 for t in tasks]  # Mb
    task_slacks = [t.slack for t in tasks]  # seconds
    task_densities = [t.computation_density for t in tasks]  # cycles/bit
    
    print(f"     • Task sizes: min={min(task_sizes):.2f} Mb, max={max(task_sizes):.2f} Mb, avg={np.mean(task_sizes):.2f} Mb")
    print(f"     • Task slacks: min={min(task_slacks):.2f} s, max={max(task_slacks):.2f} s, avg={np.mean(task_slacks):.2f} s")
    print(f"     • Computation density: min={min(task_densities):.0f}, max={max(task_densities):.0f}, avg={np.mean(task_densities):.0f} cycles/bit")
    
    print(f"\n     • Running greedy offloading scheduler...")
    # Run greedy offloading
    results = greedy_offloading_batch(
        tasks=tasks,
        iot_devices=task_iot_devices,
        uav=uav,
        bs=bs,
        time_indices=time_indices,
        params=params,
        device=device
    )
    
    # ============================================================================
    # TRACE: Offloading Decision Analysis
    # ============================================================================
    decisions = results['decisions']
    completions = results['completions']
    
    # Count decisions by target
    decision_counts = {'local': 0, 'bs': 0, 'uav': 0, 'none': 0}
    completion_by_target = {'local': [], 'bs': [], 'uav': [], 'none': []}
    times_by_target = {'local': [], 'bs': [], 'uav': []}
    channel_probs_by_target = {'local': [], 'bs': [], 'uav': []}
    
    for (target, total_time, channel_prob), completion in zip(decisions, completions):
        decision_counts[target] += 1
        completion_by_target[target].append(completion)
        if target != 'none':
            times_by_target[target].append(total_time)
            channel_probs_by_target[target].append(channel_prob)
    
    print(f"\n     [TRACE] === OFFLOADING DECISIONS ===")
    print(f"     • Local: {decision_counts['local']} ({decision_counts['local']/len(tasks)*100:.1f}%)")
    print(f"     • BS: {decision_counts['bs']} ({decision_counts['bs']/len(tasks)*100:.1f}%)")
    print(f"     • UAV: {decision_counts['uav']} ({decision_counts['uav']/len(tasks)*100:.1f}%)")
    print(f"     • None (infeasible): {decision_counts['none']} ({decision_counts['none']/len(tasks)*100:.1f}%)")
    
    print(f"\n     [TRACE] === COMPLETION RATES BY TARGET ===")
    for target in ['local', 'bs', 'uav']:
        if completion_by_target[target]:
            completed = sum(completion_by_target[target])
            total = len(completion_by_target[target])
            rate = completed / total * 100 if total > 0 else 0
            print(f"     • {target.upper()}: {completed}/{total} ({rate:.1f}%)")
            
            if times_by_target[target]:
                avg_time = np.mean(times_by_target[target])
                avg_channel = np.mean(channel_probs_by_target[target])
                print(f"       - Avg serving time: {avg_time:.3f} s")
                print(f"       - Avg channel prob: {avg_channel:.3f}")
    
    print(f"\n     [TRACE] === OVERALL RESULTS ===")
    print(f"     • Total completed: {results['total_completed']}/{len(tasks)}")
    print(f"     • Completion rate: {results['completion_rate']*100:.2f}%")
    
    # Save trajectory visualization if requested (skip for no_uav baseline)
    if save_trajectory and uav is not None:
        param_str = f"{param_value:.1f}" if isinstance(param_value, float) else str(param_value)
        save_name = f'trajectory_{sweep_name}_{optimization_mode}_param{param_str}_seed{trial_seed}'
        visualize_trajectory(uav, bs, iot_devices, optimization_mode, sweep_name, param_value, save_name)
    
    return results['completion_rate']


def run_sweep(sweep_name, sweep_config, base_config, optimization_mode, device='cuda'):
    """Run a parameter sweep with specified optimization mode.
    
    Args:
        optimization_mode: 'online', 'offline', 'circular', or 'both'
    """
    mode_name = optimization_mode.upper() if optimization_mode != 'both' else 'BOTH'
    print(f"\n{'='*70}")
    print(f"SWEEP: {sweep_config['title']} [{mode_name} OPTIMIZATION]")
    print(f"{'='*70}")
    
    values = sweep_config['values']
    param = sweep_config['param']
    
    completion_rates = []
    completion_stds = []
    
    for val_idx, val in enumerate(tqdm(values, desc=f"{sweep_name} ({optimization_mode})")):
        # Update configuration
        config = base_config.copy()
        
        if param == 'task_size':
            config['task_size_min'], config['task_size_max'] = val
        elif param == 'cycles_per_bit':
            config['cycles_per_bit_min'], config['cycles_per_bit_max'] = val
        else:
            config[param] = val
        
        # Set optimization mode
        if optimization_mode == 'circular':
            config['use_circular_baseline'] = True
            config['optimize_trajectory'] = False
        elif optimization_mode == 'online':
            config['use_circular_baseline'] = False
            config['online_optimization'] = True
        elif optimization_mode == 'offline':
            config['use_circular_baseline'] = False
            config['online_optimization'] = False
        elif optimization_mode == 'no_uav':
            config['no_uav_baseline'] = True
            config['optimize_trajectory'] = False
        # if 'both', use base_config setting
        
        # Run trials
        trial_results = []
        for trial in range(config['trials']):
            print(f"\n{'='*80}")
            print(f"  Trial {trial+1}/{config['trials']} ({optimization_mode.upper()})")
            print(f"{'='*80}")
            seed = config['seed_base'] + trial
            
            # Save trajectory visualization for the first trial of the LAST parameter value only
            is_last_value = (val_idx == len(values) - 1)
            save_traj = (trial == 0 and is_last_value)
            
            completion_rate = run_trial(
                config, seed, device, 
                save_trajectory=save_traj,
                optimization_mode=optimization_mode,
                sweep_name=sweep_name,
                param_value=val
            )
            trial_results.append(completion_rate)
            print(f"\n     ✓ Trial {trial+1} complete: {completion_rate*100:.1f}% completion rate")
            print(f"{'='*80}\n")
        
        # Compute statistics
        mean_rate = np.mean(trial_results)
        std_rate = np.std(trial_results)
        
        completion_rates.append(mean_rate)
        completion_stds.append(std_rate)
        
        # Display for tuples (task_size, cycles_per_bit)
        if param == 'task_size':
            avg_size = (val[0] + val[1]) / 2 / 1e6
            print(f"\n  📊 [{optimization_mode.upper():8s}] Task Size {avg_size:4.1f} Mb  →  Completion: {mean_rate*100:5.1f}% ± {std_rate*100:4.1f}%")
        elif param == 'cycles_per_bit':
            avg_cycles = (val[0] + val[1]) / 2
            print(f"\n  📊 [{optimization_mode.upper():8s}] Cycles/Bit {avg_cycles:5.0f}  →  Completion: {mean_rate*100:5.1f}% ± {std_rate*100:4.1f}%")
        else:
            val_display = val / sweep_config.get('scale', 1.0) if 'scale' in sweep_config else val
            print(f"\n  📊 [{optimization_mode.upper():8s}] {param}={val_display:.1f}  →  Completion: {mean_rate*100:5.1f}% ± {std_rate*100:4.1f}%")
    
    return completion_rates, completion_stds


def plot_sweep_results(values, completion_rates, completion_stds, 
                       sweep_config, save_name, online_data=None):
    """Plot sweep results.
    
    Args:
        online_data: Optional dict with 'rates' and 'stds' for online optimization comparison
    """
    # Convert values for display
    if sweep_config['param'] in ['task_size', 'cycles_per_bit']:
        x_vals = [(v[0] + v[1]) / 2 for v in values]
    else:
        x_vals = values
    
    if 'scale' in sweep_config:
        x_vals = [v / sweep_config['scale'] for v in x_vals]
    
    # Convert to percentages
    y_vals = [r * 100 for r in completion_rates]
    y_stds = [s * 100 for s in completion_stds]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if online_data is not None:
        # Comparison plot
        online_y = [r * 100 for r in online_data['rates']]
        online_stds = [s * 100 for s in online_data['stds']]
        
        ax.errorbar(x_vals, online_y, yerr=online_stds, 
                    marker='o', markersize=8, linewidth=2, capsize=5,
                    label='Online (Receding Horizon)', color='#2E86AB')
        ax.errorbar(x_vals, y_vals, yerr=y_stds, 
                    marker='s', markersize=8, linewidth=2, capsize=5,
                    label='Offline (Batch Gradient)', color='#A23B72')
    else:
        # Single plot
        ax.errorbar(x_vals, y_vals, yerr=y_stds, 
                    marker='o', markersize=8, linewidth=2, capsize=5,
                    label='Task Completion Rate', color='#2E86AB')
    
    ax.set_xlabel(sweep_config['xlabel'], fontsize=14, fontweight='bold')
    ylabel = sweep_config.get('ylabel', 'Task Completion Rate (%)')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    title = sweep_config['title']
    if online_data is not None:
        title += '\nComparison: Online vs Offline Trajectory Optimization'
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    
    # Format
    ax.tick_params(labelsize=12)
    
    # Add y-axis range starting from 0
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    # Save
    save_path = f'results/{save_name}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {save_path}")
    
    return fig


def plot_four_way_comparison(values, no_uav_rates, no_uav_stds,
                              circular_rates, circular_stds,
                              online_rates, online_stds,
                              offline_rates, offline_stds,
                              sweep_config, save_name):
    """Plot four-way comparison: No UAV vs Circular vs Online vs Offline."""
    # Convert values for display
    if sweep_config['param'] in ['task_size', 'cycles_per_bit']:
        x_vals = [(v[0] + v[1]) / 2 for v in values]
    else:
        x_vals = values
    
    if 'scale' in sweep_config:
        x_vals = [v / sweep_config['scale'] for v in x_vals]
    
    # Convert to percentages
    no_uav_y = [r * 100 for r in no_uav_rates]
    no_uav_y_stds = [s * 100 for s in no_uav_stds]
    
    circular_y = [r * 100 for r in circular_rates]
    circular_y_stds = [s * 100 for s in circular_stds]
    
    online_y = [r * 100 for r in online_rates]
    online_y_stds = [s * 100 for s in online_stds]
    
    offline_y = [r * 100 for r in offline_rates]
    offline_y_stds = [s * 100 for s in offline_stds]
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Four lines
    ax.errorbar(x_vals, no_uav_y, yerr=no_uav_y_stds, 
                marker='x', markersize=10, linewidth=2.5, capsize=5,
                label='No UAV (Local + BS only)', 
                color='#E07A5F', linestyle=':', alpha=0.8)
    
    ax.errorbar(x_vals, circular_y, yerr=circular_y_stds, 
                marker='D', markersize=8, linewidth=2.5, capsize=5,
                label='UAV Circular Baseline', 
                color='#808080', linestyle='--', alpha=0.8)
    
    ax.errorbar(x_vals, online_y, yerr=online_y_stds, 
                marker='o', markersize=8, linewidth=2.5, capsize=5,
                label='UAV Online Optimization', color='#2E86AB')
    
    ax.errorbar(x_vals, offline_y, yerr=offline_y_stds, 
                marker='s', markersize=8, linewidth=2.5, capsize=5,
                label='UAV Offline Optimization', color='#A23B72')
    
    ax.set_xlabel(sweep_config['xlabel'], fontsize=14, fontweight='bold')
    ylabel = sweep_config.get('ylabel', 'Task Completion Rate (%)')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    
    title = sweep_config['title']
    title += '\nComparison: No UAV vs UAV Deployment (Circular, Online, Offline)'
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=12, loc='best', framealpha=0.95, shadow=True)
    
    # Format
    ax.tick_params(labelsize=12)
    
    # Add y-axis range starting from 0
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    # Save
    save_path = f'results/{save_name}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved 3-way comparison: {save_path}")
    
    return fig


# ========================================================================
# Main Execution
# ========================================================================

if __name__ == '__main__':
    print(f"\n" + "─" * 80)
    print(f"  SIMULATION CONFIGURATION")
    print("─" * 80)
    print(f"  UAV Parameters:")
    print(f"    • CPU Frequency:  {BASE_CONFIG['uav_cpu_frequency']/1e9:.1f} GHz")
    print(f"    • Max Velocity:   {BASE_CONFIG['uav_max_velocity']:.1f} m/s")
    print(f"    • Height:         {BASE_CONFIG['uav_height']:.1f} m")
    print(f"  \n  Network Parameters:")
    print(f"    • Bandwidth:      {BASE_CONFIG['BW_total']/1e6:.1f} MHz")
    print(f"    • IoT Devices:    {BASE_CONFIG['num_devices']}")
    print(f"  \n  Task Parameters:")
    print(f"    • Size Range:     {BASE_CONFIG['task_size_min']/1e6:.1f}-{BASE_CONFIG['task_size_max']/1e6:.1f} Mb")
    print(f"    • Cycles/Bit:     {BASE_CONFIG['cycles_per_bit_min']:.0f}-{BASE_CONFIG['cycles_per_bit_max']:.0f}")
    print(f"  \n  Simulation:")
    print(f"    • Trials/Point:   {BASE_CONFIG['trials']}")
    print(f"    • Time Steps:     {BASE_CONFIG['T']}")
    print("─" * 80)
    print(f"  RUNNING FOUR-WAY COMPARISON")
    print(f"    1. No UAV (Local + BS only)")
    print(f"    2. Circular Baseline (No Optimization)")
    print(f"    3. Online Optimization (Receding Horizon)")
    print(f"    4. Offline Optimization (Batch Gradient)")
    print("─" * 80 + "\n")
    
    # Run all sweeps with ALL FOUR methods
    all_results = {}
    
    for sweep_name, sweep_config in SWEEP_CONFIGS.items():
        # Run NO UAV baseline
        no_uav_rates, no_uav_stds = run_sweep(
            sweep_name, sweep_config, BASE_CONFIG, 'no_uav', device
        )
        
        # Run CIRCULAR baseline
        circular_rates, circular_stds = run_sweep(
            sweep_name, sweep_config, BASE_CONFIG, 'circular', device
        )
        
        # Run ONLINE optimization
        online_rates, online_stds = run_sweep(
            sweep_name, sweep_config, BASE_CONFIG, 'online', device
        )
        
        # Run OFFLINE optimization
        offline_rates, offline_stds = run_sweep(
            sweep_name, sweep_config, BASE_CONFIG, 'offline', device
        )
        
        all_results[sweep_name] = {
            'values': sweep_config['values'],
            'no_uav': {
                'completion_rates': no_uav_rates,
                'completion_stds': no_uav_stds,
            },
            'circular': {
                'completion_rates': circular_rates,
                'completion_stds': circular_stds,
            },
            'online': {
                'completion_rates': online_rates,
                'completion_stds': online_stds,
            },
            'offline': {
                'completion_rates': offline_rates,
                'completion_stds': offline_stds,
            }
        }
        
        # # Plot individual results
        # plot_sweep_results(
        #     sweep_config['values'],
        #     circular_rates,
        #     circular_stds,
        #     sweep_config,
        #     f'{sweep_name}_circular'
        # )
        
        # plot_sweep_results(
        #     sweep_config['values'],
        #     online_rates,
        #     online_stds,
        #     sweep_config,
        #     f'{sweep_name}_online'
        # )
        
        # plot_sweep_results(
        #     sweep_config['values'],
        #     offline_rates,
        #     offline_stds,
        #     sweep_config,
        #     f'{sweep_name}_offline'
        # )
        
        # Plot four-way comparison
        plot_four_way_comparison(
            sweep_config['values'],
            no_uav_rates,
            no_uav_stds,
            circular_rates,
            circular_stds,
            online_rates,
            online_stds,
            offline_rates,
            offline_stds,
            sweep_config,
            f'{sweep_name}_comparison'
        )
    
    print("\n" + "═" * 80)
    print("  ✓ ALL SWEEPS COMPLETED")
    print("═" * 80)
    
    # Summary
    print("\n" + "═" * 80)
    print("  PERFORMANCE SUMMARY: No UAV vs UAV Deployment")
    print("═" * 80)
    for sweep_name, results in all_results.items():
        no_uav_rates = results['no_uav']['completion_rates']
        circular_rates = results['circular']['completion_rates']
        online_rates = results['online']['completion_rates']
        offline_rates = results['offline']['completion_rates']
        
        no_uav_avg = np.mean(no_uav_rates) * 100
        circular_avg = np.mean(circular_rates) * 100
        online_avg = np.mean(online_rates) * 100
        offline_avg = np.mean(offline_rates) * 100
        
        uav_benefit = circular_avg - no_uav_avg
        online_improvement = online_avg - circular_avg
        offline_improvement = offline_avg - circular_avg
        
        print(f"\n  {sweep_name.upper().replace('_', ' ')}:")
        print(f"  ┌─────────────────────────────────────────────────────────┐")
        print(f"  │  NO UAV    │  Min: {min(no_uav_rates)*100:5.1f}%  Max: {max(no_uav_rates)*100:5.1f}%  Avg: {no_uav_avg:5.1f}%  │")
        print(f"  │  CIRCULAR  │  Min: {min(circular_rates)*100:5.1f}%  Max: {max(circular_rates)*100:5.1f}%  Avg: {circular_avg:5.1f}%  │")
        print(f"  │  ONLINE    │  Min: {min(online_rates)*100:5.1f}%  Max: {max(online_rates)*100:5.1f}%  Avg: {online_avg:5.1f}%  │")
        print(f"  │  OFFLINE   │  Min: {min(offline_rates)*100:5.1f}%  Max: {max(offline_rates)*100:5.1f}%  Avg: {offline_avg:5.1f}%  │")
        print(f"  ├─────────────────────────────────────────────────────────┤")
        print(f"  │  UAV Deployment Benefit:   {uav_benefit:+6.1f}%                 │")
        print(f"  │  Online vs Circular:       {online_improvement:+6.1f}%                 │")
        print(f"  │  Offline vs Circular:      {offline_improvement:+6.1f}%                 │")
        print(f"  └─────────────────────────────────────────────────────────┘")
    
    plt.show()
