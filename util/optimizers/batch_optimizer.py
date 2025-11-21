"""
Batch optimizer for parallel processing of multiple scenarios on GPU
Author: GitHub Copilot
Date: January 2025

This module provides GPU-accelerated batch optimization for running
multiple independent optimization problems in parallel.
"""

import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from ..clustering import k_means_uav, hierarchical_uav
from . import optimize_network


def batch_optimize_trials(user_pos_trials, M, N, AREA, H_M, H, F, P_T, P_N, 
                          MAX_ITER, TOL, BW_total, R_MIN, D_m, C_m, f_UAV, f_user,
                          clustering_method='kmeans', device='cuda', num_workers=2):
    """
    Run optimization for multiple trials in parallel using threading.
    
    Since SciPy optimization releases GIL, we can use threads for parallelism.
    GPU operations are synchronized automatically by PyTorch.
    
    Args:
        user_pos_trials: List of user position tensors
        M, N, AREA, H_M, H, F, P_T, P_N: System parameters
        MAX_ITER, TOL: Clustering parameters
        BW_total, R_MIN: Resource parameters
        D_m, C_m, f_UAV, f_user: MEC parameters
        clustering_method: 'kmeans' or 'hierarchical'
        device: 'cuda' or 'cpu'
        num_workers: Number of parallel threads (default: 2 to avoid GPU contention)
    
    Returns:
        baseline_results: List of baseline throughputs
        optimized_results: List of optimized throughputs
    """
    baseline_results = []
    optimized_results = []
    
    def process_trial(trial_idx, user_pos):
        """Process a single trial"""
        # Clustering
        if clustering_method == 'kmeans':
            uav_pos, _, baseline_throughput = k_means_uav(
                user_pos, M, N, AREA, H_M, H, F, P_T, P_N, MAX_ITER, TOL, BW_total,
                D_m, C_m, f_UAV, f_user, device=device
            )
        else:
            uav_pos, _, baseline_throughput = hierarchical_uav(
                user_pos, N, H_M, H, F, P_T, P_N, BW_total,
                D_m, C_m, f_UAV, f_user, device=device
            )
        
        baseline = baseline_throughput.cpu().item() if isinstance(baseline_throughput, torch.Tensor) else baseline_throughput
        
        # Optimization
        _, _, _, _, optimized_throughput = optimize_network(
            M, N, uav_pos, BW_total, AREA, user_pos, H_M, H, F, P_T, P_N, R_MIN,
            D_m, C_m, f_UAV, f_user, device=device
        )
        
        return trial_idx, baseline, optimized_throughput
    
    # Use ThreadPoolExecutor for parallel execution
    # Note: num_workers=2 recommended to avoid GPU contention
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all trials
        futures = {
            executor.submit(process_trial, idx, user_pos): idx
            for idx, user_pos in enumerate(user_pos_trials)
        }
        
        # Collect results as they complete
        results = [None] * len(user_pos_trials)
        for future in as_completed(futures):
            trial_idx, baseline, optimized = future.result()
            results[trial_idx] = (baseline, optimized)
        
        # Separate baseline and optimized results
        for baseline, optimized in results:
            baseline_results.append(baseline)
            optimized_results.append(optimized)
    
    return baseline_results, optimized_results


def run_sweep_parallel(param_name, param_values, user_pos_generator_fn, 
                      M, N, AREA, H_M, H, F, P_T, P_N, MAX_ITER, TOL,
                      BW_total, R_MIN, D_m, C_m, f_UAV, f_user, TRIALS,
                      device='cuda', num_workers=2):
    """
    Run a parameter sweep with parallel trial processing.
    
    Args:
        param_name: Name of parameter being swept ('N', 'M', 'BW', etc.)
        param_values: Array of parameter values to sweep
        user_pos_generator_fn: Function(param_value, trial_idx) -> user_pos tensor
        (other args): System parameters
        TRIALS: Number of Monte Carlo trials per parameter value
        device: 'cuda' or 'cpu'
        num_workers: Number of parallel threads
    
    Returns:
        Dictionary with results for both clustering methods and baseline/optimized
    """
    results = {
        'kmeans_baseline': np.zeros(len(param_values)),
        'kmeans_optimized': np.zeros(len(param_values)),
        'hierarchical_baseline': np.zeros(len(param_values)),
        'hierarchical_optimized': np.zeros(len(param_values))
    }
    
    for i, param_val in enumerate(param_values):
        print(f"\n{param_name} = {param_val} ({i+1}/{len(param_values)})")
        
        # Generate user positions for all trials
        user_pos_trials = [user_pos_generator_fn(param_val, j) for j in range(TRIALS)]
        
        # Run K-means trials in parallel
        kmeans_base, kmeans_opt = batch_optimize_trials(
            user_pos_trials, M, N, AREA, H_M, H, F, P_T, P_N, MAX_ITER, TOL,
            BW_total, R_MIN, D_m, C_m, f_UAV, f_user,
            clustering_method='kmeans', device=device, num_workers=num_workers
        )
        
        # Run Hierarchical trials in parallel
        hier_base, hier_opt = batch_optimize_trials(
            user_pos_trials, M, N, AREA, H_M, H, F, P_T, P_N, MAX_ITER, TOL,
            BW_total, R_MIN, D_m, C_m, f_UAV, f_user,
            clustering_method='hierarchical', device=device, num_workers=num_workers
        )
        
        # Store averages
        results['kmeans_baseline'][i] = np.mean(kmeans_base)
        results['kmeans_optimized'][i] = np.mean(kmeans_opt)
        results['hierarchical_baseline'][i] = np.mean(hier_base)
        results['hierarchical_optimized'][i] = np.mean(hier_opt)
        
        print(f"  K-means: {results['kmeans_baseline'][i]:.2f} → {results['kmeans_optimized'][i]:.2f} Mbps")
        print(f"  Hierarchical: {results['hierarchical_baseline'][i]:.2f} → {results['hierarchical_optimized'][i]:.2f} Mbps")
    
    return results
