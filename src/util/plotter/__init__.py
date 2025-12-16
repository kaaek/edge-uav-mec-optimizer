"""
Plotting functions for UAV network visualization
Author: Khalil El Kaaki & Joe Abi Samra
Date: November 2025
"""

import matplotlib.pyplot as plt
import torch
import numpy as np
from ..common import p_received, association


def plot_network(user_pos, uav_pos, H_M, H, F, P_T, fig_title='Network Map'):
    """
    Plot UAV and user positions with association lines.
    
    Args:
        user_pos: User positions (2, M) - numpy array or tensor
        uav_pos: UAV positions (2, N) - numpy array or tensor
        H_M, H, F, P_T: Physical parameters
        fig_title: Figure title
    """
    # Convert to numpy if tensors
    if isinstance(user_pos, torch.Tensor):
        user_pos_np = user_pos.cpu().numpy()
    else:
        user_pos_np = np.array(user_pos)
    
    if isinstance(uav_pos, torch.Tensor):
        uav_pos_np = uav_pos.cpu().numpy()
    else:
        uav_pos_np = np.array(uav_pos)
    
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    plt.suptitle(fig_title)
    
    # Plot users and UAVs
    plt.scatter(user_pos_np[0, :], user_pos_np[1, :], s=50, c='b', marker='o', label='Users')
    plt.scatter(uav_pos_np[0, :], uav_pos_np[1, :], s=100, c='r', marker='x', linewidths=2, label='UAVs')
    
    # Calculate associations
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    user_pos_torch = torch.tensor(user_pos_np, dtype=torch.float32, device=device)
    uav_pos_torch = torch.tensor(uav_pos_np, dtype=torch.float32, device=device)
    
    P_r = p_received(user_pos_torch, uav_pos_torch, H_M, H, F, P_T, device=device)
    A = association(P_r)
    A_np = A.cpu().numpy()
    
    M = user_pos_np.shape[1]
    
    # Draw association lines
    for m in range(M):
        n_assoc = np.where(A_np[m, :] == 1)[0]
        if len(n_assoc) > 0:
            n_assoc = n_assoc[0]
            plt.plot([user_pos_np[0, m], uav_pos_np[0, n_assoc]],
                    [user_pos_np[1, m], uav_pos_np[1, n_assoc]],
                    'y--', alpha=0.5, linewidth=0.5)
    
    plt.legend()
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title("Network Map")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig


def plot_sweep(x, sumrate_kmeans_ref_arr, sumrate_kmeans_opt_arr, 
               sumrate_hier_ref_arr, sumrate_hier_opt_arr, 
               xlabel_text, fig_title):
    """
    Plot sweep results comparing different methods.
    
    Args:
        x: X-axis values
        sumrate_kmeans_ref_arr: K-means reference sum rates
        sumrate_kmeans_opt_arr: K-means optimized sum rates
        sumrate_hier_ref_arr: Hierarchical reference sum rates
        sumrate_hier_opt_arr: Hierarchical optimized sum rates
        xlabel_text: X-axis label
        fig_title: Figure title
    """
    # Convert to numpy if needed
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    if isinstance(sumrate_kmeans_ref_arr, torch.Tensor):
        sumrate_kmeans_ref_arr = sumrate_kmeans_ref_arr.cpu().numpy()
    if isinstance(sumrate_kmeans_opt_arr, torch.Tensor):
        sumrate_kmeans_opt_arr = sumrate_kmeans_opt_arr.cpu().numpy()
    if isinstance(sumrate_hier_ref_arr, torch.Tensor):
        sumrate_hier_ref_arr = sumrate_hier_ref_arr.cpu().numpy()
    if isinstance(sumrate_hier_opt_arr, torch.Tensor):
        sumrate_hier_opt_arr = sumrate_hier_opt_arr.cpu().numpy()
    
    fig = plt.figure(figsize=(10, 6))
    
    plt.plot(x, sumrate_kmeans_ref_arr, '-o', label='K-Means Reference')
    plt.plot(x, sumrate_kmeans_opt_arr, '-x', label='K-Means Optimized')
    plt.plot(x, sumrate_hier_ref_arr, '-s', label='Hierarchical Reference')
    plt.plot(x, sumrate_hier_opt_arr, '-d', label='Hierarchical Optimized')
    
    plt.xlabel(xlabel_text)
    plt.ylabel('Sum Rate (Mbps)')
    plt.title(fig_title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig
