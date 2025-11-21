"""
Clustering algorithms for UAV positioning
Author: Khalil El Kaaki & Joe Abi Samra
Date: 23/10/2025
Translated to Python with GPU support using PyTorch
"""

import torch
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from ..common import p_received, association, bitrate


def k_means(user_pos, N, AREA, MAX_ITER, TOL, device='cuda'):
    """
    K-means clustering to position UAVs at cluster centroids.
    
    Args:
        user_pos: Tensor of shape (2, M) - user positions
        N: Number of UAVs/clusters
        AREA: Total area (square)
        MAX_ITER: Maximum iterations
        TOL: Convergence tolerance
        device: 'cuda' or 'cpu'
    
    Returns:
        uav_pos: Tensor of shape (2, N) - UAV positions
    """
    # Convert to tensor if needed
    if not isinstance(user_pos, torch.Tensor):
        user_pos = torch.tensor(user_pos, dtype=torch.float32, device=device)
    else:
        user_pos = user_pos.to(device)
    
    M = user_pos.shape[1]
    if user_pos.shape[0] != 2:
        raise ValueError('user_pos must have dimension 2xM.')
    
    side = AREA ** 0.5
    
    # Initialize centroids randomly
    centroids = side * torch.rand(2, N, device=device)
    
    for iter in range(MAX_ITER):
        prev_centroids = centroids.clone()
        
        # Vectorized distance computation: (M, 1) - (1, N) broadcasts to (M, N)
        dx = user_pos[0, :].unsqueeze(1) - centroids[0, :].unsqueeze(0)  # (M, N)
        dy = user_pos[1, :].unsqueeze(1) - centroids[1, :].unsqueeze(0)  # (M, N)
        distance = torch.sqrt(dx**2 + dy**2)  # (M, N)
        
        # Assign each user to closest UAV
        assoc = torch.argmin(distance, dim=1)  # (M,)
        
        # Update centroids based on new association
        for n in range(N):
            connected_users = (assoc == n)
            if connected_users.any():
                centroids[:, n] = torch.mean(user_pos[:, connected_users], dim=1)
            else:
                # Reinitialize if no users assigned
                centroids[:, n] = side * torch.rand(2, device=device)
        
        # Check convergence
        if torch.max(torch.sqrt(torch.sum((centroids - prev_centroids)**2, dim=0))) < TOL:
            break
    
    return centroids


def k_means_uav(user_pos, M, N, AREA, H_M, H, F, P_T, P_N, MAX_ITER, TOL, BW, device='cuda'):
    """
    K-means based UAV positioning with rate calculation.
    
    Args:
        user_pos: Tensor/array of shape (2, M) - user positions
        M: Number of users
        N: Number of UAVs
        AREA: Coverage area
        H_M: Mobile height
        H: UAV height
        F: Frequency
        P_T: Transmit power
        P_N: Noise power
        MAX_ITER: Maximum iterations for k-means
        TOL: Convergence tolerance
        BW: Total bandwidth
        device: 'cuda' or 'cpu'
    
    Returns:
        uav_pos: Tensor (2, N) - UAV positions
        rate: Tensor (M,) - per-user rates
        sumlink_mbps: Scalar - total throughput in Mbps
    """
    # Run k-means clustering
    uav_pos = k_means(user_pos, N, AREA, MAX_ITER, TOL, device=device)
    
    # Ensure user_pos is on the right device
    if not isinstance(user_pos, torch.Tensor):
        user_pos = torch.tensor(user_pos, dtype=torch.float32, device=device)
    else:
        user_pos = user_pos.to(device)
    
    # Calculate received power
    p_r = p_received(user_pos, uav_pos, H_M, H, F, P_T, device=device)
    
    # Get association matrix
    a = association(p_r)
    
    # Calculate per-user rates
    rate = torch.sum(bitrate(p_r, P_N, BW/M, a), dim=1)  # (M,)
    
    # Total sum rate
    sumlink = torch.sum(rate)
    sumlink_mbps = sumlink / 1e6
    
    return uav_pos, rate, sumlink_mbps


def hierarchical_uav(user_pos, N, H_M, H, F, P_T, P_N, BW, device='cuda'):
    """
    Hierarchical clustering for UAV positioning.
    
    Args:
        user_pos: Tensor/array of shape (2, M) - user positions
        N: Number of UAVs
        H_M: Mobile height
        H: UAV height
        F: Frequency
        P_T: Transmit power
        P_N: Noise power
        BW: Total bandwidth
        device: 'cuda' or 'cpu'
    
    Returns:
        uav_pos_hier: Tensor (2, N) - UAV positions
        baseline_br: Tensor (M,) - per-user rates
        sumrate_mbps: Scalar - total throughput in Mbps
    """
    # Convert to numpy for scipy hierarchical clustering
    if isinstance(user_pos, torch.Tensor):
        user_pos_np = user_pos.cpu().numpy()
    else:
        user_pos_np = np.array(user_pos)
    
    M = user_pos_np.shape[1]
    
    # Hierarchical clustering using scipy (on CPU)
    # pdist expects (M, 2) format
    user_pos_T = user_pos_np.T  # (M, 2)
    
    # Build linkage tree using Ward's method
    Z = linkage(user_pos_T, method='ward')
    
    # Assign users to N clusters
    cluster_idx = fcluster(Z, N, criterion='maxclust')  # 1..N for each user
    
    # Compute UAV positions as cluster centroids
    uav_pos_hier = np.zeros((2, N))
    for i in range(1, N + 1):  # cluster_idx is 1-indexed
        members = user_pos_np[:, cluster_idx == i]
        if members.shape[1] > 0:
            uav_pos_hier[:, i-1] = np.mean(members, axis=1)
        else:
            # If empty cluster, place randomly
            uav_pos_hier[:, i-1] = np.random.rand(2) * (9e6 ** 0.5)  # fallback
    
    # Convert back to torch
    uav_pos_hier = torch.tensor(uav_pos_hier, dtype=torch.float32, device=device)
    user_pos_torch = torch.tensor(user_pos_np, dtype=torch.float32, device=device)
    
    # Calculate rates
    p_r = p_received(user_pos_torch, uav_pos_hier, H_M, H, F, P_T, device=device)
    a = association(p_r)
    baseline_br = torch.sum(bitrate(p_r, P_N, BW/M, a), dim=1)  # (M,)
    
    sumlink = torch.sum(baseline_br)
    sumrate_mbps = sumlink / 1e6
    
    return uav_pos_hier, baseline_br, sumrate_mbps
