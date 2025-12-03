"""
Channel reliability modeling with Rayleigh fading for UAV-MEC systems
Author: Khalil El Kaaki & Joe Abi Samra
Date: November 2025
"""

import torch
import numpy as np


def rayleigh_channel_gain(shape, scale=1.0, device='cuda'):
    """
    
    Generate Rayleigh-distributed channel gain samples.
    
    For Rayleigh fading, the amplitude follows Rayleigh distribution,
    and the power (|h|²) follows exponential distribution.
    
    Args:
        shape: Shape of output tensor
        scale: Scale parameter (sigma) of Rayleigh distribution
        device: 'cuda' or 'cpu'
    
    Returns:
        channel_gain: Channel power gain |h|² (exponentially distributed)
    """
    # Generate Rayleigh-distributed amplitude
    if device == 'cuda':
        mean = 2 * scale**2
        channel_gain = torch.distributions.Exponential(1.0 / mean).sample(shape).to(device)
    else:
        # Use numpy's rayleigh for amplitude, then square for power
        amplitude = np.random.rayleigh(scale=scale, size=shape)
        channel_gain = torch.tensor(amplitude**2, dtype=torch.float32, device=device)
    
    return channel_gain


def compute_instantaneous_snr(P_r_dBm, rayleigh_gain, noise_variance_dBm):
    """
    Compute instantaneous SNR with Rayleigh fading. Simplified for tests, is deprecated now.
    
    Args:
        P_r_dBm: Received power in dBm (from path loss model)
        rayleigh_gain: Rayleigh channel gain |h|²
        noise_variance_dBm: Noise variance in dBm
    
    Returns:
        snr_linear: Instantaneous SNR (linear scale)
    """
    # Convert to linear scale (mW)
    P_r_linear = 10.0 ** (P_r_dBm / 10.0)
    noise_variance_linear = 10.0 ** (noise_variance_dBm / 10.0)
    
    # Instantaneous SNR
    snr_linear = (P_r_linear * rayleigh_gain) / noise_variance_linear
    
    return snr_linear


def channel_failure_probability(P_r_dBm, noise_variance_dBm, snr_threshold_dB, 
                                 rayleigh_scale=1.0, device='cuda'):
    """
    Compute probability that channel quality falls below SNR threshold.
    
    For Rayleigh fading with mean channel gain:
    P(failure) = P(SNR < SNR_threshold) = 1 - exp(-SNR_threshold / SNR_avg)
    
    Args:
        P_r_dBm: Received power in dBm (tensor, any shape)
        noise_variance_dBm: Noise variance in dBm (scalar)
        snr_threshold_dB: Minimum required SNR in dB (scalar)
        rayleigh_scale: Scale parameter sigma for Rayleigh distribution
        device: 'cuda' or 'cpu'
    
    Returns:
        p_failure: Probability of channel failure (same shape as P_r_dBm)
    """
    # Convert to linear scale
    P_r_linear = 10.0 ** (P_r_dBm / 10.0)
    noise_variance_linear = 10.0 ** (noise_variance_dBm / 10.0)
    snr_threshold_linear = 10.0 ** (snr_threshold_dB / 10.0)
    
    # Average channel gain for Rayleigh: E[|h|²] = 2σ²
    mean_channel_gain = 2 * rayleigh_scale**2
    
    # Average SNR
    snr_avg = (P_r_linear * mean_channel_gain) / noise_variance_linear
    
    # Outage probability (channel failure probability)
    p_failure = 1.0 - torch.exp(-snr_threshold_linear / snr_avg)
    
    # Clamp to [0, 1] for numerical stability
    p_failure = torch.clamp(p_failure, 0.0, 1.0)
    
    return p_failure


def channel_success_probability(P_r_dBm, noise_variance_dBm, snr_threshold_dB, 
                                 rayleigh_scale=1.0, device='cuda'):
    """
    Compute probability that channel quality meets or exceeds SNR threshold.
    
    P(success) = 1 - P(failure) = exp(-SNR_threshold / SNR_avg)
    
    Args:
        P_r_dBm: Received power in dBm (tensor, any shape)
        noise_variance_dBm: Noise variance in dBm (scalar)
        snr_threshold_dB: Minimum required SNR in dB (scalar)
        rayleigh_scale: Scale parameter σ for Rayleigh distribution
        device: 'cuda' or 'cpu'
    
    Returns:
        p_success: Probability of channel success (same shape as P_r_dBm)
    """
    p_failure = channel_failure_probability(P_r_dBm, noise_variance_dBm, 
                                           snr_threshold_dB, rayleigh_scale, device)
    return 1.0 - p_failure


def dual_channel_reliability(P_r_bs_dBm, P_r_uav_dBm, noise_variance_dBm, 
                             snr_threshold_dB, rayleigh_scale=1.0, device='cuda'):
    """
    Compute overall reliability when both BS and UAV channels are available.
    
    System succeeds if at least one channel succeeds (diversity gain).
    
    P(system success) = 1 - P(both channels fail)
                      = 1 - P(BS fails) × P(UAV fails)
    
    Args:
        P_r_bs_dBm: Received power from base station in dBm (tensor, shape M or (M,1))
        P_r_uav_dBm: Received power from UAV in dBm (tensor, shape M or (M,1))
        noise_variance_dBm: Noise variance in dBm (scalar)
        snr_threshold_dB: Minimum required SNR in dB (scalar)
        rayleigh_scale: Scale parameter σ for Rayleigh distribution
        device: 'cuda' or 'cpu'
    
    Returns:
        p_system_success: Overall system reliability probability (shape M)
    """
    # Compute individual channel failure probabilities
    p_bs_fail = channel_failure_probability(P_r_bs_dBm, noise_variance_dBm, 
                                            snr_threshold_dB, rayleigh_scale, device)
    p_uav_fail = channel_failure_probability(P_r_uav_dBm, noise_variance_dBm, 
                                             snr_threshold_dB, rayleigh_scale, device)
    
    # System fails only if both channels fail (independence assumption)
    p_system_fail = p_bs_fail * p_uav_fail
    
    # System success probability
    p_system_success = 1.0 - p_system_fail
    
    return p_system_success


def multi_channel_reliability(P_r_all_dBm, noise_variance_dBm, snr_threshold_dB, 
                              rayleigh_scale=1.0, device='cuda'):
    """
    Compute system reliability with multiple channel options (BS + multiple UAVs).
    
    System succeeds if at least one channel succeeds.
    P(success) = 1 - ∏(P(channel_i fails))
    
    Args:
        P_r_all_dBm: Received powers from all servers in dBm (tensor, shape (M, K))
                     where K is number of edge servers (BS + UAVs)
        noise_variance_dBm: Noise variance in dBm (scalar)
        snr_threshold_dB: Minimum required SNR in dB (scalar)
        rayleigh_scale: Scale parameter σ for Rayleigh distribution
        device: 'cuda' or 'cpu'
    
    Returns:
        p_system_success: Overall system reliability probability (shape M)
    """
    # Compute failure probability for each channel
    p_fail_all = channel_failure_probability(P_r_all_dBm, noise_variance_dBm, 
                                             snr_threshold_dB, rayleigh_scale, device)
    
    # System fails only if all channels fail
    p_system_fail = torch.prod(p_fail_all, dim=1)  # Product over all servers
    
    # System success probability
    p_system_success = 1.0 - p_system_fail
    
    return p_system_success


def expected_channel_capacity_rayleigh(P_r_dBm, noise_variance_dBm, bandwidth, 
                                       rayleigh_scale=1.0, device='cuda'):
    """
    Compute expected channel capacity under Rayleigh fading.
    
    For Rayleigh fading, the expected capacity is:
    E[C] = BW × E[log2(1 + SNR × |h|²)]
    
    This can be approximated or computed using numerical integration.
    For simplicity, we use Jensen's inequality lower bound:
    E[C] ≈ BW × log2(1 + E[SNR])
    
    Args:
        P_r_dBm: Received power in dBm (tensor, any shape)
        noise_variance_dBm: Noise variance in dBm (scalar)
        bandwidth: Bandwidth in Hz (scalar)
        rayleigh_scale: Scale parameter σ for Rayleigh distribution
        device: 'cuda' or 'cpu'
    
    Returns:
        expected_capacity: Expected channel capacity in bps (same shape as P_r_dBm)
    """
    # Convert to linear scale
    P_r_linear = 10.0 ** (P_r_dBm / 10.0)
    noise_variance_linear = 10.0 ** (noise_variance_dBm / 10.0)
    
    # Average channel gain
    mean_channel_gain = 2 * rayleigh_scale**2
    
    # Average SNR
    snr_avg = (P_r_linear * mean_channel_gain) / noise_variance_linear
    
    # Expected capacity (Jensen's lower bound)
    expected_capacity = bandwidth * torch.log2(1.0 + snr_avg)
    
    return expected_capacity


def sample_rayleigh_channel_realization(P_r_dBm, noise_variance_dBm, bandwidth,
                                        rayleigh_scale=1.0, device='cuda'):
    """
    Sample a single channel realization with Rayleigh fading.
    
    Useful for Monte Carlo simulations.
    
    Args:
        P_r_dBm: Received power in dBm (tensor, any shape)
        noise_variance_dBm: Noise variance in dBm (scalar)
        bandwidth: Bandwidth in Hz (scalar)
        rayleigh_scale: Scale parameter σ for Rayleigh distribution
        device: 'cuda' or 'cpu'
    
    Returns:
        instantaneous_capacity: Channel capacity for this realization in bps
        channel_gain: The sampled channel gain |h|²
    """
    # Sample Rayleigh channel gain
    channel_gain = rayleigh_channel_gain(P_r_dBm.shape, rayleigh_scale, device)
    
    # Compute instantaneous SNR
    snr = compute_instantaneous_snr(P_r_dBm, channel_gain, noise_variance_dBm)
    
    # Instantaneous capacity
    instantaneous_capacity = bandwidth * torch.log2(1.0 + snr)
    
    return instantaneous_capacity, channel_gain
