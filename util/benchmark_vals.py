"""
Benchmark values for parameter sweeps
Author: Khalil El Kaaki & Joe Abi Samra
Date: 23/10/2025
Translated to Python with GPU support
"""

import numpy as np

def benchmark_vals():
    """
    Returns arrays of values to sweep over in benchmarks
    
    Returns:
        tuple: (N_vals, M_vals, BW_vals, P_t_vals, Rmin_vals, Area_vals)
    """
    N_vals = np.array([1, 2, 6, 10, 14, 18, 24, 30])
    M_vals = np.array([20, 50, 100, 200, 500, 700])
    BW_vals = np.array([20e6, 40e6, 80e6, 160e6])
    P_t_vals = np.array([20, 30, 40])  # dBm
    Rmin_vals = np.array([50e3, 200e3, 1e6])  # bps
    Area_vals = np.array([1e6, 4e6, 9e6, 16e6])  # m^2
    
    return N_vals, M_vals, BW_vals, P_t_vals, Rmin_vals, Area_vals
