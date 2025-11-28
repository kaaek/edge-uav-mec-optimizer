"""
Constants module
Author: Khalil El Kaaki & Joe Abi Samra
"""

def constants():
    """
    Returns system constants for UAV-MEC optimization
    
    Returns:
        tuple: (M, N, AREA, H, H_M, F, K, GAMMA, D_0, P_T, P_N, MAX_ITER, TOL, BW_total, R_MIN, SIDE, TRIALS,
                D_m, C_m, f_UAV, f_user)
    """
    # rng(42)  # Random seed - handled separately in Python
    M = 50              # Users
    N = 5               # UAVs
    AREA = 9e6          # meters squared
    H = 100             # height, meters
    H_M = 1.5           # mobile height, meters
    F = 500e6           # Hz (will be converted to MHz in propagation model)
    K = 30              # dB
    GAMMA = 3           # Path loss exponent
    D_0 = 1             # meters
    P_T = 30            # dBm
    P_N = -91           # dBm
    MAX_ITER = 50       # Maximum iterations
    TOL = 1e-3          # Tolerance for k-means convergence
    BW_total = 40e6     # Hz
    R_MIN = 0.2e6       # Set minimum throughput (bps) - now applies to MEC throughput
    SIDE = AREA ** 0.5  # Side length of square area
    TRIALS = 50         # Number of Monte Carlo trials
    
    # MEC-specific parameters
    D_m = 2e6           # Task data size per user (bits) - 2 Mb
    C_m = 2e9           # Computational complexity per user (CPU cycles) - 2 GHz-cycle
    f_UAV = 1e10        # UAV MEC server CPU frequency (Hz) - 10 GHz
    f_user = 1e9        # User device CPU frequency (Hz) - 1 GHz
    
    return M, N, AREA, H, H_M, F, K, GAMMA, D_0, P_T, P_N, MAX_ITER, TOL, BW_total, R_MIN, SIDE, TRIALS, D_m, C_m, f_UAV, f_user
