# Auto-generated CCA velocity profile model
# Generated on: 2025-03-08 19:29:02

import numpy as np

# Model parameters
PROFILE_NAME = 'CCA'
MODEL_TYPE = 'hybrid_fourier_exponential'
PERIOD = 1.0
T_MIN = 0.0
T_MAX = 1.0
TRANSITION_FRACTION = 0.4297999999999979
R_SQUARED = 0.9971354618899799

# Fourier parameters (first part)
N_HARMONICS = 6
FOURIER_PARAMS = [6.329549008182083, 0.4601954968931389, 3.284830942119048, -2.2078575037791, 0.7925650406087222, -0.571925850932343, -0.6470987122051444, 0.0446993730291796, -0.5558067813261186, 0.0650955682014047, -0.2313497966606127, 0.03057054241273849, -0.11492465324559281]

# Second part parameters
EXP_A = 1.8200881664887827
EXP_B = 5.949764710781138
EXP_C = -1.4149632179888854
EXP_D = 65.18033566066771
EXP_OFFSET = 3.745201685507205

# Function to evaluate the velocity at any time
def velocity(t):
    # Make time cyclic
    t_cycle = T_MIN + ((t - T_MIN) % PERIOD)
    t_transition = T_MIN + TRANSITION_FRACTION * PERIOD
    
    # First part (Fourier)
    if t_cycle <= t_transition:
        a0 = FOURIER_PARAMS[0]
        result = a0
        scaled_period = PERIOD * TRANSITION_FRACTION
        for i in range(1, N_HARMONICS + 1):
            a = FOURIER_PARAMS[2*i-1]
            b = FOURIER_PARAMS[2*i]
            result += a * np.cos(2*np.pi*i*(t_cycle-T_MIN)/scaled_period) + \
                     b * np.sin(2*np.pi*i*(t_cycle-T_MIN)/scaled_period)
        return result
    
    # Second part
    else:
        t_norm = (t_cycle - t_transition) / (T_MAX - t_transition)
        # Exponential method
        result = EXP_A * np.exp(-EXP_B * t_norm) + \
                EXP_C * np.exp(-EXP_D * t_norm) + EXP_OFFSET
        return result
