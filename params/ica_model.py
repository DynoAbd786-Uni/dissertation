# Auto-generated ICA velocity profile model
# Generated on: 2025-03-08 19:27:23

import numpy as np

# Model parameters
PROFILE_NAME = 'ICA'
MODEL_TYPE = 'hybrid_fourier_exponential'
PERIOD = 1.0
T_MIN = 0.0
T_MAX = 1.0
TRANSITION_FRACTION = 0.5081999999999982
R_SQUARED = 0.9973138824409234

# Fourier parameters (first part)
N_HARMONICS = 6
FOURIER_PARAMS = [4.132363161510114, 0.4210453153496071, 1.4257812470600395, -0.8400582824965988, 0.5890173565319362, -0.3471022091734739, -0.2519306666832078, -0.10514590703336024, -0.29545693251320554, 0.004950124903925744, -0.11909505434762727, 0.01019112593825568, -0.05314733679604989]

# Second part parameters
EXP_A = 14.353102463801251
EXP_B = 0.015294267482697658
EXP_C = -0.14857641553282003
EXP_D = 99.99999999999999
EXP_OFFSET = -10.92828271926996

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
