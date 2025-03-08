import numpy as np
import matplotlib.pyplot as plt
import os
import json

# Try to import both models - handle gracefully if they don't exist yet
try:
    from params.ica_model import velocity as ica_velocity, PERIOD as ICA_PERIOD, T_MIN as ICA_T_MIN, \
                                TRANSITION_FRACTION as ICA_TRANSITION, MODEL_TYPE as ICA_MODEL_TYPE
    ica_model_loaded = True
except ImportError:
    print("ICA model not found. Run test3.py first to generate the model.")
    ica_model_loaded = False

try:
    from params.cca_model import velocity as cca_velocity, PERIOD as CCA_PERIOD, T_MIN as CCA_T_MIN, \
                                TRANSITION_FRACTION as CCA_TRANSITION, MODEL_TYPE as CCA_MODEL_TYPE
    cca_model_loaded = True
except ImportError:
    print("CCA model not found. Run test3.py first to generate the model.")
    cca_model_loaded = False

# Check if at least one model is available
if not (ica_model_loaded or cca_model_loaded):
    print("No models found. Please run test3.py first to generate velocity profile models.")
    exit(1)

# Create figure for plotting
plt.figure(figsize=(16, 10))

# Define time points for evaluation
# For single cycle
num_points = 1000

# Plot each available model
if ica_model_loaded:
    # Plot a single cycle
    plt.subplot(2, 2, 1)
    t_single = np.linspace(ICA_T_MIN, ICA_T_MIN + ICA_PERIOD, num_points)
    v_single = np.array([ica_velocity(t) for t in t_single])
    
    plt.plot(t_single, v_single, 'b-', linewidth=2.5)
    
    # Mark transition point
    t_transition = ICA_T_MIN + ICA_TRANSITION * ICA_PERIOD
    transition_idx = np.argmin(np.abs(t_single - t_transition))
    plt.axvline(x=t_transition, color='r', linestyle='--')
    plt.plot(t_transition, v_single[transition_idx], 'ro', markersize=8)
    
    plt.title(f"ICA Velocity Profile - Single Cycle\n{ICA_MODEL_TYPE}", fontsize=12)
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (normalized)")
    plt.grid(alpha=0.3)
    
    # Plot multiple cycles
    plt.subplot(2, 2, 2)
    num_cycles = 3
    t_multi = np.linspace(ICA_T_MIN, ICA_T_MIN + num_cycles * ICA_PERIOD, num_points * num_cycles)
    v_multi = np.array([ica_velocity(t) for t in t_multi])
    
    plt.plot(t_multi, v_multi, 'b-', linewidth=2)
    
    # Mark cycle and transition boundaries
    for i in range(num_cycles + 1):
        cycle_boundary = ICA_T_MIN + i * ICA_PERIOD
        plt.axvline(x=cycle_boundary, color='k', linestyle='-', alpha=0.3)
        
    for i in range(num_cycles):
        transition_point = ICA_T_MIN + i * ICA_PERIOD + ICA_TRANSITION * ICA_PERIOD
        plt.axvline(x=transition_point, color='r', linestyle='--', alpha=0.5)
    
    plt.title(f"ICA Velocity Profile - {num_cycles} Cycles", fontsize=12)
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (normalized)")
    plt.grid(alpha=0.3)

if cca_model_loaded:
    # Plot a single cycle
    plt.subplot(2, 2, 3)
    t_single = np.linspace(CCA_T_MIN, CCA_T_MIN + CCA_PERIOD, num_points)
    v_single = np.array([cca_velocity(t) for t in t_single])
    
    plt.plot(t_single, v_single, 'g-', linewidth=2.5)
    
    # Mark transition point
    t_transition = CCA_T_MIN + CCA_TRANSITION * CCA_PERIOD
    transition_idx = np.argmin(np.abs(t_single - t_transition))
    plt.axvline(x=t_transition, color='r', linestyle='--')
    plt.plot(t_transition, v_single[transition_idx], 'ro', markersize=8)
    
    plt.title(f"CCA Velocity Profile - Single Cycle\n{CCA_MODEL_TYPE}", fontsize=12)
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (normalized)")
    plt.grid(alpha=0.3)
    
    # Plot multiple cycles
    plt.subplot(2, 2, 4)
    num_cycles = 3
    t_multi = np.linspace(CCA_T_MIN, CCA_T_MIN + num_cycles * CCA_PERIOD, num_points * num_cycles)
    v_multi = np.array([cca_velocity(t) for t in t_multi])
    
    plt.plot(t_multi, v_multi, 'g-', linewidth=2)
    
    # Mark cycle and transition boundaries
    for i in range(num_cycles + 1):
        cycle_boundary = CCA_T_MIN + i * CCA_PERIOD
        plt.axvline(x=cycle_boundary, color='k', linestyle='-', alpha=0.3)
        
    for i in range(num_cycles):
        transition_point = CCA_T_MIN + i * CCA_PERIOD + CCA_TRANSITION * CCA_PERIOD
        plt.axvline(x=transition_point, color='r', linestyle='--', alpha=0.5)
    
    plt.title(f"CCA Velocity Profile - {num_cycles} Cycles", fontsize=12)
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (normalized)")
    plt.grid(alpha=0.3)

# Plot both profiles together for comparison
if ica_model_loaded and cca_model_loaded:
    plt.figure(figsize=(16, 6))
    
    # Calculate derivatives for acceleration visualization
    def calculate_derivative(t, velocity_func, dt=0.001):
        # Central difference approximation
        return [(velocity_func(t+dt) - velocity_func(t-dt))/(2*dt) for t in t]
    
    t_ica = np.linspace(ICA_T_MIN, ICA_T_MIN + ICA_PERIOD, num_points)
    v_ica = np.array([ica_velocity(t) for t in t_ica])
    a_ica = calculate_derivative(t_ica, ica_velocity)
    
    t_cca = np.linspace(CCA_T_MIN, CCA_T_MIN + CCA_PERIOD, num_points)
    v_cca = np.array([cca_velocity(t) for t in t_cca])
    a_cca = calculate_derivative(t_cca, cca_velocity)
    
    # Normalize time to percentage of cardiac cycle for comparison
    t_ica_norm = (t_ica - ICA_T_MIN) / ICA_PERIOD * 100
    t_cca_norm = (t_cca - CCA_T_MIN) / CCA_PERIOD * 100
    
    # Plot velocities
    plt.subplot(1, 2, 1)
    plt.plot(t_ica_norm, v_ica, 'b-', linewidth=2.5, label="ICA")
    plt.plot(t_cca_norm, v_cca, 'g-', linewidth=2.5, label="CCA")
    plt.axvline(x=ICA_TRANSITION*100, color='blue', linestyle='--', alpha=0.5)
    plt.axvline(x=CCA_TRANSITION*100, color='green', linestyle='--', alpha=0.5)
    plt.title("Velocity Profiles Comparison", fontsize=14)
    plt.xlabel("Cardiac Cycle (%)")
    plt.ylabel("Velocity (normalized)")
    plt.grid(alpha=0.3)
    plt.legend()
    
    # Plot acceleration
    plt.subplot(1, 2, 2)
    plt.plot(t_ica_norm, a_ica, 'b-', linewidth=2.5, label="ICA")
    plt.plot(t_cca_norm, a_cca, 'g-', linewidth=2.5, label="CCA")
    plt.axvline(x=ICA_TRANSITION*100, color='blue', linestyle='--', alpha=0.5)
    plt.axvline(x=CCA_TRANSITION*100, color='green', linestyle='--', alpha=0.5)
    plt.title("Acceleration Profiles Comparison", fontsize=14)
    plt.xlabel("Cardiac Cycle (%)")
    plt.ylabel("Acceleration (normalized)")
    plt.grid(alpha=0.3)
    plt.legend()

plt.tight_layout()
plt.savefig("velocity_profile_models.png", dpi=300)
plt.show()

# Print model information
if ica_model_loaded:
    print(f"ICA model: {ICA_MODEL_TYPE}")
    print(f"ICA period: {ICA_PERIOD:.4f} seconds")
    print(f"ICA transition at: {ICA_TRANSITION*100:.2f}% of cardiac cycle")

if cca_model_loaded:
    print(f"CCA model: {CCA_MODEL_TYPE}")
    print(f"CCA period: {CCA_PERIOD:.4f} seconds")
    print(f"CCA transition at: {CCA_TRANSITION*100:.2f}% of cardiac cycle")