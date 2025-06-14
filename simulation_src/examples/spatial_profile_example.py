#!/usr/bin/env python3
"""
Example script showing how to use different spatial profiles programmatically
"""

from models.pipe_model_2D import PipeSimulation2D
from xlb import ComputeBackend, PrecisionPolicy
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pipe_run import pipe_simulation_setup

def example_poiseuille_profile():
    """Example using Poiseuille (parabolic) profile"""
    
    print("=== Poiseuille Profile Example ===")
    
    simulation = pipe_simulation_setup(
        vessel_length_mm=15.0,
        vessel_diameter_mm=6.5,
        resolution=0.02,  # mm per lattice unit
        kinematic_viscosity=0.0035/1056,  # Blood kinematic viscosity
        dt=1e-5,
        max_velocity=0.4,
        use_time_dependent_zou_he=False,
        use_non_newtonian_bgk=False,
        spatial_profile_type="poiseuille",  # Parabolic velocity profile
        spatial_profile_params={},  # No additional parameters needed
        output_path="../results/poiseuille_example"
    )
    
    # Run for short duration as example
    simulation.run_for_duration(duration_seconds=0.1, warmup_seconds=0.05)
    print(f"Results saved to: {simulation.output_dir}")


def example_blunted_paraboloid_profile():
    """Example using Blunted Paraboloid (power-law) profile for blood flow"""
    
    print("\n=== Blunted Paraboloid Profile Example ===")
    
    # Power-law parameters for blood flow
    spatial_params = {
        'n': 1.7,  # Power-law exponent (typical for blood)
        'scale_factor': 1.0  # Can be modified for time-varying flow
    }
    
    simulation = pipe_simulation_setup(
        vessel_length_mm=15.0,
        vessel_diameter_mm=6.5,
        resolution=0.02,
        kinematic_viscosity=0.0035/1056,
        dt=1e-5,
        max_velocity=0.4,
        use_time_dependent_zou_he=True,  # Good with time-dependent BC
        use_non_newtonian_bgk=True,  # Good with non-Newtonian collision
        spatial_profile_type="blunted_paraboloid",
        spatial_profile_params=spatial_params,
        output_path="../results/blunted_paraboloid_example"
    )
    
    # Run simulation
    simulation.run_for_duration(duration_seconds=0.1, warmup_seconds=0.05)
    print(f"Results saved to: {simulation.output_dir}")


def example_time_varying_profile():
    """Example showing how to update the scale factor during simulation"""
    
    print("\n=== Time-Varying Profile Example ===")
    
    # Start with blunted paraboloid profile
    spatial_params = {
        'n': 1.7,
        'scale_factor': 1.0  # Will be modified during simulation
    }
    
    simulation = pipe_simulation_setup(
        vessel_length_mm=15.0,
        vessel_diameter_mm=6.5,
        resolution=0.02,
        kinematic_viscosity=0.0035/1056,
        dt=1e-5,
        max_velocity=0.4,
        use_time_dependent_zou_he=True,
        use_non_newtonian_bgk=True,
        spatial_profile_type="blunted_paraboloid",
        spatial_profile_params=spatial_params,
        output_path="../results/time_varying_example"
    )
    
    # Example of updating the scale factor during simulation
    # This would be done inside your custom simulation loop
    import numpy as np
    
    total_steps = 100
    for i in range(total_steps):
        # Example: sinusoidal variation in scale factor
        time_s = i * simulation.dt
        scale_factor = 1.0 + 0.5 * np.sin(2 * np.pi * time_s)  # Varies between 0.5 and 1.5
        
        # Update the spatial profile scale factor
        simulation.update_spatial_profile_scale(scale_factor)
        
        # Run one simulation step
        simulation.f_0, simulation.f_1 = simulation.stepper(
            simulation.f_0, simulation.f_1, 
            simulation.bc_mask, simulation.missing_mask, i
        )
        simulation.f_0, simulation.f_1 = simulation.f_1, simulation.f_0
        
        # Post-process every 10 steps
        if i % 10 == 0:
            simulation.post_process(i)
            print(f"Step {i}: scale_factor = {scale_factor:.3f}")
    
    print(f"Results saved to: {simulation.output_dir}")


if __name__ == "__main__":
    # Run examples
    example_poiseuille_profile()
    example_blunted_paraboloid_profile()
    
    # Uncomment to run time-varying example
    # example_time_varying_profile()
    
    print("\n=== All Examples Complete ===")
