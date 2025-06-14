# Spatial Profile Configuration Examples
# This file shows different ways to configure spatial profiles

# Configuration for input_params dictionary in pipe_simulation_setup()

# 1. Poiseuille Profile (Default)
poiseuille_config = {
    "spatial_profile": {
        "type": "poiseuille"
        # No additional parameters needed
    }
}

# 2. Blunted Paraboloid for Blood Flow
blood_flow_config = {
    "spatial_profile": {
        "type": "blunted_paraboloid",
        "n": 1.7,  # Power-law exponent for blood
        "scale_factor": 1.0  # Base scale factor
    }
}

# 3. Blunted Paraboloid with Different Exponents
# n = 1.5: More blunted profile
# n = 1.9: Less blunted, closer to parabolic
custom_blood_config = {
    "spatial_profile": {
        "type": "blunted_paraboloid",
        "n": 1.9,  # Closer to parabolic
        "scale_factor": 1.0
    }
}

# 4. Time-Varying Blood Flow
pulsatile_config = {
    "spatial_profile": {
        "type": "blunted_paraboloid",
        "n": 1.7,
        "scale_factor": 1.0  # Will be updated during simulation
    }
}

# Usage examples in pipe_run.py:

# Example 1: Simple Poiseuille flow
simulation_poiseuille = pipe_simulation_setup(
    # ... other parameters ...
    spatial_profile_type="poiseuille",
    spatial_profile_params={}
)

# Example 2: Blood flow simulation
simulation_blood = pipe_simulation_setup(
    # ... other parameters ...
    spatial_profile_type="blunted_paraboloid", 
    spatial_profile_params={
        "n": 1.7,
        "scale_factor": 1.0
    }
)

# Example 3: Custom power-law profile
simulation_custom = pipe_simulation_setup(
    # ... other parameters ...
    spatial_profile_type="blunted_paraboloid",
    spatial_profile_params={
        "n": 1.5,  # More blunted
        "scale_factor": 0.8  # Reduced base velocity
    }
)
