from models.pipe_model_2D import PipeSimulation2D
from xlb import ComputeBackend, PrecisionPolicy
from utils.load_csv import load_csv_data
import xlb
import warp as wp
from utils.constants import load_profile_values
import numpy as np
import sys
import argparse
import os  # Add this import at the top with other imports


def check_tau_stability(tau, safety_margin=0.05):
    """
    Check if the relaxation time (tau) is within stable limits for LBM.
    
    Args:
        tau (float): Relaxation times
        safety_margin (float): Safety margin to keep tau away from stability limits
    
    Returns:
        tuple: (is_stable, message) where is_stable is a boolean and message is a string
    """
    min_tau = 0.5 + safety_margin
    max_tau = 2.0 - safety_margin
    
    if tau < min_tau:
        return (False, f"Calculated tau ({tau:.6f}) is too close to the minimum stability limit (0.5).\n"
                f"This would cause numerical instability. Please adjust your parameters\n"
                f"by decreasing the resolution, increasing the viscosity, or decreasing the time step.")
    elif tau > max_tau:
        return (False, f"Calculated tau ({tau:.6f}) is too close to the maximum stability limit (2.0).\n"
                f"This would cause numerical instability. Please adjust your parameters\n"
                f"by increasing the resolution, decreasing the viscosity, or increasing the time step.")
    
    return (True, f"Tau value ({tau:.6f}) is within stable range [{min_tau}, {max_tau}]")


def convert_velocity_to_lattice_units(velocity_physical, dx, dt):
    """
    Convert physical velocity (m/s) to lattice units per step.
    
    Args:
        velocity_physical (float): Velocity in physical units (m/s)
        dx (float): Spatial resolution in physical units (m)
        dt (float): Time step in physical units (s)
    
    Returns:
        float: Velocity in lattice units per step
    """
    # Formula: velocity_lu = velocity_physical * (dt/dx)
    return velocity_physical * (dt/dx)


def pipe_simulation_setup(
    grid_shape=(1000, 200),
    resolution=0.02,  # mm per lattice unit
    vessel_diameter_mm=4.0,  # mm
    vessel_length_mm=20.0,  # mm
    kinematic_viscosity=3.3e-6,  # m^2/s, blood kinematic viscosity
    max_velocity=0.2,  # m/s
    flow_profile_type="sinusoidal",
    dt=1e-5,  # seconds
    backend=ComputeBackend.WARP,
    precision_policy=PrecisionPolicy.FP32FP32,
    use_time_dependent_zou_he=False,
    use_non_newtonian_bgk=False,
    fps=100,
    flow_profile=None,  # Add flow_profile parameter with default None
    check_stability=True,  # Add stability check parameter
    output_path=None  # Add parameter for output path
) -> PipeSimulation2D:
    """Setup pipe simulation with configurable parameters"""

    wp.clear_kernel_cache()     # Clear kernel cache to avoid conflicts with new kernel code
    
    # Convert mm to meters
    mm_to_m = 0.001
    vessel_length_m = vessel_length_mm * mm_to_m
    vessel_diameter_m = vessel_diameter_mm * mm_to_m
    resolution_m = resolution * mm_to_m  # Fixed: use resolution instead of resolution_mm
    
    # Calculate base grid dimensions
    grid_x = int(round(vessel_length_m / resolution_m))
    grid_y = int(round(vessel_diameter_m / resolution_m))
    
    # Final grid shape (slightly larger than needed for boundary cells)
    grid_shape = (grid_x + 1, grid_y + 5) # Add extra cells for boundaries
    
    # Calculate vessel centerline
    vessel_centre_lu = grid_y // 2
    
    # Convert max_velocity from physical units (m/s) to lattice units (LU/step)
    max_velocity_lu = convert_velocity_to_lattice_units(max_velocity, resolution_m, dt)
    print(f"Converting max velocity: {max_velocity} m/s = {max_velocity_lu:.6f} lattice units per step")
    
    # Create dictionary of input parameters to pass to the simulation
    input_params = {
        "vessel_length_mm": vessel_length_mm,
        "vessel_diameter_mm": vessel_diameter_mm,
        "vessel_length_lu": grid_x,
        "vessel_diameter_lu": grid_y,
        "vessel_centre_lu": vessel_centre_lu,
        "kinematic_viscosity": kinematic_viscosity,
        "dt": dt,
        "dx": resolution_m,  # dx is in meters
        "fps": fps,
        "max_velocity": max_velocity,         # Physical velocity (m/s)
        "max_velocity_lu": max_velocity_lu,   # Lattice velocity (LU/step)
        "flow_profile": flow_profile or {"name": flow_profile_type}
    }

    # Preload selected CSV data for flow profile if warp selected
    if backend == ComputeBackend.WARP and flow_profile is not None and flow_profile.get('data') is not None:
        # Extract the pre-converted y data (already in lattice units)
        selected_profile_data_y = flow_profile['data']["y"]
        
        # Verify data is normalized and has 64 points
        if len(selected_profile_data_y) != 64:
            print(f"WARNING: Profile data has {len(selected_profile_data_y)} points, expected 64.")
        
        # Get min/max for debug info
        min_val = np.min(selected_profile_data_y)
        max_val = np.max(selected_profile_data_y)
        print(f"Loading profile data into matrices: min={min_val:.6f} LU, max={max_val:.6f} LU")
        
        # Load the values into the 4 mat44 matrices
        load_profile_values(selected_profile_data_y)
    
    # Calculate omega (relaxation parameter) from physical parameters
    dx = resolution_m  # dx in meters
    tau = 3.0 * kinematic_viscosity / (dx * dx / dt) + 0.5
    omega = 1.0 / tau
    
    # Check tau stability if enabled
    if check_stability:
        is_stable, message = check_tau_stability(tau)
        print(message)
        
        if not is_stable:
            print("\nCurrent parameters:")
            print(f"  - Resolution: {resolution} mm ({dx:.6e} m)")
            print(f"  - Time step: {dt:.6e} s")
            print(f"  - Kinematic viscosity: {kinematic_viscosity:.6e} m²/s")
            print(f"  - Relaxation time (tau): {tau:.6f}")
            print(f"  - Grid size: {grid_shape}")
            print("\nPlease adjust your parameters and try again.")
            sys.exit(1)
    
    # Create the velocity set for the simulation
    velocity_set = xlb.velocity_set.D2Q9(
        precision_policy=precision_policy, 
        backend=backend
    )
    
    # Now when creating the simulation, pass all required parameters
    simulation = PipeSimulation2D(
        omega=omega,
        grid_shape=grid_shape,
        velocity_set=velocity_set,
        backend=backend,
        precision_policy=precision_policy,
        resolution=resolution,
        input_params=input_params,
        use_time_dependent_zou_he=use_time_dependent_zou_he,
        use_non_newtonian_bgk=use_non_newtonian_bgk,
        output_path=output_path  # Pass the output path
    )
    
    return simulation


if __name__ == "__main__":
    # Add command line argument parsing
    parser = argparse.ArgumentParser(description='Pipe flow simulation with configurable boundary conditions and collision operators')
    parser.add_argument('--boundary-condition', choices=['standard', 'time-dependent'], required=True,
                        help='Boundary condition type: standard (Standard Zou-He) or time-dependent (Time-dependent Zou-He)')
    parser.add_argument('--collision-operator', choices=['standard', 'non-newtonian'], required=True,
                        help='Collision operator type: standard (Standard BGK) or non-newtonian (Non-Newtonian BGK)')
    args = parser.parse_args()
    
    dt = 1e-5  # Time step size (seconds)
    resolution_mm = 0.02  # 0.02mm resolution
    resolution_m = resolution_mm * 0.001  # Convert to meters
    vessel_diameter_mm = 6.5  # Male Common Carotid Artery size.
    vessel_length_mm = 15  # 15mm vessel length

    # Load CSV files - now with dx and dt parameters for lattice unit conversion
    flow_profile_data = load_csv_data(
        vessel_radius_mm=vessel_diameter_mm/2,  # Convert diameter to radius
        dx=resolution_m,                        # Pass grid spacing
        dt=dt,                                  # Pass time step
        resample_points=64,                     # Ensure exactly 64 points
        normalize_time=True                     # Normalize time to 1 second
    )

    # Auto-select CCA profile instead of prompting
    print("Available flow profiles:")
    for i, file_name in enumerate(flow_profile_data.keys()):
        print(f"{i+1}: {file_name}")
    
    # Look for a profile containing "Velocity profile"
    cca_profile_key = None
    for profile_key in flow_profile_data.keys():
        if "Velocity profile" in profile_key:
            cca_profile_key = profile_key
            break
    
    if cca_profile_key:
        selected_profile = cca_profile_key
        selected_data = flow_profile_data[selected_profile]
        
        # Auto-select the CCA column
        print("Available columns:")
        for i, col_name in enumerate(selected_data['y'].keys()):
            print(f"{i}: {col_name}")
        
        x_col = selected_data['x']
        
        # Find the CCA column
        cca_col_name = None
        for col_name in selected_data['y'].keys():
            if col_name == "CCA":
                cca_col_name = col_name
                break
        
        if cca_col_name:
            y_col_name = cca_col_name
        else:
            # Fallback to first column if CCA not found
            y_col_name = list(selected_data['y'].keys())[0]
            
        y_col = selected_data['y'][y_col_name]
        
        # Include profile name, selected column name, and units
        profile_name = f"{selected_profile}_{y_col_name}"
        units = selected_data.get('units', 'unknown')
        print(f"Automatically selected flow profile: {profile_name} (units: {units})")
        
        flow_profile = {
            'name': profile_name,
            'data': {'x': x_col, 'y': y_col}
        }
    else:
        # Fallback to default sinusoidal
        print("CCA profile not found. Defaulting to sinusoidal with 1Hz oscillation")
        flow_profile = {'name': 'Sinusoidal_1Hz', 'data': None}

    # Choose boundary condition type - auto-select based on args if available
    # Default to standard Zou-He
    use_time_dependent_zou_he = args.boundary_condition == 'time-dependent'
    
    # Choose collision operator type - auto-select based on args if available
    # Default to standard BGK 
    use_non_newtonian_bgk = args.collision_operator == 'non-newtonian'
    
    # Create output directory based on BC and CO settings
    bc_name = "time_dependent_zouhe" if use_time_dependent_zou_he else "standard_zouhe"
    co_name = "non_newtonian_bgk" if use_non_newtonian_bgk else "standard_bgk"
    
    # Construct output path from current working directory
    base_dir = os.getcwd()
    output_dir_name = f"{bc_name}_{co_name}"
    output_path = os.path.join(base_dir, "../results/pipe_flow", output_dir_name)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Output will be saved to: {output_path}")
    
    # Create simulation with realistic vessel parameters
    simulation = pipe_simulation_setup(
        vessel_length_mm=vessel_length_mm,        
        vessel_diameter_mm=vessel_diameter_mm,  # vessel diameter (typical cerebral artery)
        resolution=resolution_mm,    # Fixed: use resolution instead of resolution_mm
        kinematic_viscosity=0.0035/1056,  # Convert dynamic viscosity to kinematic (m²/s)
        dt=dt,                       # Time step
        fps=100,                     # Output frames per second
        flow_profile=flow_profile,   # Pass selected flow profile with name
        max_velocity=0.4,           # Maximum physical velocity in m/s
        use_time_dependent_zou_he=use_time_dependent_zou_he,  # Boundary condition type
        use_non_newtonian_bgk=use_non_newtonian_bgk,  # Collision operator type
        output_path=output_path  # Pass the output path
    )
    


    # Display configuration summary
    print("\n======= Pipe Simulation Configuration =======")
    print(f"Vessel length: {vessel_length_mm} mm")
    print(f"Vessel diameter: {vessel_diameter_mm} mm")
    print(f"Resolution: {resolution_mm} mm")
    print(f"Time step (dt): {dt} seconds")
    print(f"Kinematic viscosity: {0.0035/1056} m²/s")
    
    # Calculate and display tau and Reynolds number for user information
    dx = resolution_m
    kinematic_viscosity = 0.0035/1056
    tau = 3.0 * kinematic_viscosity / (dx * dx / dt) + 0.5
    max_velocity = 0.4  # Typical maximum blood velocity in m/s
    reynolds_number = (vessel_diameter_mm * 0.001 * max_velocity) / kinematic_viscosity
    
    print(f"Relaxation time (tau): {tau:.6f}")
    print(f"Reynolds number: {reynolds_number:.2f}")
    print(f"Boundary condition: {'Time-dependent Zou-He' if use_time_dependent_zou_he else 'Standard Zou-He'}")
    print(f"Collision operator: {'Non-Newtonian BGK' if use_non_newtonian_bgk else 'Standard BGK'}")
    print(f"Flow profile: {flow_profile['name']}")
    print(f"Maximum velocity: {simulation.input_params['max_velocity']} m/s ({simulation.input_params['max_velocity_lu']:.6f} LU/step)")
    print("===========================================\n")
    
    # Run simulation for 1 second with warmup
    print("\nRunning simulation...")
    simulation.run_for_duration(
        duration_seconds=1.0,
        warmup_seconds=2.0    # Run for 2.0 seconds before starting visualization
    )
    
    print("\nSimulation complete!")
    # Fix the incorrect reference to output_directories
    print(f"Results saved to: {simulation.output_dir}")
