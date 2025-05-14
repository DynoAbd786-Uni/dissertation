from models.aneurysm_model_2D import AneurysmSimulation2D
from xlb import ComputeBackend, PrecisionPolicy
from utils.load_csv import load_csv_data
from utils.directory_utils import delete_directory_if_exists
import xlb
import warp as wp
from utils.constants import load_profile_values
import numpy as np
import os
import argparse


def aneurysm_simulation_setup(
    vessel_length_mm=10,
    vessel_diameter_mm=4,
    bulge_horizontal_mm=6,
    bulge_vertical_mm=4,
    resolution_mm=0.01,
    dynamic_viscosity=0.0035,
    blood_density=1056,
    dt=5e-7,
    fps=100,
    flow_profile=None,
    output_path=None,
    save_wss_png=True,  # Flag to control WSS and wall mask PNG generation
    zero_velocity_in_walls=True  # New flag to zero out velocity in wall cells
) -> AneurysmSimulation2D:
    """Setup aneurysm simulation with configurable parameters"""
    
    # Convert mm to meters
    mm_to_m = 0.001
    vessel_length_m = vessel_length_mm * mm_to_m
    vessel_diameter_m = vessel_diameter_mm * mm_to_m
    resolution_m = resolution_mm * mm_to_m
    
    # Calculate base grid dimensions (without bulge)
    grid_x = int(round(vessel_length_m / resolution_m))
    grid_y = int(round(vessel_diameter_m / resolution_m))
    
    # Calculate bulge dimensions in lattice units
    bulge_horizontal_lu = int(round(bulge_horizontal_mm * mm_to_m / resolution_m))
    bulge_vertical_lu = int(round((bulge_vertical_mm / 2) * mm_to_m / resolution_m))
    
    # Final grid shape including bulge
    grid_shape = (grid_x + 1, grid_y + bulge_vertical_lu + 5) # Add 1 for boundary cells
    
    # Calculate vessel centerline
    vessel_centre_lu = grid_y // 2
    
    # Simulation parameters
    backend = ComputeBackend.WARP

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

    precision_policy = PrecisionPolicy.FP32FP32
    
    velocity_set = xlb.velocity_set.D2Q9(
        precision_policy=precision_policy,
        backend=backend
    )
    
    # Calculate kinematic viscosity from dynamic viscosity and density
    kinematic_viscosity = dynamic_viscosity / blood_density  # m²/s

    # Calculate relaxation parameter
    dx = resolution_m
    
    # Use consistent dt for all calculations
    nu_lbm = kinematic_viscosity * dt / (dx**2)
    omega = 1.0 / (3 * nu_lbm + 0.5)
    
    # Validate tau for stability
    tau = 1/omega
    assert 0.5 <= tau <= 2.0, f"Tau value {tau:.3f} out of stable range [0.5, 2.0]"
    
    # Create input parameters dictionary
    input_params = {
        "vessel_length_mm": vessel_length_mm,
        "vessel_diameter_mm": vessel_diameter_mm,
        "bulge_horizontal_mm": bulge_horizontal_mm,
        "bulge_vertical_mm": bulge_vertical_mm,
        "resolution_mm": resolution_mm,
        "kinematic_viscosity": kinematic_viscosity,
        "dt": dt,
        "dx": dx,
        "fps": fps,
        "vessel_length_lu": grid_x,
        "vessel_diameter_lu": grid_y,
        "bulge_horizontal_lu": bulge_horizontal_lu,
        "bulge_vertical_lu": bulge_vertical_lu,
        "vessel_centre_lu": vessel_centre_lu,
        "bulge_centre_x_lu": grid_x // 2,
        "bulge_centre_y_lu": vessel_centre_lu + (grid_y // 2),
        "flow_profile": flow_profile,
        "save_wss_png": save_wss_png,  # Add the flag to input parameters
        "zero_velocity_in_walls": zero_velocity_in_walls  # Add the flag to input parameters
    }
    
    # Create simulation
    simulation = AneurysmSimulation2D(
        omega=omega,
        grid_shape=grid_shape,
        velocity_set=velocity_set,
        backend=backend,
        precision_policy=precision_policy,
        resolution=resolution_mm,
        input_params=input_params,
        output_path=output_path
    )
    
    return simulation

def run_for_duration(simulation, duration_seconds, dt, post_process_interval=None, warmup_seconds=0.0):
    """Run simulation for a specific duration in seconds.
    
    This is a wrapper around the AneurysmSimulation2D.run_for_duration method
    for backward compatibility.
    
    Args:
        simulation: The simulation object
        duration_seconds: How long to run in physical time (seconds)
        dt: Time step size (seconds)
        post_process_interval: Steps between post-processing
        warmup_seconds: Initial period in seconds to run before starting post-processing
        
    Returns:
        The total number of steps executed
    """
    # Print deprecation warning
    print("DEPRECATION WARNING: This function is replaced by simulation.run_for_duration()")
    
    # Call the simulation's run_for_duration method
    return simulation.run_for_duration(
        duration_seconds=duration_seconds,
        warmup_seconds=warmup_seconds,
        post_process_interval=post_process_interval
    )


if __name__ == "__main__":
    # Add command line argument parsing
    parser = argparse.ArgumentParser(description='Aneurysm flow simulation with Non-Newtonian BGK and Time-Dependent Zou-He')
    parser.add_argument('--generate-pngs', action='store_true',
                        help='Generate PNG images during simulation (if not specified, only VTK files will be saved)')
    args = parser.parse_args()
    
    # Set simulation parameters
    dt = 1e-5  # Time step size (seconds)
    resolution_mm = 0.02  # Resolution in mm
    resolution_m = resolution_mm * 0.001  # Convert to meters
    vessel_diameter_mm = 6.5  # Male Common Carotid Artery size.
    vessel_length_mm = 15  # Vessel length in mm
    
    # Load CSV files - now with dx and dt parameters for lattice unit conversion
    flow_profile_data = load_csv_data(
        vessel_radius_mm=vessel_diameter_mm/2,  # Convert diameter to radius
        dx=resolution_m,                        # Pass grid spacing
        dt=dt,                                  # Pass time step
        resample_points=64,                     # Ensure exactly 64 points
        normalize_time=True                     # Normalize time to 1 second
    )

    # Auto-select CCA profile without prompting the user
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

    import warp as wp
    wp.clear_kernel_cache()     # Clear kernel cache to avoid conflicts with new kernel code

    # Create output directory with specific name
    base_dir = os.getcwd()
    output_path = os.path.join(base_dir, "../results/aneurysm_flow/CCA_simulation_results_nnbgk_tdzh")
    # Create parent directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Output will be saved to: {output_path}")

    # Delete the output directory if it exists
    delete_directory_if_exists(output_path)

    # Create simulation with realistic vessel parameters
    simulation = aneurysm_simulation_setup(
        vessel_length_mm=vessel_length_mm,         # 15mm vessel length
        vessel_diameter_mm=vessel_diameter_mm,        # vessel diameter (typical cerebral artery)
        bulge_horizontal_mm=12,       # 12mm horizontal bulge
        bulge_vertical_mm=8,         # 8mm vertical bulge
        resolution_mm=resolution_mm,          # resolution
        dynamic_viscosity=0.0035,    # Blood dynamic viscosity (Pa·s)
        blood_density=1056,          # Blood density (kg/m³)
        dt=dt,                       # Time step
        fps=100,                     # Output frames per second
        flow_profile=flow_profile,   # Pass selected flow profile with name
        output_path=output_path,     # Pass the output path
        save_wss_png=args.generate_pngs  # Pass the generate-pngs flag as save_wss_png
    )

    # Run simulation for 1 second with warmup
    simulation.run_for_duration(
        duration_seconds=1.0,
        warmup_seconds=2.0    # Run for 2.0 seconds before starting visualization
    )

    print("\nSimulation complete!")
    print(f"Results saved to: {simulation.output_dir}")
    
    # TODO:
    # look into the post_process method to see if it can be modified to save the results in a more useful format
    # this includes extraction of blood vessel features such as wall shear stress, velocity profiles, etc.
    # try to modify the speed to use a newtonian model, or better yet, a non-newtonian model
    # adjust the velocity to better reflect a fluid flow in a blood vessel (set 0.4)
    # look into the boundary conditions to see if they can be modified to better reflect the conditions in a blood vessel
    # expand to 3D timestep