from pipe_run import pipe_simulation_setup
from utils.load_csv import load_csv_data
from datetime import datetime
from pathlib import Path
import logging
import warp as wp
import os
import json
import shutil
import argparse
import time
import sys
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipe_batch_run.log"),
        logging.StreamHandler()
    ]
)

def clear_simulation_directories():
    """Clear all pipe simulation output directories before starting the batch run"""
    base_dir = Path("/home/abdua786/code/uni/3/dissertation/dissertation")
    
    # Define the directories to clear
    directories_to_clear = [
        base_dir / "pipe_tdzh_bgk_results",
        base_dir / "pipe_tdzh_nnbgk_results",
        base_dir / "pipe_zh_bgk_results",
        base_dir / "pipe_zh_nnbgk_results"
    ]
    
    for directory in directories_to_clear:
        if directory.exists():
            logging.info(f"Clearing directory: {directory}")
            try:
                shutil.rmtree(directory)
                logging.info(f"Successfully cleared {directory}")
            except Exception as e:
                logging.error(f"Error clearing directory {directory}: {str(e)}")
        else:
            logging.info(f"Directory {directory} does not exist, no need to clear")
    
    return True

def run_all_combinations(
    vessel_length_mm=15,
    vessel_diameter_mm=6.5,  # Common Carotid Artery
    resolution_mm=0.02,
    dynamic_viscosity=0.0035,
    blood_density=1056,
    dt=1e-5,
    fps=100,
    duration_seconds=1.0,
    warmup_seconds=0.5,
    profile_name="CCA"  # Common Carotid Artery
):
    """Run pipe simulations with all combinations of boundary conditions and collision operators"""
    
    # Clear all simulation directories before starting
    clear_simulation_directories()
    
    # Configurations to run
    boundary_conditions = [
        {"name": "Time_Dependent_Zou_He", "use_time_dependent": True},
        {"name": "Standard_Zou_He", "use_time_dependent": False}
    ]
    
    collision_operators = [
        {"name": "Standard_BGK", "use_non_newtonian": False},
        {"name": "Non_Newtonian_BGK", "use_non_newtonian": True}
    ]
    
    # Initialize results list to collect outcomes of all runs
    results = []
    
    # Create base results directory
    base_results_dir = os.path.join(str(Path("/home/abdua786/code/uni/3/dissertation/dissertation")), 
                                   f"pipe_batch_results")
    os.makedirs(base_results_dir, exist_ok=True)
    
    # Load CCA profile data
    resolution_m = resolution_mm * 0.001
    flow_profile_data = load_csv_data(
        vessel_radius_mm=vessel_diameter_mm/2,
        dx=resolution_m,
        dt=dt,
        resample_points=64,
        normalize_time=True
    )
    
    # Print available profiles for debugging
    logging.info(f"Available velocity profiles: {list(flow_profile_data.keys())}")
    
    # Find the CCA profile - look for both "CCA" and the default profile
    cca_profile = None
    # First, try to find a profile with "CCA" or "CAROTID" in the name
    for profile_key in flow_profile_data.keys():
        if "CCA" in profile_key.upper() or "CAROTID" in profile_key.upper():
            cca_profile = profile_key
            break
    
    # If no specific CCA profile found, use the first available profile
    if not cca_profile and flow_profile_data:
        cca_profile = list(flow_profile_data.keys())[0]
        logging.info(f"No specific CCA profile found. Using first available profile: {cca_profile}")
    
    if not cca_profile:
        logging.error("No velocity profiles found in the data")
        return
    
    # Use the CCA column from the profile if it exists, otherwise use the first column
    selected_data = flow_profile_data[cca_profile]
    x_col = selected_data['x']
    
    # Look for CCA column first, otherwise use first available column
    y_col_names = list(selected_data['y'].keys())
    cca_velocity_col = None
    
    # Try to find a column with "CCA" in the name
    for col_name in y_col_names:
        if "CCA" in col_name.upper():
            cca_velocity_col = col_name
            break
    
    # If no CCA column found, use the first column
    if not cca_velocity_col and y_col_names:
        cca_velocity_col = y_col_names[0]
    
    if not cca_velocity_col:
        logging.error(f"No velocity columns found in profile {cca_profile}")
        return
    
    y_col = selected_data['y'][cca_velocity_col]
    
    # Create the flow profile object
    profile_name_with_col = f"{cca_profile}_{cca_velocity_col}"
    units = selected_data.get('units', 'unknown')
    logging.info(f"Using flow profile: {profile_name_with_col} (units: {units})")
    
    flow_profile = {
        'name': profile_name_with_col,
        'data': {'x': x_col, 'y': y_col}
    }
    
    # Calculate kinematic viscosity from dynamic viscosity and density
    kinematic_viscosity = dynamic_viscosity / blood_density
    
    total_runs = len(boundary_conditions) * len(collision_operators)
    current_run = 0

    for co in collision_operators:
        for bc in boundary_conditions:
        
            current_run += 1
            run_name = f"{bc['name']}_{co['name']}"
            
            logging.info(f"Starting run {current_run}/{total_runs}: {run_name}")
            logging.info(f"  - Boundary condition: {bc['name']}")
            logging.info(f"  - Collision operator: {co['name']}")
            
            # Force a complete reset of Warp's state before each simulation
            wp.clear_kernel_cache()
            
            # Sleep briefly to ensure resources are fully released
            time.sleep(1)
            
            # Create simulation with a fresh initialization
            try:
                logging.info(f"Setting up simulation: {run_name}")
                
                simulation = pipe_simulation_setup(
                    vessel_length_mm=vessel_length_mm,
                    vessel_diameter_mm=vessel_diameter_mm,                                                    
                    resolution=resolution_mm,
                    kinematic_viscosity=kinematic_viscosity,
                    dt=dt,
                    fps=fps,
                    flow_profile=flow_profile,
                    use_time_dependent_zou_he=bc["use_time_dependent"],
                    use_non_newtonian_bgk=co["use_non_newtonian"]
                )
                
                # Run simulation
                logging.info(f"Running simulation: {run_name}")
                start_time = time.time()
                
                simulation.run_for_duration(
                    duration_seconds=duration_seconds,
                    warmup_seconds=warmup_seconds
                )
                
                end_time = time.time()
                
                # Explicitly synchronize Warp device before cleanup
                logging.info(f"Synchronizing Warp device for {run_name}...")
                try:
                    wp.synchronize()
                    logging.info("Warp device synchronized.")
                except Exception as e:
                    logging.error(f"Error during Warp synchronization: {e}")


                # Force cleanup of simulation resources
                logging.info(f"Cleaning up simulation resources: {run_name}")
                output_dir = str(simulation.output_dir)

                # Remove references to simulation object
                del simulation

                # Explicitly trigger Python garbage collection
                gc.collect()

                # Add an increased sleep here as well
                time.sleep(5) # Increased sleep for testing
                
                # Record results
                logging.info(f"Recording results for: {run_name}")
                results.append({
                    "run_name": run_name,
                    "boundary_condition": bc["name"],
                    "collision_operator": co["name"],
                    "status": "success",
                    "runtime_seconds": end_time - start_time,
                    "output_directory": output_dir
                })
                
                logging.info(f"Successfully completed run: {run_name}")
                logging.info(f"Runtime: {end_time - start_time:.2f} seconds")
                
            except Exception as e:
                logging.error(f"Error in run {run_name}: {str(e)}")
                import traceback
                logging.error(traceback.format_exc())
                
                # Record failure
                results.append({
                    "run_name": run_name,
                    "boundary_condition": bc["name"],
                    "collision_operator": co["name"],
                    "status": "failed",
                    "error": str(e)
                })
            

    # Save results summary
    with open(os.path.join(base_results_dir, "results_summary.json"), "w") as f:
        json.dump(results, f, indent=4)
    
    logging.info(f"All simulation runs completed. Results directory: {base_results_dir}")
    return base_results_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multiple pipe simulations with all parameter combinations")
    parser.add_argument("--length", type=float, default=15, help="Vessel length in mm")
    parser.add_argument("--diameter", type=float, default=6.5, help="Vessel diameter in mm")
    parser.add_argument("--resolution", type=float, default=0.02, help="Spatial resolution in mm")
    parser.add_argument("--dt", type=float, default=1e-5, help="Time step in seconds")
    parser.add_argument("--duration", type=float, default=1.0, help="Simulation duration in seconds")
    parser.add_argument("--warmup", type=float, default=0, help="Warmup time in seconds")
    
    args = parser.parse_args()
    
    logging.info("Starting batch run of pipe simulations")
    logging.info(f"Configuration: Length={args.length}mm, Diameter={args.diameter}mm, Resolution={args.resolution}mm")
    
    result_dir = run_all_combinations(
        vessel_length_mm=args.length,
        vessel_diameter_mm=args.diameter,
        resolution_mm=args.resolution,
        dt=args.dt,
        duration_seconds=args.duration,
        warmup_seconds=args.warmup
    )
    
    logging.info(f"Batch run completed. Results in: {result_dir}")
