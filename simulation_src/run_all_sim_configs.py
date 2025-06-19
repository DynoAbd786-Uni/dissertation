#!/usr/bin/env python3
"""
Script to run all combinations of boundary conditions and collision operators for pipe simulations
and aneurysm simulations. This script will execute multiple simulations with different configurations.
"""

import os
import subprocess
import time
import argparse
from datetime import datetime
import sys
from pathlib import Path

# Define base directory - use the directory of this script
SCRIPT_DIR = Path(__file__).parent.absolute()
BASE_DIR = SCRIPT_DIR.parent  # Dissertation directory

# Define combinations
BOUNDARY_CONDITIONS = ["standard", "time-dependent"]
COLLISION_OPERATORS = ["standard", "non-newtonian"]
SPATIAL_PROFILES = ["uniform", "poiseuille", "blunted_paraboloid"]

# Default spatial profile parameters
SPATIAL_PROFILE_DEFAULTS = {
    "uniform": {},  # No additional parameters needed for uniform profile
    "poiseuille": {},
    "blunted_paraboloid": {
        "power_law_exponent": 1.7  # Default for blood flow
    }
}

def run_pipe_simulation(boundary_condition, collision_operator, spatial_profile="poiseuille", dry_run=False, long_pipe=False):
    """
    Run a pipe simulation with the specified boundary condition, collision operator, and spatial profile.
    
    Args:
        boundary_condition (str): Boundary condition type, either "standard" or "time-dependent"
        collision_operator (str): Collision operator type, either "standard" or "non-newtonian"
        spatial_profile (str): Spatial profile type, either "poiseuille" or "blunted_paraboloid"
        dry_run (bool): If True, print command but don't execute
        long_pipe (bool): If True, use longer pipe simulation settings
    
    Returns:
        tuple: (return_code, output, error) from the process
    """
    # Get current timestamp for logs
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Create a descriptive name for this run including spatial profile
    # Use short names for directory structure
    bc_short = "tdzh" if boundary_condition == "time-dependent" else "zh"
    co_short = "nnbgk" if collision_operator == "non-newtonian" else "bgk"
    sp_short = "uniform" if spatial_profile == "uniform" else ("blunted" if spatial_profile == "blunted_paraboloid" else "poiseuille")
    pipe_type = "long" if long_pipe else "standard"
    
    run_name = f"pipe_{bc_short}_{co_short}_{sp_short}_{pipe_type}_{timestamp}"
    
    # Prepare log directory
    log_dir = BASE_DIR / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{run_name}.log"
    
    # Build the command
    cmd = [
        sys.executable,  # Use the current Python interpreter
        str(SCRIPT_DIR / "pipe_run.py"),
        f"--boundary-condition={boundary_condition}",
        f"--collision-operator={collision_operator}",
        f"--spatial-profile={spatial_profile}"
    ]
    
    # Add spatial profile specific parameters
    if spatial_profile in SPATIAL_PROFILE_DEFAULTS:
        for param, value in SPATIAL_PROFILE_DEFAULTS[spatial_profile].items():
            cmd.append(f"--{param.replace('_', '-')}={value}")
    
    # Add long pipe flag if requested
    if long_pipe:
        cmd.append("--long-pipe")
    
    # Add generate-pngs flag for production runs (not dry runs)
    if not dry_run:
        cmd.append("--generate-pngs")
    
    print(f"\n{'='*80}")
    print(f"Running pipe simulation with:")
    print(f"  Boundary Condition: {boundary_condition}")
    print(f"  Collision Operator: {collision_operator}")
    print(f"  Spatial Profile: {spatial_profile}")
    print(f"  Long Pipe: {'Enabled' if long_pipe else 'Disabled'}")
    print(f"  Log File: {log_file}")
    print(f"  Command: {' '.join(cmd)}")
    
    if dry_run:
        print("DRY RUN - Command not executed")
        return 0, "Dry run - no output", ""
    
    # Run the simulation
    print(f"Starting pipe simulation at {timestamp}")
    start_time = time.time()
    
    try:
        # Open log file for writing
        with open(log_file, 'w') as log:
            # Run the process and capture output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1  # Line buffered
            )
            
            # Process output in real-time
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())
                    log.write(output)
                    log.flush()
            
            # Get return code and any remaining output
            return_code = process.poll()
            stdout, stderr = process.communicate()
            
            # Write any remaining output and errors to log
            if stdout:
                log.write(stdout)
            if stderr:
                log.write("ERRORS:\n")
                log.write(stderr)
                print(f"ERRORS: {stderr}")
        
        # Calculate run time
        end_time = time.time()
        run_time = end_time - start_time
        
        print(f"Finished pipe simulation in {run_time:.2f} seconds")
        print(f"Return code: {return_code}")
        
        return return_code, stdout, stderr
    
    except Exception as e:
        print(f"Error running pipe simulation: {e}")
        return 1, "", str(e)

def get_all_pipe_combinations():
    """
    Get all combinations of boundary conditions, collision operators, and spatial profiles.
    
    Returns:
        list: List of tuples (boundary_condition, collision_operator, spatial_profile)
    """
    combinations = []
    for bc in BOUNDARY_CONDITIONS:
        for co in COLLISION_OPERATORS:
            for sp in SPATIAL_PROFILES:
                combinations.append((bc, co, sp))
    return combinations

def run_aneurysm_simulation(spatial_profile="blunted_paraboloid", dry_run=False):
    """
    Run an aneurysm simulation with specified spatial profile.
    Aneurysm simulations always use time-dependent boundary conditions and non-newtonian BGK.
    
    Args:
        spatial_profile (str): Spatial profile type ("poiseuille" or "blunted_paraboloid")
        dry_run (bool): If True, print command but don't execute
    
    Returns:
        tuple: (return_code, output, error) from the process
    """
    # Get current timestamp for logs
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Create a descriptive name for this run including spatial profile
    sp_short = "uniform" if spatial_profile == "uniform" else ("blunted" if spatial_profile == "blunted_paraboloid" else "poiseuille")
    run_name = f"aneurysm_tdzh_nnbgk_{sp_short}_{timestamp}"
    
    # Prepare log directory
    log_dir = BASE_DIR / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{run_name}.log"
    
    # Build the command
    cmd = [
        sys.executable,  # Use the current Python interpreter
        str(SCRIPT_DIR / "aneurysm_run.py"),
        f"--spatial-profile={spatial_profile}"
    ]
    
    # Add spatial profile specific parameters
    if spatial_profile in SPATIAL_PROFILE_DEFAULTS:
        for param, value in SPATIAL_PROFILE_DEFAULTS[spatial_profile].items():
            cmd.append(f"--{param.replace('_', '-')}={value}")
    
    # Add the generate-pngs flag only in production runs (not dry runs)
    if not dry_run:
        cmd.append("--generate-pngs")
    
    print(f"\n{'='*80}")
    print(f"Running aneurysm simulation with:")
    print(f"  Boundary Condition: time-dependent")
    print(f"  Collision Operator: non-newtonian")
    print(f"  Spatial Profile: {spatial_profile}")
    print(f"  Log File: {log_file}")
    print(f"  Command: {' '.join(cmd)}")
    
    if dry_run:
        print("DRY RUN - Command not executed")
        return 0, "Dry run - no output", ""
    
    # Run the simulation
    print(f"Starting aneurysm simulation at {timestamp}")
    start_time = time.time()
    
    try:
        # Open log file for writing
        with open(log_file, 'w') as log:
            # Run the process and capture output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1  # Line buffered
            )
            
            # Process output in real-time
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())
                    log.write(output)
                    log.flush()
            
            # Get return code and any remaining output
            return_code = process.poll()
            stdout, stderr = process.communicate()
            
            # Write any remaining output and errors to log
            if stdout:
                log.write(stdout)
            if stderr:
                log.write("ERRORS:\n")
                log.write(stderr)
                print(f"ERRORS: {stderr}")
        
        # Calculate run time
        end_time = time.time()
        run_time = end_time - start_time
        
        print(f"Finished aneurysm simulation in {run_time:.2f} seconds")
        print(f"Return code: {return_code}")
        
        return return_code, stdout, stderr
    
    except Exception as e:
        print(f"Error running aneurysm simulation: {e}")
        return 1, "", str(e)

def run_vtk_processing(dry_run=False):
    """
    Run VTK processing to convert simulation results to .npz format for analysis.
    
    Args:
        dry_run (bool): If True, print command but don't execute
    
    Returns:
        tuple: (return_code, output, error) from the process
    """
    # Get current timestamp for logs
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Prepare log directory
    log_dir = BASE_DIR / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"vtk_processing_{timestamp}.log"
    
    # VTK processing script path
    vtk_script_path = BASE_DIR / "visualisation_src" / "vtk_processing.py"
    
    # Build the command
    cmd = [
        sys.executable,  # Use the current Python interpreter
        str(vtk_script_path)
    ]
    
    print(f"\n{'='*80}")
    print(f"Running VTK processing:")
    print(f"  Script: {vtk_script_path}")
    print(f"  Log File: {log_file}")
    print(f"  Command: {' '.join(cmd)}")
    
    if dry_run:
        print("DRY RUN - VTK processing command not executed")
        return 0, "Dry run - no output", ""
    
    # Check if VTK processing script exists
    if not vtk_script_path.exists():
        print(f"ERROR: VTK processing script not found at {vtk_script_path}")
        return 1, "", f"VTK processing script not found at {vtk_script_path}"
    
    # Run the VTK processing
    print(f"Starting VTK processing at {timestamp}")
    start_time = time.time()
    
    try:
        # Open log file for writing
        with open(log_file, 'w') as log:
            # Run the process and capture output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
                cwd=str(BASE_DIR / "visualisation_src")  # Run from visualisation_src directory
            )
            
            # Process output in real-time
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())
                    log.write(output)
                    log.flush()
            
            # Get return code and any remaining output
            return_code = process.poll()
            stdout, stderr = process.communicate()
            
            # Write any remaining output and errors to log
            if stdout:
                log.write(stdout)
            if stderr:
                log.write("ERRORS:\n")
                log.write(stderr)
                print(f"ERRORS: {stderr}")
        
        # Calculate run time
        end_time = time.time()
        run_time = end_time - start_time
        
        print(f"Finished VTK processing in {run_time:.2f} seconds")
        print(f"Return code: {return_code}")
        
        return return_code, stdout, stderr
    
    except Exception as e:
        print(f"Error running VTK processing: {e}")
        return 1, "", str(e)

def handle_simulation_failure(sim_type, config_details, return_code, stderr, pipe_results, aneurysm_results):
    """
    Handle simulation failure by printing detailed error report and exiting.
    
    Args:
        sim_type (str): Type of simulation ("pipe" or "aneurysm")
        config_details (str): Configuration details for the failed simulation
        return_code (int): Return code from the failed simulation
        stderr (str): Error output from the failed simulation
        pipe_results (list): List of pipe simulation results so far
        aneurysm_results (list): List of aneurysm simulation results so far
    """
    print(f"\n{'='*80}")
    print("❌ SIMULATION FAILED - STOPPING EXECUTION")
    print(f"{'='*80}")
    print(f"Failed simulation: {sim_type}")
    print(f"Configuration: {config_details}")
    print(f"Return code: {return_code}")
    print(f"Error output: {stderr}")
    
    # Print summary of what was completed before failure
    total_planned = len(pipe_results) + len(aneurysm_results)
    completed_successfully = 0
    
    if pipe_results:
        pipe_completed = sum(1 for _, _, _, success in pipe_results if success)
        completed_successfully += pipe_completed
        print(f"\nPipe simulations completed before failure: {pipe_completed}")
        for bc, co, sp, success in pipe_results:
            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"  {bc} + {co} + {sp}: {status}")
    
    if aneurysm_results:
        aneurysm_completed = sum(1 for _, success in aneurysm_results if success)
        completed_successfully += aneurysm_completed
        print(f"\nAneurysm simulations completed before failure: {aneurysm_completed}")
        for config, success in aneurysm_results:
            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"  {config}: {status}")
    
    print(f"\n{'='*80}")
    print("FAILURE SUMMARY")
    print(f"{'='*80}")
    print(f"Simulations completed successfully: {completed_successfully}")
    print(f"Failed simulation: {config_details}")
    print("VTK processing: SKIPPED (due to failure)")
    print(f"\nTo resume, fix the issue and re-run the failed configuration:")
    if sim_type == "pipe":
        print(f"  python pipe_run.py [appropriate flags for {config_details}]")
    else:
        print(f"  python aneurysm_run.py [appropriate flags for {config_details}]")
    print(f"{'='*80}")
    
    # Exit with failure code
    sys.exit(1)

def main():
    """
    Main function to parse arguments and run all simulation combinations.
    """
    parser = argparse.ArgumentParser(description="Run all possible simulation configurations")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--aneurysm-only", action="store_true", help="Run only the aneurysm simulations")
    parser.add_argument("--pipe-only", action="store_true", help="Run only the pipe simulations")
    parser.add_argument("--long-pipe", action="store_true", 
                        help="Run ONLY long pipe simulation settings (dt=4e-5, resolution=0.08, vessel_length=800mm). "
                        "By default, BOTH standard and long pipe configurations are run.")
    parser.add_argument("--standard-pipe-only", action="store_true",
                        help="Run ONLY standard pipe simulation settings (dt=1e-5, resolution=0.02, vessel_length=15mm). "
                        "By default, BOTH standard and long pipe configurations are run.")
    parser.add_argument("--spatial-profile", choices=SPATIAL_PROFILES + ["all"], default="all",
                        help="Spatial profile to use for simulations. 'all' runs all profiles (default: all)")
    parser.add_argument("--boundary-condition", choices=BOUNDARY_CONDITIONS + ["all"], default="all",
                        help="Boundary condition to use for pipe simulations. 'all' runs all conditions (default: all)")
    parser.add_argument("--collision-operator", choices=COLLISION_OPERATORS + ["all"], default="all",
                        help="Collision operator to use for pipe simulations. 'all' runs all operators (default: all)")
    parser.add_argument("--aneurysm-spatial-profile", choices=SPATIAL_PROFILES + ["all"], default="blunted_paraboloid",
                        help="Spatial profile to use for aneurysm simulations. 'blunted_paraboloid' is default for blood flow (default: blunted_paraboloid)")
    args = parser.parse_args()
    
    # Track successes and failures
    pipe_results = []
    aneurysm_results = []
    
    start_time_all = time.time()
    
    # Run pipe simulations if requested
    if not args.aneurysm_only:
        # Check for conflicting flags
        if args.long_pipe and args.standard_pipe_only:
            print("ERROR: Cannot specify both --long-pipe and --standard-pipe-only flags.")
            print("Use one or the other, or neither for both configurations.")
            return
        
        # Modified logic: Run both standard AND long pipe configurations by default
        if args.long_pipe:
            # If --long-pipe is explicitly specified, only run long pipe configurations
            long_pipe_combinations = [
                ("standard", "non-newtonian", "uniform"),
                ("standard", "non-newtonian", "poiseuille"), 
                ("standard", "non-newtonian", "blunted_paraboloid")
            ]
            
            print(f"Running {len(long_pipe_combinations)} long pipe combinations (--long-pipe specified):")
            for bc, co, sp in long_pipe_combinations:
                print(f"  - {bc} + {co} + {sp} (LONG PIPE)")
            
            # Execute the long pipe combinations
            for bc, co, sp in long_pipe_combinations:
                return_code, _, stderr = run_pipe_simulation(bc, co, sp, args.dry_run, True)
                pipe_results.append((bc, co, sp, return_code == 0))
                # Check for failure
                if return_code != 0:
                    handle_simulation_failure("pipe", f"{bc} + {co} + {sp}", return_code, stderr, pipe_results, aneurysm_results)
        elif args.standard_pipe_only:
            # If --standard-pipe-only is specified, only run standard pipe configurations
            if args.spatial_profile == "all" and args.boundary_condition == "all" and args.collision_operator == "all":
                # Run all standard combinations
                combinations = get_all_pipe_combinations()
                print(f"Running all {len(combinations)} standard pipe combinations (--standard-pipe-only specified):")
                for bc, co, sp in combinations:
                    print(f"  - {bc} + {co} + {sp} (STANDARD)")
            else:
                # Run specific standard combinations based on arguments
                boundary_conditions = BOUNDARY_CONDITIONS if args.boundary_condition == "all" else [args.boundary_condition]
                collision_operators = COLLISION_OPERATORS if args.collision_operator == "all" else [args.collision_operator]
                spatial_profiles = SPATIAL_PROFILES if args.spatial_profile == "all" else [args.spatial_profile]
                
                combinations = []
                for bc in boundary_conditions:
                    for co in collision_operators:
                        for sp in spatial_profiles:
                            combinations.append((bc, co, sp))
                
                print(f"Running {len(combinations)} specific standard pipe combinations (--standard-pipe-only specified):")
                for bc, co, sp in combinations:
                    print(f"  - {bc} + {co} + {sp} (STANDARD)")
            
            # Execute the standard combinations
            for bc, co, sp in combinations:
                return_code, _, stderr = run_pipe_simulation(bc, co, sp, args.dry_run, False)
                pipe_results.append((bc, co, sp, return_code == 0))
                # Check for failure
                if return_code != 0:
                    handle_simulation_failure("pipe", f"{bc} + {co} + {sp}", return_code, stderr, pipe_results, aneurysm_results)
        else:
            # Default behavior: Run BOTH standard and long pipe configurations
            print("Running comprehensive pipe simulation suite (both standard and long pipe configurations)")
            
            # First, run all standard pipe combinations
            if args.spatial_profile == "all" and args.boundary_condition == "all" and args.collision_operator == "all":
                # Run all standard combinations
                combinations = get_all_pipe_combinations()
                print(f"\n1. Running all {len(combinations)} standard pipe combinations:")
                for bc, co, sp in combinations:
                    print(f"  - {bc} + {co} + {sp} (STANDARD)")
            else:
                # Run specific standard combinations based on arguments
                boundary_conditions = BOUNDARY_CONDITIONS if args.boundary_condition == "all" else [args.boundary_condition]
                collision_operators = COLLISION_OPERATORS if args.collision_operator == "all" else [args.collision_operator]
                spatial_profiles = SPATIAL_PROFILES if args.spatial_profile == "all" else [args.spatial_profile]
                
                combinations = []
                for bc in boundary_conditions:
                    for co in collision_operators:
                        for sp in spatial_profiles:
                            combinations.append((bc, co, sp))
                
                print(f"\n1. Running {len(combinations)} specific standard pipe combinations:")
                for bc, co, sp in combinations:
                    print(f"  - {bc} + {co} + {sp} (STANDARD)")
            
            # Execute the standard combinations
            for bc, co, sp in combinations:
                return_code, _, stderr = run_pipe_simulation(bc, co, sp, args.dry_run, False)
                pipe_results.append((bc, co, sp, return_code == 0))
                # Check for failure
                if return_code != 0:
                    handle_simulation_failure("pipe", f"{bc} + {co} + {sp}", return_code, stderr, pipe_results, aneurysm_results)
            
            # Second, run long pipe combinations (standard ZH + non-newtonian BGK + all spatial profiles)
            long_pipe_combinations = [
                ("standard", "non-newtonian", "uniform"),
                ("standard", "non-newtonian", "poiseuille"), 
                ("standard", "non-newtonian", "blunted_paraboloid")
            ]
            
            print(f"\n2. Running {len(long_pipe_combinations)} long pipe combinations:")
            for bc, co, sp in long_pipe_combinations:
                print(f"  - {bc} + {co} + {sp} (LONG PIPE)")
            
            # Execute the long pipe combinations
            for bc, co, sp in long_pipe_combinations:
                return_code, _, stderr = run_pipe_simulation(bc, co, sp, args.dry_run, True)
                pipe_results.append((bc, co, sp, return_code == 0))
                # Check for failure
                if return_code != 0:
                    handle_simulation_failure("pipe", f"{bc} + {co} + {sp}", return_code, stderr, pipe_results, aneurysm_results)
    
    # Run aneurysm simulations if requested
    if not args.pipe_only:
        # Determine which aneurysm spatial profiles to run
        aneurysm_spatial_profiles = SPATIAL_PROFILES if args.aneurysm_spatial_profile == "all" else [args.aneurysm_spatial_profile]
        
        print(f"Running aneurysm simulations with {len(aneurysm_spatial_profiles)} spatial profile(s)")
        for aneurysm_sp in aneurysm_spatial_profiles:
            print(f"  - Aneurysm: time-dependent + non-newtonian + {aneurysm_sp}")
            return_code, _, stderr = run_aneurysm_simulation(aneurysm_sp, args.dry_run)
            aneurysm_results.append((f"nnbgk_tdzh_{aneurysm_sp}", return_code == 0))
            # Check for failure
            if return_code != 0:
                handle_simulation_failure("aneurysm", f"time-dependent + non-newtonian + {aneurysm_sp}", return_code, stderr, pipe_results, aneurysm_results)
    
    end_time_all = time.time()
    total_time = end_time_all - start_time_all
    
    # Check if all simulations were successful
    all_pipe_successful = all(success for _, _, _, success in pipe_results) if pipe_results else True
    all_aneurysm_successful = all(success for _, success in aneurysm_results) if aneurysm_results else True
    all_simulations_successful = all_pipe_successful and all_aneurysm_successful
    
    # Print summary
    print("\n" + "="*80)
    print("SIMULATION SUMMARY")
    print("="*80)
    print(f"Total time: {total_time:.2f} seconds")
    
    # Pipe results
    if pipe_results:
        print("\nPipe Simulations:")
        print(f"Combinations run: {len(pipe_results)}")
        
        pipe_successes = sum(1 for _, _, _, success in pipe_results if success)
        print(f"Successful: {pipe_successes}/{len(pipe_results)}")
        
        print("\nResults by combination:")
        for bc, co, sp, success in pipe_results:
            status = "SUCCESS" if success else "FAILED"
            print(f"  Pipe: {bc:<15} + {co:<15} + {sp:<20} = {status}")
    
    # Aneurysm results
    if aneurysm_results:
        print("\nAneurysm Simulations:")
        print(f"Combinations run: {len(aneurysm_results)}")
        
        aneurysm_successes = sum(1 for _, success in aneurysm_results if success)
        print(f"Successful: {aneurysm_successes}/{len(aneurysm_results)}")
        
        print("\nResults by combination:")
        for config, success in aneurysm_results:
            status = "SUCCESS" if success else "FAILED"
            print(f"  Aneurysm: {config:<35} = {status}")
    
    # Run VTK processing if all simulations were successful
    if all_simulations_successful and (pipe_results or aneurysm_results):
        print(f"\n{'='*80}")
        print("ALL SIMULATIONS SUCCESSFUL - RUNNING VTK PROCESSING")
        print(f"{'='*80}")
        
        vtk_return_code, _, _ = run_vtk_processing(args.dry_run)
        
        if vtk_return_code == 0:
            print(f"\n{'='*80}")
            print("✅ VTK PROCESSING COMPLETED SUCCESSFULLY")
            print("✅ All simulation data has been processed and is ready for analysis")
            print(f"{'='*80}")
        else:
            print(f"\n{'='*80}")
            print("❌ VTK PROCESSING FAILED")
            print("❌ Simulation data may not be fully processed for analysis")
            print(f"{'='*80}")
    elif not all_simulations_successful:
        print(f"\n{'='*80}")
        print("⚠️  SOME SIMULATIONS FAILED - SKIPPING VTK PROCESSING")
        print("⚠️  Fix failed simulations and re-run to enable VTK processing")
        print(f"{'='*80}")
        
        # Show which simulations failed
        failed_pipe = [(bc, co, sp) for bc, co, sp, success in pipe_results if not success]
        failed_aneurysm = [config for config, success in aneurysm_results if not success]
        
        if failed_pipe:
            print("\nFailed pipe simulations:")
            for bc, co, sp in failed_pipe:
                print(f"  - {bc} + {co} + {sp}")
        
        if failed_aneurysm:
            print("\nFailed aneurysm simulations:")
            for config in failed_aneurysm:
                print(f"  - {config}")
    else:
        print(f"\n{'='*80}")
        print("ℹ️  NO SIMULATIONS RUN - SKIPPING VTK PROCESSING")
        print(f"{'='*80}")

if __name__ == "__main__":
    main()