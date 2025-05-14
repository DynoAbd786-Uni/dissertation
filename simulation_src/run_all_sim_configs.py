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

def run_pipe_simulation(boundary_condition, collision_operator, dry_run=False, long_pipe=False):
    """
    Run a pipe simulation with the specified boundary condition and collision operator.
    
    Args:
        boundary_condition (str): Boundary condition type, either "standard" or "time-dependent"
        collision_operator (str): Collision operator type, either "standard" or "non-newtonian"
        dry_run (bool): If True, print command but don't execute
        long_pipe (bool): If True, use longer pipe simulation settings
    
    Returns:
        tuple: (return_code, output, error) from the process
    """
    # Get current timestamp for logs
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Create a descriptive name for this run
    run_name = f"pipe_{boundary_condition}_{collision_operator}_{timestamp}"
    
    # Prepare log directory
    log_dir = BASE_DIR / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{run_name}.log"
    
    # Build the command
    cmd = [
        sys.executable,  # Use the current Python interpreter
        str(SCRIPT_DIR / "pipe_run.py"),
        f"--boundary-condition={boundary_condition}",
        f"--collision-operator={collision_operator}"
    ]
    
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

def run_aneurysm_simulation(dry_run=False):
    """
    Run an aneurysm simulation.
    Aneurysm simulations always use time-dependent boundary conditions and non-newtonian BGK.
    
    Args:
        dry_run (bool): If True, print command but don't execute
    
    Returns:
        tuple: (return_code, output, error) from the process
    """
    # Get current timestamp for logs
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Create a descriptive name for this run
    run_name = f"aneurysm_{timestamp}"
    
    # Prepare log directory
    log_dir = BASE_DIR / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{run_name}.log"
    
    # Build the command
    cmd = [
        sys.executable,  # Use the current Python interpreter
        str(SCRIPT_DIR / "aneurysm_run.py")
    ]
    
    # Add the generate-pngs flag only in production runs (not dry runs)
    if not dry_run:
        cmd.append("--generate-pngs")
    
    print(f"\n{'='*80}")
    print(f"Running aneurysm simulation with:")
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

def main():
    """
    Main function to parse arguments and run all simulation combinations.
    """
    parser = argparse.ArgumentParser(description="Run all possible simulation configurations")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--aneurysm-only", action="store_true", help="Run only the aneurysm simulations")
    parser.add_argument("--pipe-only", action="store_true", help="Run only the pipe simulations")
    parser.add_argument("--long-pipe", action="store_true", 
                        help="Use long pipe simulation settings (dt=5e-5, resolution=0.08, vessel_length=800mm)")
    args = parser.parse_args()
    
    # Track successes and failures
    pipe_results = []
    aneurysm_results = []
    
    start_time_all = time.time()
    
    # Run pipe simulations if requested
    if not args.aneurysm_only:
        # Run all pipe combinations
        print(f"Running all {len(BOUNDARY_CONDITIONS) * len(COLLISION_OPERATORS)} pipe combinations")
        
        for bc in BOUNDARY_CONDITIONS:
            for co in COLLISION_OPERATORS:
                return_code, _, _ = run_pipe_simulation(bc, co, args.dry_run, args.long_pipe)
                pipe_results.append((bc, co, return_code == 0))
    
    # Run aneurysm simulations if requested
    if not args.pipe_only:
        print("Running aneurysm simulation")
        # Aneurysm with NNBGK and TDZH (defaults)
        return_code, _, _ = run_aneurysm_simulation(args.dry_run)
        aneurysm_results.append(("nnbgk_tdzh", return_code == 0))
    
    end_time_all = time.time()
    total_time = end_time_all - start_time_all
    
    # Print summary
    print("\n" + "="*80)
    print("SIMULATION SUMMARY")
    print("="*80)
    print(f"Total time: {total_time:.2f} seconds")
    
    # Pipe results
    if pipe_results:
        print("\nPipe Simulations:")
        print(f"Combinations run: {len(pipe_results)}")
        
        pipe_successes = sum(1 for _, _, success in pipe_results if success)
        print(f"Successful: {pipe_successes}/{len(pipe_results)}")
        
        print("\nResults by combination:")
        for bc, co, success in pipe_results:
            status = "SUCCESS" if success else "FAILED"
            print(f"  Pipe: {bc:<15} + {co:<15} = {status}")
    
    # Aneurysm results
    if aneurysm_results:
        print("\nAneurysm Simulations:")
        print(f"Combinations run: {len(aneurysm_results)}")
        
        aneurysm_successes = sum(1 for _, success in aneurysm_results if success)
        print(f"Successful: {aneurysm_successes}/{len(aneurysm_results)}")
        
        print("\nResults by combination:")
        for co, success in aneurysm_results:
            status = "SUCCESS" if success else "FAILED"
            print(f"  Aneurysm: time-dependent + {co:<15} = {status}")

if __name__ == "__main__":
    main()