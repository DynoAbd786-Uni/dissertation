#!/usr/bin/env python3
"""
Script to run all combinations of boundary conditions and collision operators for pipe simulations.
This script will execute multiple pipe simulations with different configurations.
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

def run_simulation(boundary_condition, collision_operator, dry_run=False):
    """
    Run a simulation with the specified boundary condition and collision operator.
    
    Args:
        boundary_condition (str): Boundary condition type, either "standard" or "time-dependent"
        collision_operator (str): Collision operator type, either "standard" or "non-newtonian"
        dry_run (bool): If True, print command but don't execute
    
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
    
    print(f"\n{'='*80}")
    print(f"Running simulation with:")
    print(f"  Boundary Condition: {boundary_condition}")
    print(f"  Collision Operator: {collision_operator}")
    print(f"  Log File: {log_file}")
    print(f"  Command: {' '.join(cmd)}")
    
    if dry_run:
        print("DRY RUN - Command not executed")
        return 0, "Dry run - no output", ""
    
    # Run the simulation
    print(f"Starting simulation at {timestamp}")
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
        
        print(f"Finished simulation in {run_time:.2f} seconds")
        print(f"Return code: {return_code}")
        
        return return_code, stdout, stderr
    
    except Exception as e:
        print(f"Error running simulation: {e}")
        return 1, "", str(e)

def main():
    """
    Main function to parse arguments and run all simulation combinations.
    """
    parser = argparse.ArgumentParser(description="Run all combinations of pipe simulation configurations.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands but don't execute them")
    parser.add_argument("--single", action="store_true", help="Run only the standard-standard combination")
    args = parser.parse_args()
    
    # Track successes and failures
    results = []
    
    start_time_all = time.time()
    
    if args.single:
        # Run only standard-standard combination
        print("Running only standard boundary condition with standard collision operator")
        return_code, _, _ = run_simulation("standard", "standard", args.dry_run)
        results.append(("standard", "standard", return_code == 0))
    else:
        # Run all combinations
        print(f"Running all {len(BOUNDARY_CONDITIONS) * len(COLLISION_OPERATORS)} combinations")
        
        for bc in BOUNDARY_CONDITIONS:
            for co in COLLISION_OPERATORS:
                return_code, _, _ = run_simulation(bc, co, args.dry_run)
                results.append((bc, co, return_code == 0))
    
    end_time_all = time.time()
    total_time = end_time_all - start_time_all
    
    # Print summary
    print("\n" + "="*80)
    print("SIMULATION SUMMARY")
    print("="*80)
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Combinations run: {len(results)}")
    
    successes = sum(1 for _, _, success in results if success)
    print(f"Successful: {successes}/{len(results)}")
    
    print("\nResults by combination:")
    for bc, co, success in results:
        status = "SUCCESS" if success else "FAILED"
        print(f"  {bc:<15} + {co:<15} = {status}")

if __name__ == "__main__":
    main()