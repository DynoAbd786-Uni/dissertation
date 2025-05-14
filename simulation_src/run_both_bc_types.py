#!/usr/bin/env python3
"""
Automated script to run pipe simulations with both boundary condition types (zh and tdzh).
This script sequentially runs the pipe_batch_run.py script with both BC types.
"""

import subprocess
import argparse
import sys
import time
from datetime import datetime
import os

def run_command(command, description):
    """Execute a command and print its output in real-time"""
    print(f"\n{'='*80}")
    print(f"EXECUTING: {description}")
    print(f"COMMAND: {' '.join(command)}")
    print(f"TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    # Create process
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Print output in real-time
    for line in iter(process.stdout.readline, ''):
        sys.stdout.write(line)
        sys.stdout.flush()
    
    # Wait for process to complete
    return_code = process.wait()
    
    if return_code == 0:
        print(f"\n✅ {description} completed successfully.")
        return True
    else:
        print(f"\n❌ {description} failed with return code {return_code}.")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run pipe simulations with both boundary condition types")
    parser.add_argument("--length", type=float, default=15, help="Vessel length in mm (default: 15)")
    parser.add_argument("--diameter", type=float, default=6.5, help="Vessel diameter in mm (default: 6.5)")
    parser.add_argument("--resolution", type=float, default=0.02, help="Spatial resolution in mm (default: 0.02)")
    parser.add_argument("--dt", type=float, default=1e-5, help="Time step in seconds (default: 1e-5)")
    parser.add_argument("--duration", type=float, default=1.0, help="Simulation duration in seconds (default: 1.0)")
    parser.add_argument("--warmup", type=float, default=0, help="Warmup time in seconds (default: 0)")
    parser.add_argument("--skip-zh", action="store_true", help="Skip standard Zou-He simulation")
    parser.add_argument("--skip-tdzh", action="store_true", help="Skip time-dependent Zou-He simulation")
    
    args = parser.parse_args()
    
    # Store start time
    start_time = time.time()
    
    # Create commands for both boundary condition types
    common_args = [
        f"--length={args.length}",
        f"--diameter={args.diameter}",
        f"--resolution={args.resolution}",
        f"--dt={args.dt}",
        f"--duration={args.duration}",
        f"--warmup={args.warmup}"
    ]
    
    zh_command = [sys.executable, "pipe_batch_run.py", "--bc_type=zh"] + common_args
    tdzh_command = [sys.executable, "pipe_batch_run.py", "--bc_type=tdzh"] + common_args
    
    success = True
    
    # Run standard Zou-He simulation
    if not args.skip_zh:
        zh_success = run_command(
            zh_command, 
            "Standard Zou-He Boundary Condition Simulation"
        )
        success = success and zh_success
    else:
        print("Skipping standard Zou-He simulation as requested.")
    
    # Run time-dependent Zou-He simulation
    if not args.skip_tdzh:
        tdzh_success = run_command(
            tdzh_command, 
            "Time-Dependent Zou-He Boundary Condition Simulation"
        )
        success = success and tdzh_success
    else:
        print("Skipping time-dependent Zou-He simulation as requested.")
    
    # Calculate total runtime
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"EXECUTION SUMMARY")
    print(f"{'='*80}")
    print(f"Total runtime: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    if success:
        print("✅ All requested simulations completed successfully.")
    else:
        print("❌ One or more simulations failed. Please check the logs for details.")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
