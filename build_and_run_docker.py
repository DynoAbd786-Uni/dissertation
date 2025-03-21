#!/usr/bin/env python3
"""
Docker build and run script for the XLB dissertation project.
This script handles building the Docker image and running containers
with various configurations.
"""

import argparse
import os
import subprocess
import sys

def check_docker_installed():
    """Check if Docker is installed"""
    try:
        subprocess.run(['docker', '--version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        print("Error: Docker is not installed or not in PATH")
        return False

def check_nvidia_docker():
    """Check if NVIDIA Docker runtime is available and working"""
    try:
        # First check if nvidia-smi works on the host
        nvidia_smi = subprocess.run(['nvidia-smi'], check=False, 
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if nvidia_smi.returncode != 0:
            print("Warning: nvidia-smi failed - GPU might not be available")
            return False
            
        # Then check if nvidia-container-cli is available
        nvidia_container = subprocess.run(['which', 'nvidia-container-cli'], check=False,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        return nvidia_container.returncode == 0
    except subprocess.SubprocessError:
        print("Warning: Could not verify NVIDIA Docker support")
        return False

def build_image(tag="dissertation", dockerfile="Dockerfile"):
    """Build the Docker image"""
    print(f"Building Docker image '{tag}'...")
    try:
        subprocess.run(['docker', 'build', '-t', tag, '-f', dockerfile, '.'], check=True)
        print(f"✅ Image '{tag}' built successfully!")
        return True
    except subprocess.SubprocessError as e:
        print(f"❌ Failed to build image: {e}")
        return False

def run_container(tag="dissertation", mode="standard", script=None, gpu=True):
    """Run the Docker container in specified mode"""
    pwd = os.getcwd()
    
    # Use sudo consistently (or remove it in all places if you fixed permissions)
    cmd = ['sudo', 'docker', 'run']
    
    # Add GPU support if requested and available
    if gpu:
        # Check both ways to support older Docker versions
        has_nvidia = check_nvidia_docker()
        if has_nvidia:
            cmd.extend(['--gpus', 'all'])
            print("GPU support enabled")
        else:
            print("Warning: GPU support requested but not available")
    
    # Mount volumes for code and results
    cmd.extend(['-v', f"{pwd}:/app", '-v', f"{pwd}/results:/app/results"])
    
    # Configure container based on mode
    if mode == "interactive":
        cmd.extend(['-it', tag, '/bin/bash'])
    elif mode == "jupyter":
        cmd.extend(['-p', '8888:8888', tag, 
                   'jupyter', 'notebook', '--ip=0.0.0.0', '--allow-root', '--no-browser'])
    else:  # standard mode
        if script:
            cmd.extend([tag, 'python', script])
        else:
            cmd.append(tag)  # Use default CMD from Dockerfile
    
    # Run the container
    print(f"Running container in {mode} mode...")
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.SubprocessError as e:
        print(f"❌ Container execution failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Docker build and run script for XLB dissertation')
    parser.add_argument('--build', action='store_true', help='Build the Docker image')
    parser.add_argument('--run', action='store_true', help='Run the Docker container')
    parser.add_argument('--mode', choices=['standard', 'interactive', 'jupyter'], 
                       default='standard', help='Container run mode')
    parser.add_argument('--script', help='Custom script to run (for standard mode)')
    parser.add_argument('--tag', default='dissertation', help='Docker image tag')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU support')
    
    args = parser.parse_args()
    
    # Default: both build and run if neither specified
    if not args.build and not args.run:
        args.build = True
        args.run = True
    
    # Check Docker installation
    if not check_docker_installed():
        sys.exit(1)
    
    # Build image if requested
    if args.build:
        if not build_image(tag=args.tag):
            sys.exit(1)
    
    # Run container if requested
    if args.run:
        run_container(tag=args.tag, mode=args.mode, script=args.script, gpu=not args.no_gpu)

if __name__ == "__main__":
    main()