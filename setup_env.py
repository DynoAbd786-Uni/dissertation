#!/usr/bin/env python3
"""
Set up conda environment and install requirements for the dissertation project.
"""

import subprocess
import os
import sys
import json

def run_command(command, shell=False):
    """Run a command and return its output"""
    try:
        if shell:
            process = subprocess.run(command, shell=True, check=True, 
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            process = subprocess.run(command, check=True, 
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return process.stdout.decode().strip()
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e.stderr.decode().strip()}")
        return None

def check_env_exists(env_name):
    """Check if conda environment exists"""
    result = run_command(['conda', 'env', 'list', '--json'])
    if not result:
        return False
    
    env_data = json.loads(result)
    envs = [os.path.basename(env) for env in env_data['envs']]
    return env_name in envs

def create_conda_env(env_name):
    """Create conda environment"""
    print(f"Creating conda environment '{env_name}'...")
    result = run_command(['conda', 'create', '-n', env_name, 'python=3.10', '-y'])
    return result is not None

def install_requirements(env_name):
    """Install packages from requirements.txt"""
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    
    if not os.path.exists(requirements_path):
        print(f"Error: Could not find requirements.txt at {requirements_path}")
        return False
    
    print(f"Installing requirements in environment '{env_name}'...")
    
    # Use conda run to execute pip in the environment
    cmd = f"conda run -n {env_name} pip install -r {requirements_path}"
    result = run_command(cmd, shell=True)
    
    return result is not None

def main():
    env_name = "xlb"
    
    # Check if conda is available
    if not run_command(['conda', '--version']):
        print("Error: Conda is not available. Please install conda first.")
        return
    
    # Check if environment exists
    if check_env_exists(env_name):
        print(f"Conda environment '{env_name}' already exists.")
    else:
        if not create_conda_env(env_name):
            print(f"Failed to create conda environment '{env_name}'.")
            return
    
    # Install requirements
    if install_requirements(env_name):
        print("\n✅ Setup complete!")
        print(f"\nTo activate the environment, run:")
        print(f"    conda activate {env_name}")
    else:
        print("\n❌ Setup incomplete. Please check the errors above.")

if __name__ == "__main__":
    main()