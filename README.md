# Aneurysm Simulation and Visualization

This repository contains a comprehensive solution for simulating and visualizing blood flow in aneurysms. The project consists of two main components:

1. **Simulation Source (`simulation_src/`)**: A computational fluid dynamics (CFD) simulation framework for modeling blood flow in arterial geometries with aneurysms.
2. **Visualization Tools (`visualisation_src/`)**: Tools for analyzing and visualizing the simulation results, including interactive Jupyter notebooks.

## Table of Contents

- [Requirements](#requirements)
- [Quick Start with Docker](#quick-start-with-docker)
  - [Using build_and_run_docker.py Script](#using-build_and_run_dockerpy-script)
  - [Using Docker Compose](#using-docker-compose)
  - [Manual Docker Commands](#manual-docker-commands)
- [Simulation Source Code](#simulation-source-code)
  - [Features](#features)
  - [Directory Structure](#directory-structure)
  - [Running Simulations](#running-simulations)
  - [Customizing Simulations](#customizing-simulations)
- [Visualization Tools](#visualization-tools)
  - [VTK Visualization](#vtk-visualization)
  - [Interactive Analysis](#interactive-analysis)
  - [Available Visualizations](#available-visualizations)
- [Data Output](#data-output)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Requirements

### System Requirements
- CUDA-compatible NVIDIA GPU (recommended for faster simulations)
- Docker and Docker Compose
- 8GB+ RAM
- 5GB+ free disk space

### Software Dependencies
All dependencies are handled by the Docker container. If running locally:
- Python 3.8+
- CUDA Toolkit 11.0+ (for GPU acceleration)
- Python packages listed in `requirements.txt`

## Quick Start with Docker

The easiest way to run simulations and visualizations is using the provided Docker container. There are multiple ways to build and run the container:

### Using build_and_run_docker.py Script

The `build_and_run_docker.py` script provides a convenient way to build and run Docker containers with various configurations.

```bash
# Build and run in standard mode (runs the default simulation)
python build_and_run_docker.py

# Build the image only
python build_and_run_docker.py --build

# Run in interactive mode (bash shell)
python build_and_run_docker.py --run --mode interactive

# Run in Jupyter mode (starts a notebook server)
python build_and_run_docker.py --run --mode jupyter

# Run a specific simulation script
python build_and_run_docker.py --run --script simulation_src/standard_run.py

# Run without GPU support
python build_and_run_docker.py --run --no-gpu
```

### Using Docker Compose

```bash
# Build and start the container in detached mode
docker-compose up -d

# Check if the container is running
docker ps

# Stop the container
docker-compose down
```

### Manual Docker Commands

```bash
# Build the Docker image
docker build -t dissertation .

# Run with GPU support
docker run --gpus all -v ${PWD}:/app -v ${PWD}/results:/app/results dissertation

# Run in interactive mode
docker run -it --gpus all -v ${PWD}:/app -v ${PWD}/results:/app/results dissertation /bin/bash

# Run Jupyter Notebook
docker run -p 8888:8888 --gpus all -v ${PWD}:/app -v ${PWD}/results:/app/results dissertation jupyter notebook --ip=0.0.0.0 --allow-root --no-browser

# Execute a command in a running container
docker exec dissertation python simulation_src/standard_run.py
```

## Simulation Source Code

The simulation source code is located in the `simulation_src/` directory and provides a computational fluid dynamics framework for modeling blood flow.

### Features

- Lattice Boltzmann Method (LBM) for fluid dynamics
- Non-Newtonian blood flow modeling with Carreau-Yasuda model
- Pulsatile flow support for realistic cardiac cycles
- GPU acceleration for faster simulations
- Customizable boundary conditions
- VTK output for detailed analysis

### Directory Structure

- `simulation_src/`: Root directory for simulation code
  - `boundary_conditions/`: Handles inlet, outlet, and wall boundary conditions
  - `collision/`: Collision operators for the LBM method
  - `models/`: Physical models for blood rheology
  - `profiles/`: Flow profiles for boundary conditions
  - `stepper/`: Time stepping and simulation control
  - `utils/`: Utility functions

### Running Simulations

The primary entry point is `standard_run.py`:

```bash
python simulation_src/standard_run.py [options]
```

#### Command Line Options

- `--reynolds`: Reynolds number for the simulation (default: 400)
- `--steps`: Total number of simulation steps (default: 150000)
- `--save-interval`: Interval for saving simulation data (default: 1000)
- `--vtk-interval`: Interval for saving VTK files (default: 1000)
- `--width`: Width of the simulation domain (default: 100)
- `--height`: Height of the simulation domain (default: 200)
- `--aneurysm`: Include aneurysm geometry (default: True)
- `--pulsatile`: Use pulsatile flow (default: True)

### Customizing Simulations

To customize the simulation parameters, modify the configuration files:

1. Edit the `params/` directory files for flow parameters
2. Create custom boundary conditions in `simulation_src/boundary_conditions/`
3. Modify the geometry in `simulation_src/utils/geometry.py`

## Visualization Tools

The visualization tools in `visualisation_src/` provide interactive ways to analyze simulation results.

### VTK Visualization

The main visualization notebook is `vtk_visualization.ipynb`, which provides:

1. Interactive field visualization
2. Multi-frame analysis
3. Vector field visualization
4. Streamline generation
5. Curl (vorticity) and divergence analysis

### Interactive Analysis

To use the visualization notebook:

1. Start Jupyter Notebook (locally or via Docker)
2. Open `visualisation_src/vtk_visualization.ipynb`
3. Run the cells to load VTK files
4. Use the interactive widgets to explore the data

### Available Visualizations

- **Field Visualization**: View any scalar field (velocity magnitude, pressure, etc.)
- **Vector Field**: Interactive visualization of velocity vectors
- **Curl Analysis**: Analyze vorticity in the flow
- **Divergence Analysis**: Examine flow expansion/contraction
- **Frame Comparison**: Compare frames side-by-side with difference visualization
- **Trend Analysis**: Track field values across all simulation frames

## Data Output

Simulation results are stored in multiple formats:

- **VTK Files**: Contains full 3D data fields for velocity, pressure, and other variables
- **Images**: Rendered snapshots of the simulation state
- **Parameter Files**: Records of simulation parameters

The default output location is the `results/` directory, with subdirectories for each simulation run.

## Examples

The `examples/` directory contains sample simulations and visualizations:

- Basic arterial flow
- Aneurysm with pulsatile flow
- Stenosis models

## Troubleshooting

### Common Issues

1. **GPU Not Detected**:
   - Ensure NVIDIA drivers are properly installed
   - Check `nvidia-smi` works on your host
   - Verify Docker has GPU access

2. **Out of Memory Errors**:
   - Reduce domain size
   - Decrease the number of saved timesteps
   - Run on a machine with more GPU memory

3. **Visualization Problems**:
   - Ensure VTK files are generated correctly
   - Check file paths in the visualization notebook
   - Update matplotlib and ipywidgets if interactive widgets don't display correctly

### Getting Help

For additional support, please check the issues section of the repository or contact the project maintainers.
