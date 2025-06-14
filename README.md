# Advanced CFD Blood Flow Simulation and Visualization

This repository contains a comprehensive solution for simulating and visualizing blood flow in arterial geometries with aneurysms using the Lattice Boltzmann Method (LBM). The project consists of two main components:

1. **Simulation Source (`simulation_src/`)**: A high-performance computational fluid dynamics (CFD) simulation framework for modeling blood flow in arterial geometries with aneurysms.
2. **Visualization Tools (`visualisation_src/`)**: Tools for analyzing and visualizing the simulation results, including interactive Jupyter notebooks.

## 🚀 Key Features

- **High-Performance Computing**: GPU-accelerated simulations achieving 8,000-16,000 MLUPS (Million Lattice Updates Per Second)
- **Advanced Blood Flow Modeling**: Non-Newtonian Carreau-Yasuda rheology model for realistic blood viscosity
- **Comprehensive Spatial Profiles**: Support for uniform, Poiseuille, and blunted paraboloid velocity profiles optimized for blood flow
- **Pulsatile Flow Support**: Time-dependent boundary conditions with realistic cardiac cycle profiles
- **Multi-Domain Analysis**: Both standard and long pipe configurations for detailed performance comparison
- **Automated Batch Processing**: Comprehensive parameter studies with organized results management
- **Professional Visualization**: VTK output for ParaView analysis and interactive Jupyter notebooks

## Table of Contents

- [Requirements](#requirements)
- [Quick Start with Docker](#quick-start-with-docker)
  - [Using build_and_run_docker.py Script](#using-build_and_run_dockerpy-script)
  - [Using Docker Compose](#using-docker-compose)
  - [Manual Docker Commands](#manual-docker-commands)
- [Simulation Source Code](#simulation-source-code)
  - [Advanced Features](#advanced-features)
  - [Performance Characteristics](#performance-characteristics)
  - [Directory Structure](#directory-structure)
  - [Running Simulations](#running-simulations)
  - [Spatial Velocity Profiles](#spatial-velocity-profiles)
  - [Customizing Simulations](#customizing-simulations)
- [Visualization Tools](#visualization-tools)
  - [VTK Visualization](#vtk-visualization)
  - [Interactive Analysis](#interactive-analysis)
  - [Available Visualizations](#available-visualizations)
- [Data Output](#data-output)
- [Performance Analysis](#performance-analysis)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Requirements

### System Requirements
- CUDA-compatible NVIDIA GPU (recommended for optimal performance)
- Docker and Docker Compose
- 8GB+ RAM (16GB+ recommended for large simulations)
- 10GB+ free disk space (for comprehensive simulation suites)

### Software Dependencies
All dependencies are handled by the Docker container. If running locally:
- Python 3.8+
- CUDA Toolkit 11.0+ (for GPU acceleration)
- JAX or Warp backend support
- Python packages listed in `requirements.txt`

### Performance Notes
- **GPU Acceleration**: Achieves 8,000-16,000 MLUPS on modern GPUs
- **CPU Fallback**: Available but significantly slower (~100-500 MLUPS)
- **Memory Usage**: Scales with domain size; large simulations may require 16GB+ RAM

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

### Advanced Features

- **Lattice Boltzmann Method (LBM)**: High-performance fluid dynamics simulation
- **Non-Newtonian Blood Flow**: Carreau-Yasuda rheology model with shear-rate dependent viscosity
- **Multiple Spatial Velocity Profiles**:
  - **Uniform**: Flat velocity profile for testing and comparison
  - **Poiseuille**: Parabolic profile for developed laminar flow
  - **Blunted Paraboloid**: Power-law profile optimized for blood flow (n=1.7)
- **Pulsatile Flow Support**: Realistic cardiac cycle with time-dependent boundary conditions
- **Multi-Backend Support**: JAX (CPU/GPU) and Warp (GPU) acceleration
- **Advanced Boundary Conditions**:
  - Standard and time-dependent Zou-He inlet conditions
  - Extrapolation outflow boundaries
  - Full-way bounce-back walls
- **Collision Operators**:
  - Standard BGK for Newtonian fluids
  - Non-Newtonian BGK with Carreau-Yasuda model for blood
- **Comprehensive Batch System**: Automated parameter studies with organized results
- **Performance Optimization**: Domain-size dependent MLUPS scaling
- **Professional Output**: VTK files for ParaView visualization

### Performance Characteristics

- **Standard Pipe Simulations** (751×330 nodes): ~3,000-3,700 MLUPS
- **Long Pipe Simulations** (10,001×86 nodes): ~8,000-13,000 MLUPS  
- **Aneurysm Simulations** (1751×530 nodes): ~12,000-16,000 MLUPS
- **GPU Utilization**: Larger domains achieve higher MLUPS due to better parallelization

### Directory Structure

```
simulation_src/                         # Root simulation framework
├── run_all_sim_configs.py             # 🚀 Main batch execution script  
├── pipe_run.py                         # Individual pipe simulations
├── aneurysm_run.py                     # Individual aneurysm simulations
├── models/                             # Physical simulation models
│   ├── pipe_model_2D.py               # Pipe flow implementation
│   └── aneurysm_model_2D.py            # Aneurysm flow implementation
├── boundary_conditions/                # Boundary condition implementations
│   ├── bc_zouhe_time_dependant.py     # Time-dependent inlet conditions
│   └── __init__.py
├── collision/                          # LBM collision operators
│   ├── bgk_newtonian.py               # Standard BGK collision
│   ├── bgk_non_newtonian.py           # Non-Newtonian BGK (Carreau-Yasuda)
│   └── __init__.py
├── profiles/                           # Flow and spatial profiles
│   ├── spacial_flow_profiles.py       # Spatial velocity distributions
│   └── __init__.py
├── stepper/                            # Time integration
│   └── custom_nse_stepper.py          # Custom Navier-Stokes stepper
├── utils/                              # Utility functions
│   ├── load_csv.py                    # Data loading utilities
│   ├── wss_calculation.py             # Wall shear stress computation
│   ├── directory_utils.py             # File management
│   └── constants.py                   # Physical constants
├── examples/                           # Example configurations
├── 📄 SPATIAL_PROFILES_SUMMARY.md     # Spatial profiles documentation
├── 📄 MLUPS_FIXES_SUMMARY.md          # Performance optimization guide
└── 📄 UNIFORM_SPATIAL_PROFILE_IMPLEMENTATION.md  # Uniform profile guide
```

### Running Simulations

The simulation framework provides multiple entry points for different types of simulations with comprehensive configuration options:

#### Comprehensive Batch Execution (Recommended)

Run complete simulation suites with automatic organization:

```bash
# Complete simulation suite (15 pipe + 3 aneurysm = 18 total simulations)
# Includes: 12 standard pipe + 3 long pipe + aneurysm configurations
python simulation_src/run_all_sim_configs.py

# Comprehensive pipe study only (15 simulations: 12 standard + 3 long)
python simulation_src/run_all_sim_configs.py --pipe-only

# Only aneurysm simulations with all spatial profiles
python simulation_src/run_all_sim_configs.py --aneurysm-only

# Test configurations without running (recommended first step)
python simulation_src/run_all_sim_configs.py --dry-run
```

#### Focused Configuration Studies

```bash
# Only standard pipe configurations (12 simulations)
python simulation_src/run_all_sim_configs.py --pipe-only --standard-pipe-only

# Only long pipe configurations (3 simulations)
python simulation_src/run_all_sim_configs.py --pipe-only --long-pipe

# Specific boundary condition study
python simulation_src/run_all_sim_configs.py --pipe-only --boundary-condition time-dependent

# Blood flow optimized configurations
python simulation_src/run_all_sim_configs.py \
  --boundary-condition time-dependent \
  --collision-operator non-newtonian \
  --spatial-profile blunted_paraboloid
```

#### Individual Pipe Simulations

```bash
# Basic pipe simulation with Poiseuille profile
python simulation_src/pipe_run.py \
  --boundary-condition standard \
  --collision-operator standard \
  --spatial-profile poiseuille

# Optimized blood flow simulation
python simulation_src/pipe_run.py \
  --boundary-condition time-dependent \
  --collision-operator non-newtonian \
  --spatial-profile blunted_paraboloid \
  --power-law-exponent 1.7 \
  --generate-pngs

# Long pipe for enhanced performance
python simulation_src/pipe_run.py \
  --boundary-condition standard \
  --collision-operator non-newtonian \
  --spatial-profile uniform \
  --long-pipe \
  --generate-pngs
```

#### Individual Aneurysm Simulations

```bash
# Standard aneurysm with Poiseuille profile  
python simulation_src/aneurysm_run.py \
  --spatial-profile poiseuille \
  --generate-pngs

# Blood flow aneurysm simulation (recommended)
python simulation_src/aneurysm_run.py \
  --spatial-profile blunted_paraboloid \
  --power-law-exponent 1.7 \
  --generate-pngs

# Uniform profile for comparison
python simulation_src/aneurysm_run.py \
  --spatial-profile uniform \
  --generate-pngs
```

#### Command Line Options Reference

**run_all_sim_configs.py:**
- `--pipe-only`: Run only pipe simulations (default: includes aneurysm)
- `--aneurysm-only`: Run only aneurysm simulations
- `--long-pipe`: Run ONLY long pipe configurations (3 simulations)
- `--standard-pipe-only`: Run ONLY standard pipe configurations (12 simulations)
- `--boundary-condition {standard,time-dependent,all}`: Filter by boundary condition
- `--collision-operator {standard,non-newtonian,all}`: Filter by collision operator  
- `--spatial-profile {uniform,poiseuille,blunted_paraboloid,all}`: Filter by spatial profile
- `--aneurysm-spatial-profile {uniform,poiseuille,blunted_paraboloid,all}`: Aneurysm spatial profiles
- `--dry-run`: Test configurations without executing simulations

**pipe_run.py:**
- `--boundary-condition {standard,time-dependent}`: Boundary condition type
- `--collision-operator {standard,non-newtonian}`: Collision operator type
- `--spatial-profile {uniform,poiseuille,blunted_paraboloid}`: Spatial velocity profile
- `--power-law-exponent FLOAT`: Power-law exponent for blunted paraboloid (default: 1.7)
- `--long-pipe`: Use longer pipe geometry (800mm vs 15mm)
- `--generate-pngs`: Generate PNG images during simulation

**aneurysm_run.py:**
- `--spatial-profile {uniform,poiseuille,blunted_paraboloid}`: Spatial velocity profile
- `--power-law-exponent FLOAT`: Power-law exponent for blunted paraboloid (default: 1.7)
- `--generate-pngs`: Generate PNG images during simulation

### Spatial Velocity Profiles

The simulation framework supports three spatial velocity profiles:

#### 1. **Uniform Profile** (`--spatial-profile uniform`)
- **Description**: Flat velocity distribution across the inlet
- **Use Case**: Testing, validation, and comparison baseline
- **Characteristics**: Constant velocity magnitude across the channel height
- **Physics**: Non-physical but useful for numerical validation

#### 2. **Poiseuille Profile** (`--spatial-profile poiseuille`)  
- **Description**: Parabolic velocity distribution for developed laminar flow
- **Use Case**: Fully developed flow in long straight channels
- **Characteristics**: Maximum velocity at center, zero at walls
- **Physics**: Analytical solution for steady laminar flow

#### 3. **Blunted Paraboloid Profile** (`--spatial-profile blunted_paraboloid`)
- **Description**: Power-law profile optimized for blood flow
- **Use Case**: Realistic blood flow modeling (recommended)
- **Characteristics**: Flattened center with steep near-wall gradients
- **Physics**: Power-law exponent n=1.7 matches blood flow characteristics
- **Configuration**: `--power-law-exponent 1.7` (adjustable)

### Simulation Configurations Matrix

| Configuration | Boundary Condition | Collision Operator | Best Spatial Profile | Use Case |
|---------------|-------------------|-------------------|---------------------|----------|
| **Basic Flow** | standard | standard | poiseuille | Simple validation |
| **Blood Flow** | time-dependent | non-newtonian | blunted_paraboloid | Realistic modeling |
| **High Performance** | standard | non-newtonian | uniform | MLUPS benchmarking |
| **Full Physics** | time-dependent | non-newtonian | blunted_paraboloid | Research applications |

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

Simulation results are stored in multiple organized formats with comprehensive metadata:

### Output Structure

```
results/
├── pipe_flow/                          # Pipe simulation results
│   ├── zh_bgk_uniform_standard/        # Standard configurations
│   ├── zh_nnbgk_blunted_standard/      # 
│   ├── tdzh_nnbgk_poiseuille_standard/ # 
│   ├── zh_nnbgk_uniform_long/          # Long pipe configurations
│   └── ...                             # (15 pipe configurations total)
├── aneurysm_flow/                      # Aneurysm simulation results  
│   ├── CCA_simulation_results_nnbgk_tdzh_blunted/
│   └── ...                             # (Aneurysm configurations)
└── logs/                               # Simulation execution logs
    ├── pipe_zh_nnbgk_uniform_long_*.log
    ├── aneurysm_tdzh_nnbgk_blunted_*.log
    └── ...
```

### File Types

- **VTK Files** (`*.vtk`): Complete 3D field data for ParaView visualization
  - Velocity fields (u_x, u_y, u_magnitude)
  - Pressure/density fields (rho)
  - Wall shear stress (WSS) data
  - Boundary masks (wall, inlet, outlet)
  
- **PNG Images** (`*.png`): Real-time visualization snapshots (optional)
  - Velocity magnitude contours
  - Wall shear stress distributions
  - Boundary visualization
  
- **Parameter Files** (`*.json`): Complete simulation metadata
  - Physical parameters (viscosity, time step, grid resolution)
  - Numerical settings (backend, collision operator, boundary conditions)
  - Performance metrics (MLUPS, runtime, efficiency)
  - Reproducibility information

### Naming Convention

All outputs follow a consistent naming pattern for easy identification:

- **Boundary Conditions**: `zh` (standard Zou-He), `tdzh` (time-dependent Zou-He)
- **Collision Operators**: `bgk` (standard BGK), `nnbgk` (non-Newtonian BGK)
- **Spatial Profiles**: `uniform`, `poiseuille`, `blunted`
- **Pipe Types**: `standard`, `long`

Example: `zh_nnbgk_blunted_long` = Standard Zou-He + Non-Newtonian BGK + Blunted Paraboloid + Long Pipe

## Performance Analysis

### MLUPS Performance Scaling

The simulation framework demonstrates domain-size dependent performance scaling:

| Domain Type | Grid Size | Nodes | Typical MLUPS | Use Case |
|-------------|-----------|-------|---------------|----------|
| **Standard Pipe** | 751×330 | 247,830 | 3,000-3,700 | Detailed analysis |
| **Long Pipe** | 10,001×86 | 860,086 | 8,000-13,000 | High performance |
| **Aneurysm** | 1751×530 | 928,030 | 12,000-16,000 | Complex geometry |

### Performance Optimization

- **GPU Utilization**: Larger domains achieve higher MLUPS due to better parallelization
- **Memory Bandwidth**: Optimal balance between compute and memory access
- **Kernel Efficiency**: Better warp/thread block utilization in larger simulations
- **Backend Selection**: Warp GPU > JAX GPU > JAX CPU for performance

### Benchmarking Commands

```bash
# Performance comparison across configurations
python simulation_src/run_all_sim_configs.py --pipe-only --dry-run

# High-performance long pipe benchmark
python simulation_src/run_all_sim_configs.py --pipe-only --long-pipe

# Complete performance characterization
python simulation_src/run_all_sim_configs.py --dry-run
```

## Examples

The repository provides comprehensive examples for various simulation scenarios:

### Quick Start Examples

```bash
# 1. Basic validation run (fastest)
python simulation_src/pipe_run.py \
  --boundary-condition standard \
  --collision-operator standard \
  --spatial-profile uniform \
  --generate-pngs

# 2. Realistic blood flow simulation  
python simulation_src/pipe_run.py \
  --boundary-condition time-dependent \
  --collision-operator non-newtonian \
  --spatial-profile blunted_paraboloid \
  --generate-pngs

# 3. High-performance benchmark
python simulation_src/pipe_run.py \
  --boundary-condition standard \
  --collision-operator non-newtonian \
  --spatial-profile uniform \
  --long-pipe \
  --generate-pngs

# 4. Complete aneurysm analysis
python simulation_src/aneurysm_run.py \
  --spatial-profile blunted_paraboloid \
  --generate-pngs
```

### Research-Grade Examples

```bash
# Comprehensive parameter study (18 simulations)
python simulation_src/run_all_sim_configs.py

# Blood flow optimization study
python simulation_src/run_all_sim_configs.py \
  --boundary-condition time-dependent \
  --collision-operator non-newtonian \
  --spatial-profile all

# Performance scaling analysis
python simulation_src/run_all_sim_configs.py --pipe-only
```

### Visualization Examples

After running simulations, use the visualization tools:

```bash
# Start Jupyter for interactive analysis
jupyter notebook visualisation_src/vtk_visualization.ipynb

# ParaView batch processing
paraview results/pipe_flow/zh_nnbgk_blunted_standard/vtk/
```

## Troubleshooting

### Common Issues and Solutions

#### 1. **GPU and Performance Issues**

**GPU Not Detected**:
```bash
# Check GPU availability
nvidia-smi
docker run --gpus all nvidia/cuda:11.0-base nvidia-smi
```
- Ensure NVIDIA drivers are properly installed
- Verify Docker has GPU access (`--gpus all` flag)
- Check CUDA toolkit compatibility

**Low MLUPS Performance**:
- **Expected Values**: 8,000+ MLUPS on modern GPUs, 100-500 on CPU
- **Solutions**: 
  - Use larger domains (`--long-pipe` flag for pipes)
  - Ensure GPU backend is active (check simulation output)
  - Verify CUDA/GPU setup with `nvidia-smi`

#### 2. **Memory and Resource Issues**

**Out of Memory Errors**:
```bash
# Monitor GPU memory usage
nvidia-smi -l 1
```
- **Solutions**:
  - Reduce domain size or time steps
  - Use CPU backend as fallback: `export XLA_FLAGS=--xla_force_host_platform_device_count=1`
  - Increase system RAM (16GB+ recommended for large simulations)

**Slow Simulations**:
- **Standard pipe**: ~3,000 MLUPS expected
- **Long pipe**: ~8,000-13,000 MLUPS expected  
- **Aneurysm**: ~12,000-16,000 MLUPS expected
- Check backend selection in simulation output

#### 3. **Simulation Configuration Issues**

**Numerical Instability**:
```
ERROR: NaN or infinity detected in velocity field
```
- **Causes**: Relaxation parameter too small, high inlet velocity, sharp geometry
- **Solutions**:
  - Check tau value in output (should be 0.55-1.95)
  - Reduce inlet velocity
  - Use `--spatial-profile uniform` for testing

**Missing Results**:
```bash
# Check simulation logs
ls results/logs/
tail results/logs/pipe_*_latest.log
```
- Verify command syntax with `--dry-run` flag first
- Check file permissions in results directory
- Review log files for error messages

#### 4. **Visualization Problems**

**VTK Files Not Loading**:
- Ensure simulations completed successfully (check logs)
- Verify file paths in visualization notebooks
- Use `ls results/*/vtk/` to confirm VTK file generation

**Interactive Widgets Not Working**:
```bash
# Update visualization dependencies
pip install --upgrade matplotlib ipywidgets
jupyter nbextension enable --py widgetsnbextension
```

#### 5. **Docker and Environment Issues**

**Container Build Failures**:
```bash
# Clean rebuild
docker system prune -a
docker build --no-cache -t dissertation .
```

**Permission Issues**:
```bash
# Fix results directory permissions
sudo chown -R $USER:$USER results/
chmod -R 755 results/
```

### Performance Optimization Tips

1. **Use Long Pipe Configurations**: Achieve 2-3x higher MLUPS
2. **Batch Processing**: Run `--dry-run` first to validate configurations
3. **Resource Monitoring**: Use `nvidia-smi` and `htop` to monitor usage
4. **Backend Selection**: Prefer Warp GPU > JAX GPU > JAX CPU
5. **Memory Management**: Close other applications for large simulations

### Getting Help

For additional support:

1. **Check Documentation**: Review `simulation_src/*.md` files for detailed information
2. **Validate Setup**: Run example commands with `--dry-run` flag
3. **Performance Baseline**: Compare MLUPS against expected values above
4. **Log Analysis**: Check `results/logs/` for detailed error information

### Debug Commands

```bash
# Quick system check
python -c "import jax; print('JAX devices:', jax.devices())"
nvidia-smi

# Validate configuration
python simulation_src/run_all_sim_configs.py --dry-run --pipe-only

# Test minimal simulation
python simulation_src/pipe_run.py \
  --boundary-condition standard \
  --collision-operator standard \
  --spatial-profile uniform
```

---

## 🎯 Recent Improvements and Features

This simulation framework has been extensively enhanced with the following key improvements:

### ✅ **Performance Optimizations**
- **MLUPS Calculation Fixes**: Robust performance metrics with safety checks and accurate averaging
- **Domain-Size Scaling**: Automatic optimization for different simulation scales
- **GPU Utilization**: Enhanced parallelization for 8,000-16,000 MLUPS performance

### ✅ **Advanced Spatial Profiles**  
- **Three Profile Types**: Uniform, Poiseuille, and Blunted Paraboloid implementations
- **Blood Flow Modeling**: Optimized power-law profiles (n=1.7) for realistic blood flow
- **Flexible Configuration**: Command-line control over spatial velocity distributions

### ✅ **Comprehensive Batch System**
- **Multi-Configuration Support**: 15 pipe + 3 aneurysm configuration matrix
- **Intelligent Naming**: Organized results with descriptive directory names
- **Performance Comparison**: Standard vs long pipe domains for scaling analysis

### ✅ **Enhanced Usability**
- **Improved CLI**: Comprehensive command-line options with validation
- **Error Handling**: Robust error checking and user-friendly messages  
- **Documentation**: Extensive guides and troubleshooting resources

### ✅ **Professional Output**
- **VTK Integration**: Full ParaView compatibility with rich field data
- **Metadata Tracking**: Complete simulation parameters and performance metrics
- **Reproducibility**: JSON parameter files for research reproducibility

## 📊 Expected Performance Benchmarks

When properly configured, you should see:

- **Standard Pipe Simulations**: 3,000-3,700 MLUPS
- **Long Pipe Simulations**: 8,000-13,000 MLUPS  
- **Aneurysm Simulations**: 12,000-16,000 MLUPS
- **Complete Batch Run**: ~15-20 minutes on modern GPU hardware

---

*This README reflects the comprehensive improvements made to create a production-ready CFD simulation framework optimized for blood flow analysis and high-performance computing.*
