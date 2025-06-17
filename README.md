# Advanced CFD Blood Flow Simulation and Visualization

This repository contains a comprehensive solution for simulating and visualizing blood flow in arterial geometries with aneurysms using the Lattice Boltzmann Method (LBM). The project consists of two main components:

1. **Simulation Source (`simulation_src/`)**: A high-performance computational fluid dynamics (CFD) simulation framework for modeling blood flow in arterial geometries with aneurysms.
2. **Visualization Tools (`visualisation_src/`)**: Tools for analyzing and visualizing the simulation results, including interactive Jupyter notebooks.

## 🚀 Key Features

- **High-Performance Computing**: GPU-accelerated simulations achieving 8,000-16,000 MLUPS (Million Lattice Updates Per Second)
- **Advanced Performance Testing**: Dedicated MLUPS benchmarking tools with JAX vs WARP backend comparison
- **Advanced Blood Flow Modeling**: Non-Newtonian Carreau-Yasuda rheology model for realistic blood viscosity
- **Comprehensive Spatial Profiles**: Support for uniform, Poiseuille, and blunted paraboloid velocity profiles optimized for blood flow
- **Pulsatile Flow Support**: Time-dependent boundary conditions with realistic cardiac cycle profiles
- **Multi-Domain Analysis**: Both standard (751×330) and long pipe (10,001×86) configurations for performance scaling analysis
- **Intelligent Batch Processing**: Automated execution of 18 total configurations (15 pipe + 3 aneurysm) with organized results management
- **Performance Scaling**: Domain-size dependent MLUPS optimization (3,000-16,000 MLUPS range)
- **Scientific Performance Testing**: Process-isolated backend testing with thermal management and pure simulation measurement
- **Professional Visualization**: VTK output for ParaView analysis and interactive Jupyter notebooks
- **Robust Error Handling**: Comprehensive safety checks, validation, and user-friendly error messages

## Table of Contents

- [Quick Start](#quick-start)
- [System Requirements](#system-requirements)
- [Docker Setup](#docker-setup)
- [Simulation Framework](#simulation-framework)
  - [Running Simulations](#running-simulations)
  - [Configuration Options](#configuration-options)
  - [Batch Execution](#batch-execution)
- [Results and Visualization](#results-and-visualization)
- [Performance Guide](#performance-guide)
- [Advanced Performance Testing](#advanced-performance-testing)
- [Troubleshooting](#troubleshooting)
- [Technical Documentation](#technical-documentation)

## Quick Start

### 🚀 **Immediate Start (Recommended)**
```bash
# Run complete simulation suite (18 configurations, ~60-70 minutes)
python simulation_src/run_all_sim_configs.py

# Test configurations first (recommended)
python simulation_src/run_all_sim_configs.py --dry-run

# Run single high-performance simulation
python simulation_src/pipe_run.py --long-pipe --generate-pngs
```

### 📊 **Expected Performance**
- **Standard Pipe**: 3,000-3,700 MLUPS (2-3 min each)
- **Long Pipe**: 8,000-13,000 MLUPS (3-4 min each)  
- **Aneurysm**: 12,000-16,000 MLUPS (4-5 min each)

## System Requirements

- **GPU**: CUDA-compatible NVIDIA GPU (recommended for 8,000+ MLUPS)
- **Memory**: 16GB+ RAM for large simulations
- **Storage**: 10GB+ free disk space
- **Software**: Docker and Docker Compose

**CPU Fallback**: Available but significantly slower (~100-500 MLUPS)

## Docker Setup

### Using build_and_run_docker.py (Recommended)
```bash
# Build and run with GPU support
python build_and_run_docker.py

# Interactive mode
python build_and_run_docker.py --run --mode interactive

# Jupyter notebook server
python build_and_run_docker.py --run --mode jupyter
```

### Alternative Docker Methods
```bash
# Docker Compose
docker-compose up -d

# Manual Docker
docker build -t dissertation .
docker run --gpus all -v ${PWD}:/app dissertation
```

## Simulation Framework

### Core Features
- **Lattice Boltzmann Method (LBM)**: High-performance fluid dynamics
- **Non-Newtonian Blood Flow**: Carreau-Yasuda rheology model
- **Pulsatile Flow**: Time-dependent boundary conditions with cardiac cycle profiles
- **Multi-Backend**: JAX (CPU/GPU) and Warp (GPU) acceleration
- **Professional Output**: VTK files for ParaView visualization

### Running Simulations

#### **Complete Batch Execution (Recommended)**
```bash
# All 18 configurations (15 pipe + 3 aneurysm)
python simulation_src/run_all_sim_configs.py

# Pipe simulations only (15 configurations)
python simulation_src/run_all_sim_configs.py --pipe-only

# Aneurysm simulations only (3 configurations)
python simulation_src/run_all_sim_configs.py --aneurysm-only
```

#### **Individual Simulations**
```bash
# Blood flow simulation (recommended)
python simulation_src/pipe_run.py \
  --boundary-condition time-dependent \
  --collision-operator non-newtonian \
  --spatial-profile blunted_paraboloid \
  --generate-pngs

# High-performance benchmark
python simulation_src/pipe_run.py \
  --boundary-condition standard \
  --collision-operator non-newtonian \
  --spatial-profile uniform \
  --long-pipe

# Aneurysm analysis
python simulation_src/aneurysm_run.py \
  --spatial-profile blunted_paraboloid \
  --generate-pngs
```

### Configuration Options

#### **Boundary Conditions**
- `standard`: Standard Zou-He (steady inlet)
- `time-dependent`: Pulsatile cardiac cycle

#### **Collision Operators**
- `standard`: Standard BGK (Newtonian fluid)
- `non-newtonian`: Carreau-Yasuda blood model

#### **Spatial Profiles**
- `uniform`: Flat velocity (testing/validation)
- `poiseuille`: Parabolic profile (laminar flow)
- `blunted_paraboloid`: Power-law profile (blood flow, n=1.7)

#### **Domain Types**
- **Standard Pipe**: 751×330 (247k nodes) - detailed analysis
- **Long Pipe**: 10,001×86 (860k nodes) - high performance
- **Aneurysm**: 1751×530 (928k nodes) - complex geometry

### Batch Execution

The framework runs **18 total configurations** by default:
- **Standard Pipe**: 12 combinations (2 BC × 2 CO × 3 SP)
- **Long Pipe**: 3 combinations (1 BC × 1 CO × 3 SP)
- **Aneurysm**: 3 combinations (1 BC × 1 CO × 3 SP)

#### **Selective Execution**
```bash
# Standard pipe only (12 combinations)
python simulation_src/run_all_sim_configs.py --standard-pipe-only

# Long pipe only (3 combinations)
python simulation_src/run_all_sim_configs.py --long-pipe

# Specific filters
python simulation_src/run_all_sim_configs.py \
  --boundary-condition time-dependent \
  --collision-operator non-newtonian \
  --spatial-profile blunted_paraboloid
```

## Results and Visualization

### Data Organization

Results follow the naming pattern: `{bc}_{co}_{sp}_{pipe_type}`

- **BC**: `zh` (standard), `tdzh` (time-dependent)
- **CO**: `bgk` (standard), `nnbgk` (non-newtonian)  
- **SP**: `uniform`, `poiseuille`, `blunted`
- **Pipe**: `standard`, `long`

Example: `zh_nnbgk_blunted_long` = Standard BC + Non-Newtonian CO + Blunted Profile + Long Pipe

### Output Structure
```
results/
├── pipe_flow/                          # 15 pipe configurations
│   ├── zh_bgk_uniform_standard/        # Standard pipe combinations (12)
│   ├── zh_nnbgk_blunted_standard/      
│   ├── tdzh_nnbgk_poiseuille_standard/ 
│   ├── zh_nnbgk_uniform_long/          # Long pipe combinations (3)
│   └── zh_nnbgk_blunted_long/          
├── aneurysm_flow/                      # 3 aneurysm configurations
│   ├── CCA_simulation_results_nnbgk_tdzh_uniform/
│   └── CCA_simulation_results_nnbgk_tdzh_blunted/
└── logs/                               # Timestamped execution logs
    ├── pipe_zh_nnbgk_blunted_long_2025-06-14_12-15-45.log
    └── aneurysm_tdzh_nnbgk_blunted_2025-06-14_12-21-30.log
```

Each simulation directory contains:
- **VTK files**: For ParaView visualization
- **JSON parameters**: Complete configuration and performance metrics
- **PNG images**: Quick visualization (with `--generate-pngs`)

### Visualization Tools

#### **Interactive Analysis**
```bash
# Start Jupyter notebook
jupyter notebook visualisation_src/vtk_visualization.ipynb

# ParaView professional visualization
paraview results/pipe_flow/zh_nnbgk_blunted_standard/vtk/
```

#### **Available Visualizations**
- Field visualization (velocity, pressure, WSS)
- Vector field analysis with interactive controls
- Multi-frame comparison and trend analysis
- Curl (vorticity) and divergence analysis

## Performance Guide

### MLUPS Performance Scaling

**Benchmark Hardware**: Intel 11th Gen CPU, NVIDIA RTX 3060 Mobile (6GB VRAM), 16GB RAM

| Domain Type | Grid Size | Nodes | Typical MLUPS | Runtime | Use Case |
|-------------|-----------|-------|---------------|---------|----------|
| **Standard Pipe** | 751×330 | 247,830 | 3,000-3,700 | 2-3 min | Detailed analysis |
| **Long Pipe** | 10,001×86 | 860,086 | 8,000-13,000 | 3-4 min | High performance |
| **Aneurysm** | 1751×530 | 928,030 | 12,000-16,000 | 4-5 min | Complex geometry |

### Why Larger Domains Achieve Higher MLUPS

1. **GPU Occupancy**: Larger domains saturate GPU cores more effectively
2. **Memory Bandwidth**: Better balance between compute and memory access
3. **Kernel Efficiency**: Reduced launch overhead amortization
4. **Parallelization**: Enhanced warp/thread block efficiency

### Performance Testing
```bash
# Quick validation
python simulation_src/run_all_sim_configs.py --dry-run

# High-performance benchmark
python simulation_src/pipe_run.py --long-pipe

# Complete performance characterization
python simulation_src/run_all_sim_configs.py --pipe-only
```

## Advanced Performance Testing

### 🚀 **Dedicated Performance Testing Framework**

New dedicated performance testing tools for precise MLUPS benchmarking and backend comparison:

#### **Individual Backend Performance Testing**
```bash
# Test JAX backend (CPU) performance
python simulation_src/performance_testing_aneurysm_model.py --backend JAX --duration-seconds 2.0 --warmup-seconds 2.0

# Test WARP backend (GPU) performance  
python simulation_src/performance_testing_aneurysm_model.py --backend WARP --duration-seconds 2.0 --warmup-seconds 2.0
```

#### **Comprehensive Backend Comparison**
```bash
# Run both JAX and WARP tests with proper isolation
python simulation_src/run_all_performance_tests.py

# Quick comparison test
python simulation_src/run_all_performance_tests.py --duration-seconds 1.0 --warmup-seconds 1.0

# Detailed comparison with longer duration
python simulation_src/run_all_performance_tests.py --duration-seconds 5.0 --warmup-seconds 5.0

# Test single backend only
python simulation_src/run_all_performance_tests.py --jax-only
python simulation_src/run_all_performance_tests.py --warp-only

# Skip thermal cooldown for faster testing (less accurate)
python simulation_src/run_all_performance_tests.py --skip-cooldown

# Verbose output with detailed backend information
python simulation_src/run_all_performance_tests.py --verbose
```

### 🔬 **Performance Testing Features**

#### **Scientific Accuracy**:
- **Process Isolation**: Each backend runs in separate subprocess to prevent memory contamination
- **Thermal Management**: GPU cooldown periods between tests to prevent thermal throttling
- **Pure Simulation**: Disables all post-processing and file I/O during measurement
- **MLUPS Calculation**: Measured after warmup completion for stable performance metrics

#### **Comprehensive Results**:
- **Side-by-side Comparison**: Direct JAX vs WARP performance analysis
- **Detailed Metrics**: Average, peak, minimum MLUPS with statistical analysis
- **Thermal Monitoring**: GPU temperature tracking (when available)
- **JSON Export**: Complete performance data saved for analysis

#### **Expected Performance Results**:
```
Backend    Status     Avg MLUPS    Peak MLUPS   Grid Size    Device         
--------------------------------------------------------------------------
JAX        ✅ OK       210.2        1620.5       1751x530     CPU            
WARP       ✅ OK       2134.7       14688.5      1751x530     GPU            

🚀 GPU Performance Advantage:
   WARP (GPU) is 10.2x faster than JAX (CPU)
   Performance gain: 920%
   Wall-clock speedup: 10.3x (43s vs 7m21s)
```

**Note**: The above performance results were obtained from individual test executions using:
```bash
python simulation_src/performance_testing_aneurysm_model.py --backend JAX --duration-seconds 0.5
python simulation_src/performance_testing_aneurysm_model.py --backend WARP --duration-seconds 0.5
```

*The comprehensive testing script (`run_all_performance_tests.py`) may introduce slight overhead due to subprocess isolation and thermal management. For most accurate MLUPS measurements, use the individual testing script directly.*

### 📊 **Performance Testing Output Structure**
```
results/performance_tests/
├── aneurysm_performance_jax/
│   └── parameters/
│       └── performance_test_results_jax.json
├── aneurysm_performance_warp/
│   └── parameters/
│       └── performance_test_results_warp.json
└── comprehensive_performance_comparison.json
```

### 🎯 **Performance Testing Use Cases**

#### **System Benchmarking**:
```bash
# Validate your system's LBM performance
python simulation_src/run_all_performance_tests.py --duration-seconds 3.0
```

#### **Hardware Optimization**:
```bash
# Compare CPU vs GPU efficiency
python simulation_src/run_all_performance_tests.py --verbose
```

#### **Development Testing**:
```bash
# Quick performance validation during development
python simulation_src/run_all_performance_tests.py --duration-seconds 0.5 --skip-cooldown
```

### Expected Complete Batch Performance
- **Total Time**: ~60-70 minutes (18 configurations)
- **Standard Pipes**: ~36 minutes (12 × 3 min)
- **Long Pipes**: ~12 minutes (3 × 4 min)
- **Aneurysms**: ~15 minutes (3 × 5 min)

## Troubleshooting

### Common Issues

#### **GPU Performance Issues**
```bash
# Check GPU availability
nvidia-smi
docker run --gpus all nvidia/cuda:11.0-base nvidia-smi
```

**Low MLUPS Performance**:
- Expected: 8,000+ MLUPS on GPU, 100-500 on CPU
- Solutions: Use `--long-pipe`, ensure GPU backend, verify CUDA setup

#### **Memory Issues**
```bash
# Monitor GPU memory
nvidia-smi -l 1
```
- Use CPU backend: `export XLA_FLAGS=--xla_force_host_platform_device_count=1`
- Increase system RAM (16GB+ recommended)

#### **Configuration Issues**
```bash
# Always validate first
python simulation_src/run_all_sim_configs.py --dry-run

# Check logs
ls results/logs/
tail results/logs/pipe_*_latest.log
```

#### **Docker Issues**
```bash
# Clean rebuild
docker system prune -a
docker build --no-cache -t dissertation .

# Fix permissions
sudo chown -R $USER:$USER results/
```

### Quick Debug Commands
```bash
# System check
python -c "import jax; print('JAX devices:', jax.devices())"

# Test minimal simulation
python simulation_src/pipe_run.py \
  --boundary-condition standard \
  --collision-operator standard \
  --spatial-profile uniform
```

## Technical Documentation

### 📊 **Configuration Matrix (18 Total)**
- **Standard Pipe**: 12 combinations (2 BC × 2 CO × 3 SP)
- **Long Pipe**: 3 combinations (1 BC × 1 CO × 3 SP)  
- **Aneurysm**: 3 combinations (1 BC × 1 CO × 3 SP)

### 🔬 **Research Configurations**
| Study Type | Command | Use Case |
|------------|---------|----------|
| **Blood Flow** | `--boundary-condition time-dependent --collision-operator non-newtonian --spatial-profile blunted_paraboloid` | Cardiovascular modeling |
| **Performance** | `--long-pipe` | MLUPS benchmarking |
| **Validation** | `--boundary-condition standard --collision-operator standard --spatial-profile poiseuille` | Fluid mechanics |

### 📁 **Key Files**
- `simulation_src/run_all_sim_configs.py`: Main batch execution
- `simulation_src/examples/run_all_sim_configs_usage.py`: Comprehensive usage guide
- `simulation_src/performance_testing_aneurysm_model.py`: Dedicated performance testing framework
- `simulation_src/run_all_performance_tests.py`: Comprehensive backend comparison tool
- `visualisation_src/vtk_visualization.ipynb`: Interactive analysis
- `simulation_src/WSS_Fluctuation_Analysis_and_Profile_Trade-offs.txt`: WSS artifact analysis and recommendations
- `SPATIAL_PROFILES_SUMMARY.md`: Detailed spatial profiles documentation
- `MLUPS_FIXES_SUMMARY.md`: Performance optimization guide

### 🎯 **Recent Improvements**
- **Dedicated Performance Testing**: Complete MLUPS benchmarking framework with JAX vs WARP comparison
- **WSS Artifact Analysis**: Comprehensive investigation and documentation of wall shear stress fluctuations
- **Default Multi-Configuration**: Runs 18 configurations by default
- **Intelligent Naming**: Pipe type distinction prevents result overwrites  
- **Performance Scaling**: Domain-size dependent MLUPS optimization (2-3× improvement)
- **Enhanced Spatial Profiles**: Uniform, Poiseuille, and Blunted Paraboloid implementations with restored u_max scaling
- **Professional Output**: Complete VTK integration with JSON parameter files

---

*This framework provides a production-ready CFD simulation system optimized for blood flow analysis, high-performance computing, and systematic parameter studies.*
