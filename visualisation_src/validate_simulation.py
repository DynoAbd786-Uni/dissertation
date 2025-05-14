#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simulation Validation Script

This script validates the LBM simulation results by comparing them with analytical models.
It loads processed data from the .npz file, extracts velocity profiles at specific time points,
and calculates correlation metrics between simulated and theoretical values.

Validation tasks:
1. Load velocity data from processed .npz files
2. Extract cross-sectional velocity profiles at specific time points (t = 0.1, 0.4, 0.7)
3. Compare with analytical inflow profiles
4. Calculate waveform correlation and normalized RMSE
5. Generate visualizations for validation
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import scipy.stats
from scipy.interpolate import interp1d
import scipy.signal
from matplotlib.colors import Normalize
import matplotlib.cm as cm

# Set up paths
base_dir = Path('/home/abdua786/code/uni/3/dissertation/dissertation')
npz_file_path = base_dir / 'aneurysm_simulation_results' / 'processed_data' / 'aneurysm_vtk_data.npz'
output_dir = base_dir / 'results' / 'validation'
os.makedirs(output_dir, exist_ok=True)

# Function to convert simulation time steps to simulation time (0-1)
def timestep_to_time(timestep, total_timesteps):
    """Convert a timestep to normalized time (0-1)"""
    return timestep / total_timesteps

# Function to find the nearest timestep for a given time
def find_nearest_timestep(time_points, target_time):
    """Find the index of the timestep closest to the target time"""
    return np.argmin(np.abs(time_points - target_time))

# Function to calculate normalized RMSE
def normalized_rmse(y_true, y_pred):
    """Calculate normalized Root Mean Square Error"""
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    # Normalize by the range of true values
    norm_factor = np.max(y_true) - np.min(y_true)
    if norm_factor == 0:
        return float('inf')  # Avoid division by zero
    return rmse / norm_factor

# Function to extract cross-sectional profile
def extract_cross_section(velocity_field, dimensions, position='middle'):
    """
    Extract a cross-sectional profile of the velocity field
    
    Args:
        velocity_field: 2D velocity field (reshaped from 1D)
        dimensions: (ny, nx) shape of the domain
        position: where to take the cross-section ('inlet', 'middle', 'outlet')
    
    Returns:
        1D array of velocity values across the chosen cross-section
    """
    ny, nx = dimensions
    
    if position == 'inlet':
        idx = 5  # Near inlet (with some buffer from boundary)
    elif position == 'outlet':
        idx = nx - 5  # Near outlet (with some buffer from boundary)
    else:  # middle
        idx = nx // 2  # Middle of the domain
        
    # Extract cross-section (vertical profile at position idx)
    cross_section = velocity_field[:, idx]
    
    return cross_section

# Function to calculate theoretical Poiseuille profile
def poiseuille_profile(y_positions, height, max_velocity=1.0):
    """
    Calculate theoretical Poiseuille velocity profile
    
    Args:
        y_positions: y-coordinates across the channel
        height: height of the channel
        max_velocity: maximum velocity at the center
    
    Returns:
        Velocity profile following parabolic distribution
    """
    # Normalize y-positions to [-1, 1] range
    y_norm = 2 * (y_positions / height) - 1
    
    # Calculate parabolic profile: u(y) = u_max * (1 - (y/R)²)
    profile = max_velocity * (1 - y_norm**2)
    
    return profile

# Function to calculate theoretical Womersley profile
def womersley_profile(y_positions, height, time, period=1.0, womersley_number=3.0, amplitude=1.0):
    """
    Calculate theoretical Womersley velocity profile for pulsatile flow
    
    This is a simplified implementation that captures the main characteristics
    of Womersley flow without the full complex analytical solution.
    
    Args:
        y_positions: y-coordinates across the channel
        height: height of the channel
        time: current time in the cycle (0-1)
        period: period of oscillation
        womersley_number: Womersley number (dimensionless frequency parameter)
        amplitude: amplitude of oscillation
    
    Returns:
        Velocity profile with time-dependent phase shift
    """
    # Normalize y-positions to [-1, 1] range
    y_norm = 2 * (y_positions / height) - 1
    
    # Base parabolic profile
    base_profile = 1 - y_norm**2
    
    # Time-dependent factor (simplified Womersley effect)
    # Higher Womersley numbers cause phase lag between pressure and flow
    phase_shift = 0.2 * womersley_number * time
    time_factor = np.sin(2 * np.pi * time - phase_shift)
    
    # Combine to create time-varying profile with flatter shape for higher Womersley numbers
    flatten_factor = 1.0 / (1.0 + 0.1 * womersley_number**2)
    profile = amplitude * time_factor * (base_profile ** flatten_factor)
    
    return profile

# Function to generate validation visualizations
def generate_validation_plots(time_points, velocity_data, dimensions, analytical_model='poiseuille'):
    """
    Generate validation plots comparing simulated and analytical velocity profiles
    
    Args:
        time_points: array of normalized time points (0-1)
        velocity_data: dictionary of velocity data by time
        dimensions: (ny, nx) shape of the domain
        analytical_model: which analytical model to use ('poiseuille' or 'womersley')
    """
    # Times to visualize
    target_times = [0.1, 0.4, 0.7]
    
    # Get indices of nearest time points
    target_indices = [find_nearest_timestep(time_points, t) for t in target_times]
    
    # Set up the figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Create arrays to store data for correlation analysis
    all_simulated = []
    all_analytical = []
    
    # Loop through target times
    for i, (time_idx, target_time) in enumerate(zip(target_indices, target_times)):
        # Get the actual time from the array
        actual_time = time_points[time_idx]
        
        # Extract velocity for this time
        velocity_field = velocity_data[time_idx]
        
        # Reshape to 2D
        ny, nx = dimensions
        velocity_field_2d = velocity_field.reshape(ny, nx)
        
        # Extract cross-section at middle of the domain
        y_positions = np.arange(ny)
        simulated_profile = extract_cross_section(velocity_field_2d, (ny, nx), 'middle')
        
        # Calculate analytical profile
        if analytical_model == 'poiseuille':
            # Scale max velocity based on time (for simple time variation)
            max_vel = 1.0 + 0.5 * np.sin(2 * np.pi * actual_time)
            analytical_profile = poiseuille_profile(y_positions, ny, max_vel)
        else:  # womersley
            analytical_profile = womersley_profile(y_positions, ny, actual_time)
        
        # Store for correlation analysis
        all_simulated.append(simulated_profile)
        all_analytical.append(analytical_profile)
        
        # Plot comparison
        ax = axes[i]
        ax.plot(simulated_profile, y_positions, 'b-', linewidth=2, label='Simulated')
        ax.plot(analytical_profile, y_positions, 'r--', linewidth=2, label='Analytical')
        ax.set_title(f'Time = {actual_time:.3f}')
        ax.set_xlabel('Velocity Magnitude')
        ax.set_ylabel('Channel Height')
        ax.legend()
        ax.grid(True)
        
        # Calculate metrics for this time point
        correlation, _ = scipy.stats.pearsonr(simulated_profile, analytical_profile)
        nrmse = normalized_rmse(analytical_profile, simulated_profile)
        
        # Add metrics to the plot
        ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}\nnRMSE: {nrmse:.3f}', 
                transform=ax.transAxes, fontsize=10, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / f'validation_profiles_{analytical_model}.png', dpi=300)
    
    # Calculate overall correlation and RMSE
    all_simulated = np.concatenate(all_simulated)
    all_analytical = np.concatenate(all_analytical)
    
    overall_correlation, _ = scipy.stats.pearsonr(all_simulated, all_analytical)
    overall_nrmse = normalized_rmse(all_analytical, all_simulated)
    
    # Create scatter plot of all simulated vs analytical values
    plt.figure(figsize=(8, 8))
    plt.scatter(all_analytical, all_simulated, alpha=0.5, s=10)
    
    # Add identity line (perfect correlation)
    min_val = min(np.min(all_analytical), np.min(all_simulated))
    max_val = max(np.max(all_analytical), np.max(all_simulated))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.title(f'Correlation Analysis\nPearson r: {overall_correlation:.3f}, nRMSE: {overall_nrmse:.3f}')
    plt.xlabel('Analytical Velocity')
    plt.ylabel('Simulated Velocity')
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(output_dir / f'correlation_analysis_{analytical_model}.png', dpi=300)
    
    return overall_correlation, overall_nrmse

# Function to create 2D fluid visualization at specific time points
def visualize_flow_field(velocity_magnitude, velocity_x, velocity_y, dimensions, time_point, output_file):
    """
    Create a visualization of the flow field with streamlines
    
    Args:
        velocity_magnitude: 1D array of velocity magnitudes
        velocity_x: 1D array of x-components of velocity
        velocity_y: 1D array of y-components of velocity
        dimensions: (ny, nx) shape of the domain
        time_point: current time point for the title
        output_file: path to save the visualization
    """
    ny, nx = dimensions
    
    # Reshape to 2D
    vel_mag_2d = velocity_magnitude.reshape(ny, nx)
    vel_x_2d = velocity_x.reshape(ny, nx)
    vel_y_2d = velocity_y.reshape(ny, nx)
    
    # Create a grid for streamplot
    y, x = np.mgrid[0:ny, 0:nx]
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot velocity magnitude as a color map
    magnitude_plot = ax.imshow(vel_mag_2d, cmap='viridis', origin='lower')
    plt.colorbar(magnitude_plot, ax=ax, label='Velocity Magnitude')
    
    # Add streamlines to show flow direction
    # Subsample the grid to avoid cluttered streamlines
    step = max(1, min(nx, ny) // 20)  # Adjust based on grid size
    ax.streamplot(x[::step, ::step], y[::step, ::step], 
                 vel_x_2d[::step, ::step], vel_y_2d[::step, ::step], 
                 color='white', linewidth=1.5, density=1, arrowsize=1.5)
    
    ax.set_title(f'Fluid Flow at t = {time_point:.2f}')
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

# Function to generate waveform comparison plot
def generate_waveform_comparison(time_points, velocity_data, dimensions, analytical_models=None):
    """
    Generate a waveform comparison plot showing max velocity over time
    
    Args:
        time_points: array of normalized time points (0-1)
        velocity_data: 3D array of velocity data (frames x height x width)
        dimensions: (ny, nx) shape of the domain
        analytical_models: dict of model names -> functions to compute analytical values
    """
    if analytical_models is None:
        analytical_models = {
            'Poiseuille': lambda t: 1.0 + 0.5 * np.sin(2 * np.pi * t),
            'Womersley': lambda t: np.sin(2 * np.pi * t - 0.2 * 3.0 * t)
        }
    
    # Extract the max velocity at each time point from the simulation data
    ny, nx = dimensions
    max_velocities = []
    centerline_velocities = []
    cross_section_means = []
    
    print("Calculating velocity metrics for each frame...")
    for i, time in enumerate(time_points):
        # Get the velocity data for this time point
        vel_field = velocity_data[i]
        
        # Reshape to 2D
        vel_field_2d = vel_field.reshape(ny, nx)
        
        # Calculate the maximum velocity in the field
        max_vel = np.max(vel_field_2d)
        max_velocities.append(max_vel)
        
        # Extract the centerline velocity (middle of the channel)
        centerline_idx = ny // 2
        centerline_vel = np.max(vel_field_2d[centerline_idx, :])
        centerline_velocities.append(centerline_vel)
        
        # Get cross-section at the middle of the domain
        cross_section = extract_cross_section(vel_field_2d, (ny, nx), 'middle')
        cross_section_means.append(np.mean(cross_section))
    
    # Create the figure
    plt.figure(figsize=(12, 8))
    
    # Plot the max velocity over time
    plt.plot(time_points, max_velocities, 'b-', linewidth=2, label='Max Velocity')
    plt.plot(time_points, centerline_velocities, 'g-', linewidth=2, label='Centerline Velocity')
    plt.plot(time_points, cross_section_means, 'c-', linewidth=1.5, label='Cross-section Mean')
    
    # Plot the analytical models
    for name, model_func in analytical_models.items():
        # Scale the analytical model to match the range of the simulation data
        analytical_values = model_func(time_points)
        # Normalize and scale
        analytical_values = analytical_values - np.min(analytical_values)
        analytical_values = analytical_values / np.max(analytical_values)
        analytical_values = analytical_values * (np.max(max_velocities) - np.min(max_velocities)) + np.min(max_velocities)
        
        plt.plot(time_points, analytical_values, '--', linewidth=1.5, label=f'{name} Model')
        
        # Calculate correlation with max velocity
        correlation, _ = scipy.stats.pearsonr(max_velocities, analytical_values)
        nrmse = normalized_rmse(analytical_values, max_velocities)
        print(f"{name} model correlation with max velocity: {correlation:.4f}, nRMSE: {nrmse:.4f}")
    
    # Calculate FFT of the velocity waveform to find dominant frequencies
    if len(max_velocities) > 10:  # Only if we have enough data points
        plt.figure(figsize=(12, 6))
        # Compute FFT
        fft_values = np.fft.rfft(max_velocities)
        fft_freqs = np.fft.rfftfreq(len(max_velocities))
        
        # Plot magnitude spectrum
        plt.plot(fft_freqs, np.abs(fft_values))
        plt.title('Frequency Spectrum of Velocity Waveform')
        plt.xlabel('Frequency (normalized)')
        plt.ylabel('Magnitude')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_dir / 'velocity_frequency_spectrum.png', dpi=300)
    
    # Setup the main waveform plot
    plt.figure(figsize=(12, 8))
    plt.plot(time_points, max_velocities, 'b-', linewidth=2, label='Max Velocity')
    plt.plot(time_points, centerline_velocities, 'g-', linewidth=2, label='Centerline Velocity')
    
    # Add markers for the specific time points we visualized (t=0.1, 0.4, 0.7)
    target_times = [0.1, 0.4, 0.7]
    for target_time in target_times:
        time_idx = find_nearest_timestep(time_points, target_time)
        actual_time = time_points[time_idx]
        plt.axvline(x=actual_time, color='r', linestyle='--', alpha=0.5)
        plt.plot(actual_time, max_velocities[time_idx], 'ro', markersize=8)
        plt.text(actual_time, max_velocities[time_idx] * 1.05, f't={actual_time:.2f}',
                 horizontalalignment='center')
    
    plt.title('Velocity Waveform Comparison')
    plt.xlabel('Normalized Time')
    plt.ylabel('Velocity Magnitude')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / 'velocity_waveform_comparison.png', dpi=300)
    
    return max_velocities, centerline_velocities, cross_section_means

# Main function to execute validation
def main():
    # Check if the NPZ file exists
    if not npz_file_path.exists():
        print(f"Error: NPZ file not found at {npz_file_path}")
        print("Please run the VTK processing script first to generate the processed data.")
        return
    
    print(f"Loading data from {npz_file_path}...")
    
    # Load the processed data
    npz_data = np.load(npz_file_path)
    
    # Extract relevant fields
    print("Available fields in the dataset:")
    for field in npz_data.files:
        print(f"- {field}: {npz_data[field].shape}")
    
    # Extract dimensions
    if 'dimensions' in npz_data.files:
        dimensions = npz_data['dimensions']
        raw_dims = dimensions
        # Actual dimensions for reshaping (subtract 1 as in the processing)
        dims = tuple(d-1 for d in raw_dims if d > 1)
        ny, nx = dims
        print(f"Domain dimensions: {ny} x {nx}")
    else:
        print("Warning: dimensions not found in the dataset. Using default values.")
        ny, nx = 100, 400  # Default values
    
    # Extract frame numbers and indices
    frame_numbers = npz_data['frame_numbers']
    frame_indices = npz_data['frame_indices']
    
    # Calculate normalized time points (0-1)
    total_frames = len(frame_numbers)
    time_points = np.array([timestep_to_time(i, total_frames-1) for i in range(total_frames)])
    
    # Extract velocity data
    if 'u_magnitude' in npz_data.files:
        velocity_magnitude = npz_data['u_magnitude']
        print(f"Velocity magnitude shape: {velocity_magnitude.shape}")
    else:
        print("Warning: velocity magnitude not found. Checking for x and y components.")
        
        if 'u_x' in npz_data.files and 'u_y' in npz_data.files:
            u_x = npz_data['u_x']
            u_y = npz_data['u_y']
            print(f"Computing velocity magnitude from components (shapes: {u_x.shape}, {u_y.shape})")
            
            # Calculate magnitude from components
            velocity_magnitude = np.sqrt(u_x**2 + u_y**2)
        else:
            print("Error: No velocity data found in the dataset.")
            return
    
    # Extract velocity components if available
    u_x = npz_data.get('u_x', None)
    u_y = npz_data.get('u_y', None)
    
    # Target times for visualization (t = 0.1, 0.4, 0.7)
    target_times = [0.1, 0.4, 0.7]
    
    # Generate flow field visualizations for target times
    for time_val in target_times:
        time_idx = find_nearest_timestep(time_points, time_val)
        actual_time = time_points[time_idx]
        
        print(f"Generating flow visualization for t = {actual_time:.2f} (requested: {time_val})")
        
        # Get the velocity data for this time point
        vel_mag = velocity_magnitude[time_idx]
        
        if u_x is not None and u_y is not None:
            vel_x = u_x[time_idx]
            vel_y = u_y[time_idx]
        else:
            # If components not available, create mock data for visualization
            print("Warning: Using mock velocity components for visualization")
            vel_x = np.zeros_like(vel_mag)
            vel_y = np.zeros_like(vel_mag)
        
        # Create output file path
        output_file = output_dir / f'flow_field_t{actual_time:.2f}.png'
        
        # Visualize flow field
        visualize_flow_field(vel_mag, vel_x, vel_y, (ny, nx), actual_time, output_file)
    
    # Generate validation plots comparing with analytical models
    print("Generating validation plots comparing with Poiseuille flow...")
    poiseuille_corr, poiseuille_nrmse = generate_validation_plots(
        time_points, velocity_magnitude, (ny, nx), 'poiseuille')
    
    print("Generating validation plots comparing with Womersley flow...")
    womersley_corr, womersley_nrmse = generate_validation_plots(
        time_points, velocity_magnitude, (ny, nx), 'womersley')
    
    # Output validation metrics
    print("\nValidation Metrics:")
    print(f"Poiseuille model - Correlation: {poiseuille_corr:.4f}, nRMSE: {poiseuille_nrmse:.4f}")
    print(f"Womersley model - Correlation: {womersley_corr:.4f}, nRMSE: {womersley_nrmse:.4f}")
    
    # Generate waveform comparison across all frames
    print("\nGenerating velocity waveform comparison across all frames...")
    max_velocities, centerline_velocities, cross_section_means = generate_waveform_comparison(
        time_points, velocity_magnitude, (ny, nx))
    
    # Determine which model fits better
    if poiseuille_corr > womersley_corr and poiseuille_nrmse < womersley_nrmse:
        best_model = "Poiseuille"
    elif womersley_corr > poiseuille_corr and womersley_nrmse < poiseuille_nrmse:
        best_model = "Womersley"
    else:
        # If metrics are mixed, check correlation as primary indicator
        best_model = "Poiseuille" if poiseuille_corr > womersley_corr else "Womersley"
    
    print(f"\nBest matching analytical model: {best_model}")
    
    # Generate summary results
    with open(output_dir / 'validation_summary.txt', 'w') as f:
        f.write("Simulation Validation Summary\n")
        f.write("============================\n\n")
        f.write(f"Domain dimensions: {ny} x {nx}\n")
        f.write(f"Total frames analyzed: {total_frames}\n\n")
        
        f.write("Validation Metrics:\n")
        f.write(f"- Poiseuille model - Correlation: {poiseuille_corr:.4f}, nRMSE: {poiseuille_nrmse:.4f}\n")
        f.write(f"- Womersley model - Correlation: {womersley_corr:.4f}, nRMSE: {womersley_nrmse:.4f}\n\n")
        
        f.write(f"Best matching analytical model: {best_model}\n\n")
        
        f.write("Visualizations generated:\n")
        for time_val in target_times:
            time_idx = find_nearest_timestep(time_points, time_val)
            actual_time = time_points[time_idx]
            f.write(f"- Flow field at t = {actual_time:.2f}\n")
        
        f.write("- Velocity profile comparison with Poiseuille model\n")
        f.write("- Velocity profile comparison with Womersley model\n")
        f.write("- Correlation analysis plots\n")
    
    print(f"\nValidation complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()