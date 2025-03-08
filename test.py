import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import pandas as pd
from typing import Dict, List, Tuple
import warp as wp
import jax.numpy as jnp
from xlb.compute_backend import ComputeBackend

def fit_hybrid_profile(csv_filepath: str, x_column=None, y_column=None, 
                     n_harmonics=6, poly_degree=5, fourier_fraction=0.45, 
                     overlap_fraction=0.10, plot_result=True):
    """
    Create a hybrid velocity profile using:
    - Fourier series for the first portion of the cardiac cycle
    - Polynomial fit for the remainder with smooth cycle transitions
    
    Parameters:
        csv_filepath: Path to the CSV file
        x_column, y_column: Column names
        n_harmonics: Number of harmonics for Fourier
        poly_degree: Degree of polynomial for second half
        fourier_fraction: Fraction of cycle to use Fourier (0.45 = 45%)
        overlap_fraction: Fraction of cycle to use for boundary smoothing
        plot_result: Whether to plot the results
    """
    # Read data
    data = pd.read_csv(csv_filepath)
    
    # Handle column selection
    if x_column is None:
        x_column = data.columns[0]
    if y_column is None:
        # Find first non-time column
        y_column = [col for col in data.columns if col != x_column][0]
    
    # Extract clean data
    t = data[x_column].values.astype(np.float32)
    v = data[y_column].values.astype(np.float32)
    valid_mask = ~(np.isnan(t) | np.isnan(v))
    t_clean = t[valid_mask]
    v_clean = v[valid_mask]
    
    # Get period and calculate transition point
    period = float(t_clean[-1] - t_clean[0])
    t_min = float(t_clean[0])
    t_max = float(t_clean[-1])
    
    # Use specified fraction for transition point
    t_transition = t_min + fourier_fraction * period
    
    # Split data at transition point
    first_part_mask = t_clean <= t_transition
    second_part_mask = t_clean > t_transition
    
    t_first = t_clean[first_part_mask]
    v_first = v_clean[first_part_mask]
    t_second = t_clean[second_part_mask]
    v_second = v_clean[second_part_mask]
    
    # PART 1: Fit Fourier series to first part
    # Define Fourier function for first part
    def fourier_func(t, *params):
        """Fourier series with n harmonics"""
        a0 = params[0]
        result = a0
        for i in range(1, len(params)//2 + 1):
            a = params[2*i-1]
            b = params[2*i]
            # Scale the period to match the first part's time range
            scaled_period = period * fourier_fraction
            # Normalize t to be relative to t_min
            t_norm = t - t_min
            result += a * np.cos(2*np.pi*i*t_norm/scaled_period) + b * np.sin(2*np.pi*i*t_norm/scaled_period)
        return result
    
    # Fit the Fourier function to first part
    initial_params = [np.mean(v_first)] + [0.0] * (2*n_harmonics)
    
    try:
        fourier_params, _ = optimize.curve_fit(
            fourier_func, t_first, v_first, p0=initial_params
        )
        
        # Calculate R-squared for Fourier fit
        residuals_fourier = v_first - fourier_func(t_first, *fourier_params)
        ss_res_fourier = np.sum(residuals_fourier**2)
        ss_tot_fourier = np.sum((v_first - np.mean(v_first))**2)
        r_squared_fourier = 1 - (ss_res_fourier / ss_tot_fourier)
        
    except Exception as e:
        print(f"Fourier fitting error: {e}")
        return None
    
    # Calculate values at key points for continuity
    v_transition = fourier_func(t_transition, *fourier_params)  # Value at transition point
    v_start = fourier_func(t_min, *fourier_params)             # Value at start of cycle
    
    # PART 2: Fit polynomial to second part WITH BOUNDARY SHARING
    # Calculate values at key points for continuity
    v_transition = fourier_func(t_transition, *fourier_params)  # Value at transition point
    
    # Get early cycle points for cycle boundary smoothing
    # Take the first X% of points from the original data (start of cycle)
    overlap_point_count = max(3, int(len(t_clean) * overlap_fraction))
    early_points_t = t_clean[:overlap_point_count].copy()
    early_points_v = v_clean[:overlap_point_count].copy()
    
    # Shift these points to represent the next cycle (t + period)
    early_points_t_next_cycle = early_points_t + period
    
    # Combine original second part with shifted early points
    t_second_extended = np.concatenate([t_second, early_points_t_next_cycle])
    v_second_extended = np.concatenate([v_second, early_points_v])
    
    # Sort by time to ensure correct order
    sort_idx = np.argsort(t_second_extended)
    t_second_extended = t_second_extended[sort_idx]
    v_second_extended = v_second_extended[sort_idx]
    
    # Normalize time for polynomial fit (0 at transition, 1 at end of cycle)
    # Note: Some points will have normalized times > 1 (the early points from next cycle)
    t_poly_norm = (t_second_extended - t_transition) / (t_max - t_transition)
    
    # Define polynomial function with constraint at transition point
    def poly_func(t_norm, *params):
        """Polynomial function with constraint that poly(0) = v_transition"""
        # First coefficient is fixed to ensure continuity
        result = v_transition
        # Add polynomial terms
        for i, coef in enumerate(params):
            result += coef * (t_norm ** (i+1))
        return result
    
    # Fit polynomial to extended second part (includes boundary points)
    initial_poly_params = [0.0] * poly_degree
    
    try:
        poly_params, _ = optimize.curve_fit(
            poly_func, t_poly_norm, v_second_extended, p0=initial_poly_params
        )
        
        # Calculate R-squared for polynomial fit (on original points only)
        t_orig_norm = (t_second - t_transition) / (t_max - t_transition)
        v_poly_fit = poly_func(t_orig_norm, *poly_params)
        residuals_poly = v_second - v_poly_fit
        ss_res_poly = np.sum(residuals_poly**2)
        ss_tot_poly = np.sum((v_second - np.mean(v_second))**2)
        r_squared_poly = 1 - (ss_res_poly / ss_tot_poly)
        
    except Exception as e:
        print(f"Polynomial fitting error: {e}")
        return None
    
    # Create combined hybrid function
    def hybrid_func(t):
        """Combined function using Fourier for first part, polynomial for second part"""
        # Make time cyclic
        t_cycle = t_min + ((t - t_min) % period)
        
        # Determine which part of the cycle we're in
        if t_cycle <= t_transition:
            # First part - use Fourier
            return fourier_func(t_cycle, *fourier_params)
        else:
            # Second part - use polynomial with normalized time
            t_norm = (t_cycle - t_transition) / (t_max - t_transition)
            return poly_func(t_norm, *poly_params)
    
    # Calculate overall R-squared for hybrid model
    v_hybrid_at_points = np.array([hybrid_func(t) for t in t_clean])
    residuals_hybrid = v_clean - v_hybrid_at_points
    ss_res_hybrid = np.sum(residuals_hybrid**2)
    ss_tot_hybrid = np.sum((v_clean - np.mean(v_clean))**2)
    r_squared_hybrid = 1 - (ss_res_hybrid / ss_tot_hybrid)
    
    # Plot results if requested
    if plot_result:
        # Create high-resolution time points for plotting
        t_fine = np.linspace(t_min, t_max, 1000)
        v_hybrid = np.array([hybrid_func(t) for t in t_fine])
        
        # Plot 1: Original data with hybrid fit
        plt.figure(figsize=(14, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(t_clean, v_clean, 'o', alpha=0.5, label=f'Original {y_column} data')
        plt.plot(t_fine, v_hybrid, '-', linewidth=2, label=f'Hybrid model (R²={r_squared_hybrid:.4f})')
        
        # Highlight the transition point
        plt.axvline(x=t_transition, color='g', linestyle='--', alpha=0.5, 
                  label=f'Transition point ({fourier_fraction*100:.1f}%)')
        
        # Add labels and legend
        plt.xlabel(f'{x_column}')
        plt.ylabel(f'{y_column}')
        plt.title(f'Hybrid Model: Fourier ({n_harmonics} harmonics) + Polynomial (degree {poly_degree})')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot 2: Residuals
        plt.subplot(2, 1, 2)
        plt.plot(t_clean, residuals_hybrid, 'o-', alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.axvline(x=t_transition, color='g', linestyle='--', alpha=0.5)
        plt.xlabel(f'{x_column}')
        plt.ylabel('Residuals')
        plt.title(f'Hybrid Model Residuals (R²={r_squared_hybrid:.4f})')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{y_column}_hybrid_fit_{fourier_fraction*100:.1f}.png", dpi=300)
        plt.show()
    
    # Return parameters and R-squared for comparison
    return {
        'fourier_params': fourier_params.tolist(),
        'poly_params': poly_params.tolist(),
        't_min': t_min,
        't_transition': t_transition,
        't_max': t_max,
        'period': period,
        'v_transition': float(v_transition),
        'fourier_fraction': fourier_fraction,
        'r_squared': r_squared_hybrid,
        'r_squared_fourier': r_squared_fourier,
        'r_squared_poly': r_squared_poly
    }

def optimize_transition_point(csv_filepath: str, x_column=None, y_column=None,
                           n_harmonics=6, poly_degree=5,
                           min_fraction=0.450, max_fraction=0.500, step=0.001,
                           plot_comparison=True):
    """
    Find the optimal transition point between Fourier and polynomial fits.
    
    Parameters:
        csv_filepath: Path to the CSV file
        x_column, y_column: Column names to use
        n_harmonics: Number of harmonics for Fourier part
        poly_degree: Degree of polynomial for second part
        min_fraction, max_fraction: Range to search for optimal transition
        step: Step size for search
        plot_comparison: Whether to plot comparison of different transition points
    
    Returns:
        Dictionary with optimal parameters
    """
    print(f"\nOptimizing transition point for {y_column} between {min_fraction*100:.1f}% and {max_fraction*100:.1f}%")
    print(f"Using step size of {step*100:.1f}% ({int((max_fraction-min_fraction)/step + 1)} points to evaluate)")
    
    # Track results
    results = []
    best_r_squared = -1.0
    best_params = None
    
    # Try different transition points
    fractions = np.arange(min_fraction, max_fraction + step/2, step)
    for fraction in fractions:
        print(f"Testing transition at {fraction*100:.1f}%...")
        
        # Fit hybrid model with this transition point
        params = fit_hybrid_profile(
            csv_filepath=csv_filepath,
            x_column=x_column,
            y_column=y_column,
            n_harmonics=n_harmonics,
            poly_degree=poly_degree,
            fourier_fraction=fraction,
            plot_result=False  # Don't plot individual fits during sweep
        )
        
        if params is not None:
            r_squared = params['r_squared']
            results.append((fraction, r_squared))
            print(f"  R²: {r_squared:.6f}")
            
            # Update best if improved
            if r_squared > best_r_squared:
                best_r_squared = r_squared
                best_params = params
    
    # Convert results to arrays for plotting
    fractions_array = np.array([f for f, _ in results]) * 100  # Convert to percentage
    r_squared_array = np.array([r for _, r in results])
    
    # Plot comparison of R-squared values
    if plot_comparison and results:
        plt.figure(figsize=(12, 6))
        plt.plot(fractions_array, r_squared_array, 'o-', linewidth=2)
        
        # Mark the best point
        best_idx = np.argmax(r_squared_array)
        plt.plot(fractions_array[best_idx], r_squared_array[best_idx], 'ro', 
               markersize=10, label=f'Best: {fractions_array[best_idx]:.1f}% (R²={r_squared_array[best_idx]:.6f})')
        
        plt.xlabel('Fourier/Polynomial Transition Point (%)')
        plt.ylabel('R-squared')
        plt.title(f'R-squared vs. Transition Point for {y_column}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{y_column}_transition_optimization.png", dpi=300)
        plt.show()
    
    # Report best results
    if best_params:
        print(f"\nBest transition point: {best_params['fourier_fraction']*100:.1f}%")
        print(f"Best R-squared: {best_params['r_squared']:.6f}")
        print(f"  Fourier part R-squared: {best_params['r_squared_fourier']:.6f}")
        print(f"  Polynomial part R-squared: {best_params['r_squared_poly']:.6f}")
        
        # Refit with best parameters and visualize
        final_params = fit_hybrid_profile(
            csv_filepath=csv_filepath,
            x_column=x_column,
            y_column=y_column,
            n_harmonics=n_harmonics,
            poly_degree=poly_degree,
            fourier_fraction=best_params['fourier_fraction'],
            plot_result=True  # Plot the best fit
        )
        
        return final_params
    else:
        print("No valid results found.")
        return None

def optimize_transition_point_wide(csv_filepath: str, x_column=None, y_column=None,
                           n_harmonics=6, poly_degree=5,
                           coarse_min=0.30, coarse_max=0.70, coarse_step=0.01,
                           fine_window=0.05, fine_step=0.0001,
                           plot_comparison=True):
    """
    Find the optimal transition point using a two-stage approach:
    1. Coarse search across a wide range
    2. Fine search around the best region
    
    Parameters:
        csv_filepath: Path to the CSV file
        x_column, y_column: Column names to use
        n_harmonics: Number of harmonics for Fourier part
        poly_degree: Degree of polynomial for second part
        coarse_min, coarse_max: Range for initial coarse search (0.3 = 30%)
        coarse_step: Step size for coarse search (0.01 = 1%)
        fine_window: Window size around best coarse point for fine search (0.05 = ±5%)
        fine_step: Step size for fine search (0.0001 = 0.01%)
        plot_comparison: Whether to plot comparison of different transition points
    
    Returns:
        Dictionary with optimal parameters
    """
    print(f"\nSTAGE 1: Coarse optimization for {y_column}")
    print(f"Searching between {coarse_min*100:.1f}% and {coarse_max*100:.1f}% with {coarse_step*100:.1f}% steps")
    
    # Track results for coarse search
    coarse_results = []
    best_coarse_r_squared = -1.0
    best_coarse_fraction = None
    
    # Try different transition points - coarse search
    coarse_fractions = np.arange(coarse_min, coarse_max + coarse_step/2, coarse_step)
    for fraction in coarse_fractions:
        print(f"Testing transition at {fraction*100:.1f}%...")
        
        # Fit hybrid model with this transition point
        params = fit_hybrid_profile(
            csv_filepath=csv_filepath,
            x_column=x_column,
            y_column=y_column,
            n_harmonics=n_harmonics,
            poly_degree=poly_degree,
            fourier_fraction=fraction,
            plot_result=False  # Don't plot individual fits during sweep
        )
        
        if params is not None:
            r_squared = params['r_squared']
            coarse_results.append((fraction, r_squared))
            print(f"  R²: {r_squared:.6f}")
            
            # Update best if improved
            if r_squared > best_coarse_r_squared:
                best_coarse_r_squared = r_squared
                best_coarse_fraction = fraction
    
    # Plot coarse search results
    if plot_comparison and coarse_results:
        fractions_array = np.array([f for f, _ in coarse_results]) * 100
        r_squared_array = np.array([r for _, r in coarse_results])
        
        plt.figure(figsize=(14, 6))
        plt.plot(fractions_array, r_squared_array, 'o-', linewidth=2, label='Coarse search')
        
        # Mark the best point
        best_idx = np.argmax(r_squared_array)
        plt.plot(fractions_array[best_idx], r_squared_array[best_idx], 'ro', 
               markersize=10, label=f'Best coarse: {fractions_array[best_idx]:.1f}% (R²={r_squared_array[best_idx]:.6f})')
        
        plt.xlabel('Fourier/Polynomial Transition Point (%)')
        plt.ylabel('R-squared')
        plt.title(f'Coarse Search: R-squared vs. Transition Point for {y_column}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{y_column}_coarse_optimization.png", dpi=300)
        plt.show()
    
    # If no good results found in coarse search
    if best_coarse_fraction is None:
        print("No valid results found in coarse search.")
        return None
    
    # STAGE 2: Fine-grained search around the best coarse point
    print(f"\nSTAGE 2: Fine optimization for {y_column}")
    fine_min = max(coarse_min, best_coarse_fraction - fine_window/2)
    fine_max = min(coarse_max, best_coarse_fraction + fine_window/2)
    print(f"Searching between {fine_min*100:.2f}% and {fine_max*100:.2f}% with {fine_step*100:.2f}% steps")
    
    # Track results for fine search
    fine_results = []
    best_fine_r_squared = -1.0
    best_params = None
    
    # Try different transition points - fine search
    fine_fractions = np.arange(fine_min, fine_max + fine_step/2, fine_step)
    total_steps = len(fine_fractions)
    print(f"Evaluating {total_steps} points...")
    
    # Process in batches and show progress
    for i, fraction in enumerate(fine_fractions):
        if i % 100 == 0:  # Show progress every 100 steps
            print(f"Progress: {i}/{total_steps} points evaluated ({i/total_steps*100:.1f}%)")
        
        # Fit hybrid model with this transition point
        params = fit_hybrid_profile(
            csv_filepath=csv_filepath,
            x_column=x_column,
            y_column=y_column,
            n_harmonics=n_harmonics,
            poly_degree=poly_degree,
            fourier_fraction=fraction,
            plot_result=False  # Don't plot individual fits during sweep
        )
        
        if params is not None:
            r_squared = params['r_squared']
            fine_results.append((fraction, r_squared))
            
            # Update best if improved
            if r_squared > best_fine_r_squared:
                best_fine_r_squared = r_squared
                best_params = params
    
    # Plot fine search results
    if plot_comparison and fine_results:
        fractions_array = np.array([f for f, _ in fine_results]) * 100
        r_squared_array = np.array([r for _, r in fine_results])
        
        plt.figure(figsize=(14, 6))
        plt.plot(fractions_array, r_squared_array, '-', linewidth=1, alpha=0.7, label='Fine search')
        plt.plot(fractions_array, r_squared_array, '.', markersize=3, alpha=0.5)
        
        # Mark the best point
        best_idx = np.argmax(r_squared_array)
        plt.plot(fractions_array[best_idx], r_squared_array[best_idx], 'ro', 
               markersize=10, label=f'Optimal: {fractions_array[best_idx]:.2f}% (R²={r_squared_array[best_idx]:.6f})')
        
        plt.xlabel('Fourier/Polynomial Transition Point (%)')
        plt.ylabel('R-squared')
        plt.title(f'Fine Search: R-squared vs. Transition Point for {y_column}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{y_column}_fine_optimization.png", dpi=300)
        plt.show()
    
    # Report best results
    if best_params:
        print(f"\nOptimal transition point: {best_params['fourier_fraction']*100:.2f}%")
        print(f"Best R-squared: {best_params['r_squared']:.6f}")
        print(f"  Fourier part R-squared: {best_params['r_squared_fourier']:.6f}")
        print(f"  Polynomial part R-squared: {best_params['r_squared_poly']:.6f}")
        
        # Refit with best parameters and visualize
        final_params = fit_hybrid_profile(
            csv_filepath=csv_filepath,
            x_column=x_column,
            y_column=y_column,
            n_harmonics=n_harmonics,
            poly_degree=poly_degree,
            fourier_fraction=best_params['fourier_fraction'],
            plot_result=True  # Plot the best fit
        )
        
        return final_params
    else:
        print("No valid results found in fine search.")
        return None

# Create WARP implementation using hybrid model
def create_warp_hybrid_profile(params, dt=0.01):
    """Generate a WARP-compatible velocity profile using hybrid model"""
    
    # Extract parameters
    fourier_params = params['fourier_params']
    poly_params = params['poly_params']
    t_min = params['t_min']
    t_transition = params['t_transition']
    t_max = params['t_max']
    period = params['period']
    v_transition = params['v_transition']
    fourier_fraction = params['fourier_fraction']
    
    @wp.func
    def hybrid_profile_warp(index: wp.vec3i, timestep: int = 0):
        """WARP implementation of hybrid profile with smooth cycle transition"""
        # Current time from timestep
        t_raw = wp.float32(timestep) * wp.float32(dt)
        
        # Make cyclic
        t_shift = t_raw - wp.float32(t_min)
        cycles = wp.int32(t_shift / wp.float32(period))
        t_cycle = t_shift - wp.float32(cycles) * wp.float32(period) + wp.float32(t_min)
        
        result = wp.float32(0.0)
        
        # First part - Fourier series
        if t_cycle <= wp.float32(t_transition):
            # Base value (a0)
            result = wp.float32(fourier_params[0])
            
            # Calculate normalized time for Fourier
            scaled_period = wp.float32(period) * wp.float32(fourier_fraction)
            t_norm = t_cycle - wp.float32(t_min)
            
            # Add harmonic terms
            # First harmonic
            if len(fourier_params) >= 3:
                result += wp.float32(fourier_params[1]) * wp.cos(wp.float32(2.0) * wp.pi * wp.float32(1.0) * t_norm / scaled_period)
                result += wp.float32(fourier_params[2]) * wp.sin(wp.float32(2.0) * wp.pi * wp.float32(1.0) * t_norm / scaled_period)
            
            # Second harmonic
            if len(fourier_params) >= 5:
                result += wp.float32(fourier_params[3]) * wp.cos(wp.float32(2.0) * wp.pi * wp.float32(2.0) * t_norm / scaled_period)
                result += wp.float32(fourier_params[4]) * wp.sin(wp.float32(2.0) * wp.pi * wp.float32(2.0) * t_norm / scaled_period)
            
            # Third harmonic
            if len(fourier_params) >= 7:
                result += wp.float32(fourier_params[5]) * wp.cos(wp.float32(2.0) * wp.pi * wp.float32(3.0) * t_norm / scaled_period)
                result += wp.float32(fourier_params[6]) * wp.sin(wp.float32(2.0) * wp.pi * wp.float32(3.0) * t_norm / scaled_period)
            
        else:
            # Second part - Polynomial
            # Start with transition value for continuity
            result = wp.float32(v_transition)
            
            # Calculate normalized time for polynomial (0 at transition, 1 at t_max)
            t_norm = (t_cycle - wp.float32(t_transition)) / (wp.float32(t_max) - wp.float32(t_transition))
            
            # Add polynomial terms
            # Linear term
            if len(poly_params) >= 1:
                result += wp.float32(poly_params[0]) * t_norm
            
            # Quadratic term
            if len(poly_params) >= 2:
                result += wp.float32(poly_params[1]) * t_norm * t_norm
            
            # Cubic term
            if len(poly_params) >= 3:
                result += wp.float32(poly_params[2]) * t_norm * t_norm * t_norm
            
            # 4th degree term
            if len(poly_params) >= 4:
                term = t_norm * t_norm * t_norm * t_norm
                result += wp.float32(poly_params[3]) * term
            
            # 5th degree term
            if len(poly_params) >= 5:
                term = t_norm * t_norm * t_norm * t_norm * t_norm
                result += wp.float32(poly_params[4]) * term
        
        return wp.vec(result, length=1)
    
    return hybrid_profile_warp

# Create a Python version of the hybrid function for visualization
def create_python_hybrid_func(params):
    """Create a Python version of the hybrid function for plotting"""
    fourier_params = params['fourier_params']
    poly_params = params['poly_params']
    t_min = params['t_min']
    t_transition = params['t_transition']
    t_max = params['t_max']
    period = params['period']
    v_transition = params['v_transition']
    fourier_fraction = params['fourier_fraction']
    
    def fourier_part(t):
        # Calculate normalized time for Fourier
        scaled_period = period * fourier_fraction
        t_norm = t - t_min
        
        # Start with base value
        result = fourier_params[0]
        
        # Add harmonic terms
        for i in range(1, len(fourier_params)//2 + 1):
            a = fourier_params[2*i-1]
            b = fourier_params[2*i]
            result += a * np.cos(2*np.pi*i*t_norm/scaled_period) + b * np.sin(2*np.pi*i*t_norm/scaled_period)
        
        return result
    
    def poly_part(t_norm):
        # Start with transition value
        result = v_transition
        
        # Add polynomial terms
        for i, coef in enumerate(poly_params):
            result += coef * (t_norm ** (i+1))
            
        return result
    
    def hybrid_func(t):
        # Make time cyclic
        t_cycle = t_min + ((t - t_min) % period)
        
        # Determine which part of the cycle we're in
        if t_cycle <= t_transition:
            # First part - use Fourier
            return fourier_part(t_cycle)
        else:
            # Second part - use polynomial with normalized time
            t_norm = (t_cycle - t_transition) / (t_max - t_transition)
            return poly_part(t_norm)
    
    return hybrid_func

if __name__ == "__main__":
    # CSV file path
    csv_filepath = "params/velocity_profile_normalized.csv"
    
    # Test both ICA and CCA profiles
    for profile_name in ["ICA", "CCA"]:
        print(f"\n{'='*80}")
        print(f"Processing {profile_name} profile")
        print(f"{'='*80}")
        
        # Find optimal transition point with wide search range
        optimal_params = optimize_transition_point_wide(
            csv_filepath=csv_filepath,
            x_column=None,
            y_column=profile_name,
            n_harmonics=6,
            poly_degree=5,
            coarse_min=0.30,     # 30%
            coarse_max=0.70,     # 70%
            coarse_step=0.01,    # 1% steps for coarse search
            fine_window=0.05,    # ±2.5% around best coarse result
            fine_step=0.0001,    # 0.01% steps for fine search
            plot_comparison=True
        )
        
        if optimal_params is not None:
            # Create WARP implementation with optimal parameters
            warp_profile = create_warp_hybrid_profile(
                params=optimal_params,
                dt=0.01
            )
            
            print(f"Successfully created WARP-compatible {profile_name} hybrid profile")
            
            # Plot cyclic behavior with optimal parameters
            t_min = optimal_params['t_min']
            period = optimal_params['period']
            t_transition = optimal_params['t_transition']
            
            # Create a Python function that generates the velocity at a given time
            optimal_hybrid_func = create_python_hybrid_func(optimal_params)
            
            # Create extended time domain covering multiple cycles
            t_extended = np.linspace(t_min, t_min + 3*period, 3000)
            v_extended = np.array([optimal_hybrid_func(t) for t in t_extended])
            
            plt.figure(figsize=(14, 6))
            plt.plot(t_extended, v_extended, '-', linewidth=2)
            
            # Highlight the period boundaries
            plt.axvline(x=t_min + period, color='r', linestyle='--', alpha=0.5, label='Period boundary')
            plt.axvline(x=t_min + 2*period, color='r', linestyle='--', alpha=0.5)
            
            # Highlight transitions between Fourier and polynomial parts
            fourier_fraction = optimal_params['fourier_fraction']
            plt.axvline(x=t_transition, color='g', linestyle='--', alpha=0.5, 
                     label=f'Optimal transition ({fourier_fraction*100:.2f}%)')
            plt.axvline(x=t_transition + period, color='g', linestyle='--', alpha=0.5)
            plt.axvline(x=t_transition + 2*period, color='g', linestyle='--', alpha=0.5)
            
            plt.xlabel('Time')
            plt.ylabel('Velocity')
            plt.title(f'Cyclic Behavior of {profile_name} with Optimal Transition (R²={optimal_params["r_squared"]:.6f})')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(f"{profile_name}_optimal_cyclic.png", dpi=300)
            plt.show()
    
    print("\nWide-range optimization completed.")