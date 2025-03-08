import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import pandas as pd
from typing import Dict, List, Tuple
import warp as wp
import jax.numpy as jnp
from xlb.compute_backend import ComputeBackend
from scipy.fft import rfft, rfftfreq

# Define modified exponential function with complex oscillatory components
def exp_func_global(t, params):
    """
    Sum of complex exponential terms (sines and cosines): 
    f(t) = a₀ + Σ a_i*sin(b_i*t + c_i) + Σ d_j*exp(e_j*t)
    """
    n_sin_terms = params[0]  # Number of sine terms
    n_exp_terms = params[1]  # Number of exponential terms
    baseline = params[2]  # Baseline value
    
    result = baseline
    
    # Add sine terms (for oscillatory components)
    param_idx = 3
    for i in range(int(n_sin_terms)):
        a = params[param_idx]     # Amplitude
        b = params[param_idx+1]   # Frequency
        c = params[param_idx+2]   # Phase
        
        result += a * np.sin(b * t + c)
        param_idx += 3
    
    # Add exponential terms (for growth/decay components)
    for i in range(int(n_exp_terms)):
        d = params[param_idx]     # Amplitude
        e = params[param_idx+1]   # Rate
        
        result += d * np.exp(e * t)
        param_idx += 2
    
    return result

def fit_exp_profile(csv_filepath: str, x_column=None, y_column=None, 
                  n_sin_terms=5, n_exp_terms=5, plot_result=True):
    """
    Create a velocity profile using a sum of sines and exponential functions
    
    Parameters:
        csv_filepath: Path to the CSV file
        x_column, y_column: Column names
        n_sin_terms: Number of sine terms
        n_exp_terms: Number of exponential terms
        plot_result: Whether to plot the results
    """
    # Read data
    data = pd.read_csv(csv_filepath)
    
    # Handle column selection
    if x_column is None:
        x_column = data.columns[0]
    if y_column is None:
        y_column = [col for col in data.columns if col != x_column][0]
    
    # Extract clean data
    t = data[x_column].values.astype(np.float32)
    v = data[y_column].values.astype(np.float32)
    valid_mask = ~(np.isnan(t) | np.isnan(v))
    t_clean = t[valid_mask]
    v_clean = v[valid_mask]
    
    # Get period parameters
    period = float(t_clean[-1] - t_clean[0])
    t_min = float(t_clean[0])
    t_max = float(t_clean[-1])
    
    # Normalize time for numerical stability
    t_norm = (t_clean - t_min) / period
    
    # Use FFT to get initial guesses for frequencies
    yf = rfft(v_clean)
    xf = rfftfreq(len(t_norm), t_norm[1] - t_norm[0])
    
    # Find dominant frequencies
    power = np.abs(yf)**2
    freq_idx = np.argsort(power)[::-1]  # Sort by descending power
    dominant_freqs = xf[freq_idx[:max(n_sin_terms, 3)]]  # Get top frequencies
    
    # For curve_fit compatibility
    def exp_func_wrapper(t, *params):
        return exp_func_global(t, params)
    
    # Initial parameter values
    baseline = np.mean(v_clean)
    initial_params = [n_sin_terms, n_exp_terms, baseline]  # Meta-params and baseline
    
    # Add sine terms with good initial guesses
    for i in range(n_sin_terms):
        # Amplitude based on data range
        amp = (np.max(v_clean) - np.min(v_clean)) * 0.2
        
        # Frequency from FFT if available, otherwise use harmonics
        if i < len(dominant_freqs):
            freq = dominant_freqs[i] * 2 * np.pi
        else:
            freq = 2 * np.pi * (i + 1)  # i-th harmonic
        
        # Random phase
        phase = np.random.uniform(-np.pi, np.pi)
        
        initial_params.extend([amp, freq, phase])
    
    # Add exponential terms
    for i in range(n_exp_terms):
        # Small amplitude
        amp = 0.05 * (np.random.rand() - 0.5)
        
        # Rate constants that decay at different speeds
        rate = -5.0 - i*2.0
        
        initial_params.extend([amp, rate])
    
    # Fix parameters that should not be optimized
    fixed_params = [True, True] + [False] * (len(initial_params) - 2)
    
    # Fit the model
    try:
        print(f"Fitting hybrid sine-exponential model with {n_sin_terms} sine terms and {n_exp_terms} exponential terms...")
        
        # First fit using curve_fit with only variable parameters
        variable_params_idx = [i for i, fixed in enumerate(fixed_params) if not fixed]
        variable_initial_params = [initial_params[i] for i in variable_params_idx]
        
        def wrapped_func(t, *variable_params):
            full_params = initial_params.copy()
            for i, param_idx in enumerate(variable_params_idx):
                full_params[param_idx] = variable_params[i]
            return exp_func_global(t, full_params)
        
        optimized_variable_params, _ = optimize.curve_fit(
            wrapped_func, t_norm, v_clean, 
            p0=variable_initial_params,
            method='trf',
            maxfev=10000,
            bounds=([-np.inf] * len(variable_initial_params), 
                   [np.inf] * len(variable_initial_params))
        )
        
        # Reconstruct full parameter list
        exp_params = initial_params.copy()
        for i, param_idx in enumerate(variable_params_idx):
            exp_params[param_idx] = optimized_variable_params[i]
        
        print("Fitting completed successfully")
        
        # Calculate R-squared for the fit
        v_exp_fit = exp_func_global(t_norm, exp_params)
        residuals_exp = v_clean - v_exp_fit
        ss_res_exp = np.sum(residuals_exp**2)
        ss_tot_exp = np.sum((v_clean - np.mean(v_clean))**2)
        r_squared_exp = 1 - (ss_res_exp / ss_tot_exp)
        
    except Exception as e:
        print(f"Fitting error: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Create periodic function
    def periodic_func(t):
        """Wraps the function to be periodic"""
        # Make time cyclic
        t_cycle = t_min + ((t - t_min) % period)
        
        # Convert to normalized time
        t_norm_cycle = (t_cycle - t_min) / period
        
        # Evaluate the function
        return exp_func_global(t_norm_cycle, exp_params)
    
    # Plot results if requested
    if plot_result:
        # Create high-resolution time points for plotting
        t_fine = np.linspace(t_min, t_max, 1000)
        t_fine_norm = (t_fine - t_min) / period
        v_model = exp_func_global(t_fine_norm, exp_params)
        
        # Plot original data with fit
        plt.figure(figsize=(14, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(t_clean, v_clean, 'o', alpha=0.3, label=f'Original {y_column} data')
        plt.plot(t_fine, v_model, '-', linewidth=2, 
                label=f'Combined model (R²={r_squared_exp:.4f})')
        
        # Add labels and legend
        plt.xlabel(f'{x_column}')
        plt.ylabel(f'{y_column}')
        plt.title(f'Hybrid Sine-Exponential Model ({n_sin_terms} sine + {n_exp_terms} exp terms)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot residuals
        plt.subplot(2, 1, 2)
        plt.plot(t_clean, residuals_exp, 'o-', alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.xlabel(f'{x_column}')
        plt.ylabel('Residuals')
        plt.title(f'Model Residuals (R²={r_squared_exp:.4f})')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{y_column}_hybrid_model_fit.png", dpi=300)
        plt.show()
        
        # Plot cyclic behavior over multiple periods
        t_extended = np.linspace(t_min, t_min + 3*period, 3000)
        v_extended = np.array([periodic_func(t) for t in t_extended])
        
        plt.figure(figsize=(14, 6))
        plt.plot(t_extended, v_extended, '-', linewidth=2)
        
        # Highlight the period boundaries
        plt.axvline(x=t_min + period, color='r', linestyle='--', alpha=0.5, label='Period boundary')
        plt.axvline(x=t_min + 2*period, color='r', linestyle='--', alpha=0.5)
        
        plt.xlabel('Time')
        plt.ylabel('Velocity')
        plt.title(f'Cyclic Behavior of {y_column} (R²={r_squared_exp:.6f})')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{y_column}_hybrid_model_cyclic.png", dpi=300)
        plt.show()
    
    # Return parameters and R-squared
    return {
        'exp_params': exp_params,
        't_min': t_min,
        't_max': t_max,
        'period': period,
        'n_sin_terms': n_sin_terms,
        'n_exp_terms': n_exp_terms,
        'r_squared': r_squared_exp
    }

# Create WARP implementation 
def create_warp_hybrid_profile(params, dt=0.01):
    """Generate a WARP-compatible velocity profile"""
    
    # Extract parameters
    exp_params = params['exp_params']
    t_min = params['t_min']
    t_max = params['t_max']
    period = params['period']
    n_sin_terms = int(exp_params[0])
    n_exp_terms = int(exp_params[1])
    
    @wp.func
    def hybrid_profile_warp(index: wp.vec3i, timestep: int = 0):
        """WARP implementation of hybrid profile model"""
        # Current time from timestep
        t_raw = wp.float32(timestep) * wp.float32(dt)
        
        # Make cyclic
        t_shift = t_raw - wp.float32(t_min)
        cycles = wp.int32(t_shift / wp.float32(period))
        t_cycle = t_shift - wp.float32(cycles) * wp.float32(period) + wp.float32(t_min)
        
        # Convert to normalized time
        t_norm = (t_cycle - wp.float32(t_min)) / wp.float32(period)
        
        # Baseline value
        result = wp.float32(exp_params[2])
        
        # Add sine terms - we hard code the first few to ensure we capture major features
        # Parameter indices
        param_idx = 3
        
        # First sine term
        if n_sin_terms >= 1:
            a = wp.float32(exp_params[param_idx])     # Amplitude
            b = wp.float32(exp_params[param_idx+1])   # Frequency
            c = wp.float32(exp_params[param_idx+2])   # Phase
            result += a * wp.sin(b * t_norm + c)
            param_idx += 3
        
        # Second sine term
        if n_sin_terms >= 2:
            a = wp.float32(exp_params[param_idx])
            b = wp.float32(exp_params[param_idx+1])
            c = wp.float32(exp_params[param_idx+2])
            result += a * wp.sin(b * t_norm + c)
            param_idx += 3
        
        # Third sine term
        if n_sin_terms >= 3:
            a = wp.float32(exp_params[param_idx])
            b = wp.float32(exp_params[param_idx+1])
            c = wp.float32(exp_params[param_idx+2])
            result += a * wp.sin(b * t_norm + c)
            param_idx += 3
        
        # Fourth sine term
        if n_sin_terms >= 4:
            a = wp.float32(exp_params[param_idx])
            b = wp.float32(exp_params[param_idx+1])
            c = wp.float32(exp_params[param_idx+2])
            result += a * wp.sin(b * t_norm + c)
            param_idx += 3
        
        # Fifth sine term
        if n_sin_terms >= 5:
            a = wp.float32(exp_params[param_idx])
            b = wp.float32(exp_params[param_idx+1])
            c = wp.float32(exp_params[param_idx+2])
            result += a * wp.sin(b * t_norm + c)
            param_idx += 3
        
        # Add first two exponential terms
        if n_exp_terms >= 1:
            d = wp.float32(exp_params[param_idx])
            e = wp.float32(exp_params[param_idx+1])
            result += d * wp.exp(e * t_norm)
            param_idx += 2
        
        if n_exp_terms >= 2:
            d = wp.float32(exp_params[param_idx])
            e = wp.float32(exp_params[param_idx+1])
            result += d * wp.exp(e * t_norm)
            param_idx += 2
        
        return wp.vec(result, length=1)
    
    return hybrid_profile_warp

if __name__ == "__main__":
    # CSV file path
    csv_filepath = "params/velocity_profile_normalized.csv"
    
    # Test both ICA and CCA profiles
    for profile_name in ["ICA", "CCA"]:
        print(f"\n{'='*80}")
        print(f"Processing {profile_name} profile with hybrid model")
        print(f"{'='*80}")
        
        # Fit with hybrid model - use both sine and exponential terms
        hybrid_params = fit_exp_profile(
            csv_filepath=csv_filepath,
            x_column=None,
            y_column=profile_name,
            n_sin_terms=5,  # Sine terms for oscillatory features
            n_exp_terms=2,  # Exponential terms for transitions
            plot_result=True
        )
        
        if hybrid_params is not None:
            print(f"Successfully created hybrid model for {profile_name}")
            print(f"R-squared: {hybrid_params['r_squared']:.6f}")
            
            # Create WARP implementation
            warp_profile = create_warp_hybrid_profile(
                params=hybrid_params,
                dt=0.01
            )
            
            print(f"Successfully created WARP-compatible {profile_name} hybrid profile")
    
    print("\nHybrid modeling completed.")