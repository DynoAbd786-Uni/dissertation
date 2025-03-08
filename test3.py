import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import pandas as pd
from typing import Dict, List, Tuple
import warp as wp
from numpy.polynomial.chebyshev import chebfit, chebval
import json
from datetime import datetime


def compare_fitting_methods(csv_filepath: str, x_column=None, y_column=None, 
                          n_harmonics=6, transition_range=(0.3, 0.7), step=0.05):
    """
    Compare different fitting methods across a range of transition points
    
    Parameters:
        csv_filepath: Path to the CSV file
        x_column, y_column: Column names
        n_harmonics: Number of harmonics for Fourier
        transition_range: Range of fourier_fraction values to test
        step: Step size for fourier_fraction
        
    Returns:
        Results for all methods across transition points
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
    
    # Get period
    period = float(t_clean[-1] - t_clean[0])
    t_min = float(t_clean[0])
    t_max = float(t_clean[-1])
    
    # Generate transition points to test
    fourier_fractions = np.arange(transition_range[0], transition_range[1]+step/2, step)
    
    # Store results for each method at each transition point
    # Only include methods with derivable mathematical equations (for WARP compatibility)
    results = {
        'polynomial': [],
        'exponential': [],
        'chebyshev': [],
        'windkessel': []
    }
    
    # For each transition point
    for fourier_fraction in fourier_fractions:
        print(f"\n{'-'*50}")
        print(f"Testing transition point at {fourier_fraction*100:.1f}% of cardiac cycle")
        print(f"{'-'*50}")
        
        # Calculate transition time
        t_transition = t_min + fourier_fraction * period
        
        # Split data
        first_part_mask = t_clean <= t_transition
        second_part_mask = t_clean > t_transition
        
        t_first = t_clean[first_part_mask]
        v_first = v_clean[first_part_mask]
        t_second = t_clean[second_part_mask]
        v_second = v_clean[second_part_mask]
        
        # Fit Fourier to first part (same for all methods)
        def fourier_func(t, *params):
            """Fourier series with n harmonics"""
            a0 = params[0]
            result = a0
            for i in range(1, len(params)//2 + 1):
                a = params[2*i-1]
                b = params[2*i]
                scaled_period = period * fourier_fraction
                result += a * np.cos(2*np.pi*i*(t-t_min)/scaled_period) + b * np.sin(2*np.pi*i*(t-t_min)/scaled_period)
            return result
        
        # Initialize Fourier parameters
        initial_fourier_params = [np.mean(v_first)]
        for _ in range(n_harmonics):
            initial_fourier_params.extend([0.0, 0.0])
        
        # Fit Fourier to first part
        fourier_params, _ = optimize.curve_fit(fourier_func, t_first, v_first, p0=initial_fourier_params)
        v_transition = fourier_func(t_transition, *fourier_params)
        
        # Calculate R-squared for Fourier fit
        v_fourier_fit = fourier_func(t_first, *fourier_params)
        residuals_fourier = v_first - v_fourier_fit
        ss_res_fourier = np.sum(residuals_fourier**2)
        ss_tot_fourier = np.sum((v_first - np.mean(v_first))**2)
        r_squared_fourier = 1 - (ss_res_fourier / ss_tot_fourier)
        
        # Define evaluation functions for all methods with derivable equations
        
        # Method 1: Polynomial - v(t) = v_transition + Σ[coef_i * (t^i+1)]
        t_poly_norm = (t_second - t_transition) / (t_max - t_transition)
        
        def poly_func(t_norm, *params):
            result = v_transition
            for i, coef in enumerate(params):
                result += coef * (t_norm ** (i+1))
            return result
        
        poly_params, _ = optimize.curve_fit(
            poly_func, t_poly_norm, v_second, p0=[0.0] * 5
        )
        
        # Method 2: Exponential Decay - v(t) = a*exp(-b*t) + c*exp(-d*t) + offset
        t_exp_norm = (t_second - t_transition) / (t_max - t_transition)
        
        def exp_decay_func(t, a, b, c, d):
            offset = v_transition - a - c
            return a * np.exp(-b * t) + c * np.exp(-d * t) + offset
        
        p0 = [
            (v_transition - np.min(v_second)) * 0.5,
            5.0,
            (v_transition - np.min(v_second)) * 0.5,
            20.0
        ]
        
        try:
            exp_params, _ = optimize.curve_fit(
                exp_decay_func, t_exp_norm, v_second, p0=p0,
                bounds=([-np.inf, 0, -np.inf, 0], [np.inf, 100, np.inf, 100])
            )
        except:
            exp_params = p0  # Fallback if fitting fails
        
        # Method 3: Chebyshev Polynomials - v(t) = Σ[c_i*T_i(t')]
        t_cheb_norm = -1 + 2 * (t_second - t_transition) / (t_max - t_transition)
        coeffs = chebfit(t_cheb_norm, v_second, deg=5)
        v_at_trans = chebval(-1, coeffs)
        coeffs[0] += v_transition - v_at_trans
        
        # Method 4: Windkessel Model - v(t) = a*exp(-b*t)*(1 + c*sin(d*t + k))
        t_wk_norm = (t_second - t_transition) / (t_max - t_transition)
        
        def windkessel_func(t, a, b, c, d, k):
            c_mod = c * np.sin(k)
            a_adjusted = v_transition / (1 + c_mod)
            return a_adjusted * np.exp(-b * t) * (1 + c * np.sin(d * t + k))
        
        wk_p0 = [v_transition, 5.0, 0.1, 20.0, 0.0]
        
        try:
            wk_params, _ = optimize.curve_fit(
                windkessel_func, t_wk_norm, v_second, p0=wk_p0,
                bounds=([0, 0, -0.5, 0, -np.pi], [np.inf, 50, 0.5, 100, np.pi])
            )
        except:
            wk_params = wk_p0  # Fallback if fitting fails
        
        # Calculate R-squared for each method
        v_poly = poly_func(t_poly_norm, *poly_params)
        v_exp = exp_decay_func(t_exp_norm, *exp_params)
        v_cheb = chebval(t_cheb_norm, coeffs)
        v_wk = windkessel_func(t_wk_norm, *wk_params)
        
        method_outputs = {
            'polynomial': v_poly,
            'exponential': v_exp,
            'chebyshev': v_cheb,
            'windkessel': v_wk
        }
        
        # Calculate R-squared for each method
        for method, v_pred in method_outputs.items():
            residuals = v_second - v_pred
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((v_second - np.mean(v_second))**2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # Store result for this method and transition point
            results[method].append({
                'fourier_fraction': fourier_fraction,
                'r_squared_second': r_squared,
                'r_squared_fourier': r_squared_fourier,
                # Calculate overall R-squared by combining first and second parts
                'r_squared_overall': (len(t_first) * r_squared_fourier + len(t_second) * r_squared) / len(t_clean)
            })
            
            print(f"{method.capitalize()} R² = {r_squared:.6f}")
    
    # Visualize the comparison
    # Create a plot showing R-squared vs transition point for each method
    plt.figure(figsize=(14, 10))
    
    # Plot overall R-squared
    plt.subplot(2, 1, 1)
    for method, method_results in results.items():
        fractions = [r['fourier_fraction'] * 100 for r in method_results]
        r_squared = [r['r_squared_overall'] for r in method_results]
        plt.plot(fractions, r_squared, 'o-', linewidth=2, label=f"{method.capitalize()}")
    
    plt.xlabel('Fourier/Second Method Transition Point (%)')
    plt.ylabel('Overall R-squared')
    plt.title(f'Comparison of Fitting Methods for {y_column} - Overall R-squared')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot second half R-squared
    plt.subplot(2, 1, 2)
    for method, method_results in results.items():
        fractions = [r['fourier_fraction'] * 100 for r in method_results]
        r_squared = [r['r_squared_second'] for r in method_results]
        plt.plot(fractions, r_squared, 'o-', linewidth=2, label=f"{method.capitalize()}")
    
    plt.xlabel('Fourier/Second Method Transition Point (%)')
    plt.ylabel('Second Half R-squared')
    plt.title('R-squared for Second Half Only')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{y_column}_method_comparison.png", dpi=300)
    plt.show()
    
    # Find the best method and transition point
    best_method = None
    best_transition = None
    best_r_squared = -1.0
    
    for method, method_results in results.items():
        for result in method_results:
            if result['r_squared_overall'] > best_r_squared:
                best_r_squared = result['r_squared_overall']
                best_method = method
                best_transition = result['fourier_fraction']
    
    print(f"\n{'='*60}")
    print(f"Best Method: {best_method.capitalize()}")
    print(f"Best Transition Point: {best_transition*100:.1f}%")
    print(f"Best Overall R-squared: {best_r_squared:.6f}")
    print(f"{'='*60}")
    
    return {
        'results': results,
        't_min': t_min,
        't_max': t_max,
        'period': period,
        't_clean': t_clean,
        'v_clean': v_clean,
        'best_method': best_method,
        'best_transition': best_transition,
        'best_r_squared': best_r_squared
    }


if __name__ == "__main__":
    n_harmonics = 6  # Add this line at the beginning of the main block

    # Test both ICA and CCA profiles
    for profile_name in ["ICA", "CCA"]:
        print(f"\n{'='*80}")
        print(f"Processing {profile_name} profile")
        print(f"{'='*80}")
        
        # Two-stage approach for computational efficiency
        # Stage 1: Find rough optimal region with 1% granularity
        print("STAGE 1: Coarse search with 1% granularity")
        coarse_data = compare_fitting_methods(
            csv_filepath="params/velocity_profile_normalized.csv", 
            y_column=profile_name,
            n_harmonics=6,
            transition_range=(0.3, 0.7),  
            step=0.01  # 1% increments for initial scan
        )
        
        # Extract data from coarse search results
        coarse_results = coarse_data['results']
        t_min = coarse_data['t_min']
        t_max = coarse_data['t_max']
        period = coarse_data['period']
        t_clean = coarse_data['t_clean']
        v_clean = coarse_data['v_clean']
        
        # Find best method and approximate transition point from coarse search
        best_method = None
        best_transition_coarse = None
        best_r_squared = -1.0
        
        for method, method_results in coarse_results.items():
            for result in method_results:
                if result['r_squared_overall'] > best_r_squared:
                    best_r_squared = result['r_squared_overall']
                    best_method = method
                    best_transition_coarse = result['fourier_fraction']
        
        print(f"Coarse search best method: {best_method}")
        print(f"Coarse search best transition: {best_transition_coarse*100:.1f}%")
        
        # Stage 2: Fine search around best region with 0.01% granularity
        fine_min = max(0.3, best_transition_coarse - 0.02)  # +/- 2% around best point
        fine_max = min(0.7, best_transition_coarse + 0.02)
        
        print(f"\nSTAGE 2: Fine search from {fine_min*100:.2f}% to {fine_max*100:.2f}% with 0.01% granularity")
        fine_data = compare_fitting_methods(
            csv_filepath="params/velocity_profile_normalized.csv", 
            y_column=profile_name,
            n_harmonics=6,
            transition_range=(fine_min, fine_max),
            step=0.0001  # 0.01% increments for detailed analysis
        )
        
        # Extract data from fine search
        fine_results = fine_data['results']
        
        # Plot high-resolution results
        plt.figure(figsize=(16, 12))
        
        # Plot overall R-squared with enhanced styling
        plt.subplot(2, 1, 1)
        method_colors = {
            'polynomial': '#1f77b4',
            'exponential': '#2ca02c',
            'chebyshev': '#d62728',
            'windkessel': '#9467bd'
        }
        
        for method, method_results in fine_results.items():
            fractions = [r['fourier_fraction'] * 100 for r in method_results]
            r_squared = [r['r_squared_overall'] for r in method_results]
            
            # Use line for connecting points
            plt.plot(fractions, r_squared, '-', linewidth=2, color=method_colors[method], 
                    label=f"{method.capitalize()}")
            
            # Add sparse markers for better visibility
            if len(fractions) > 20:
                marker_indices = np.linspace(0, len(fractions)-1, 20, dtype=int)
                plt.plot(np.array(fractions)[marker_indices], np.array(r_squared)[marker_indices], 
                        'o', markersize=6, color=method_colors[method])
            else:
                plt.plot(fractions, r_squared, 'o', markersize=6, color=method_colors[method])
        
        plt.xlabel('Fourier/Second Method Transition Point (%)', fontsize=12)
        plt.ylabel('Overall R-squared', fontsize=12)
        plt.title(f'Comparison of Derivable Equation Methods for {profile_name} - Overall R-squared (0.01% granularity)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        # Enhanced tickmarks
        plt.tick_params(axis='both', which='major', labelsize=10)
        plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        
        # Find the best results for visualization
        best_results = {}
        for method in fine_results:
            best_idx = np.argmax([r['r_squared_overall'] for r in fine_results[method]])
            best_results[method] = {
                'transition': fine_results[method][best_idx]['fourier_fraction'] * 100,
                'r_squared': fine_results[method][best_idx]['r_squared_overall']
            }
            
        # Add markers for best points of each method
        for method, result in best_results.items():
            plt.plot(result['transition'], result['r_squared'], 'D', 
                   markersize=10, color=method_colors[method], 
                   markeredgecolor='black', markeredgewidth=1.5,
                   label=f"Best {method}: {result['transition']:.2f}% (R²={result['r_squared']:.6f})")
        
        # Plot second half R-squared with enhanced styling
        plt.subplot(2, 1, 2)
        for method, method_results in fine_results.items():
            fractions = [r['fourier_fraction'] * 100 for r in method_results]
            r_squared = [r['r_squared_second'] for r in method_results]
            
            # Use line for connecting points
            plt.plot(fractions, r_squared, '-', linewidth=2, color=method_colors[method], 
                    label=f"{method.capitalize()}")
            
            # Add sparse markers for better visibility
            if len(fractions) > 20:
                marker_indices = np.linspace(0, len(fractions)-1, 20, dtype=int)
                plt.plot(np.array(fractions)[marker_indices], np.array(r_squared)[marker_indices], 
                        'o', markersize=6, color=method_colors[method])
            else:
                plt.plot(fractions, r_squared, 'o', markersize=6, color=method_colors[method])
        
        plt.xlabel('Fourier/Second Method Transition Point (%)', fontsize=12)
        plt.ylabel('Second Half R-squared', fontsize=12)
        plt.title('R-squared for Second Half Only (0.01% granularity)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.tick_params(axis='both', which='major', labelsize=10)
        plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        
        plt.tight_layout()
        plt.savefig(f"{profile_name}_high_resolution_comparison.png", dpi=300)
        plt.show()
        
        # Create plot visualizing the best fit for each method
        plt.figure(figsize=(16, 10))
        
        # Create dense time points for smooth visualization
        t_dense = np.linspace(t_min, t_max, 1000)
        
        # Get original data for reference
        plt.scatter(t_clean, v_clean, s=30, color='black', alpha=0.5, label='Original Data')
        
        # For each method, recreate the best fit using the optimal transition point
        for method, result in best_results.items():
            # Calculate optimal transition point
            optimal_fraction = result['transition'] / 100
            t_transition_opt = t_min + optimal_fraction * period
            
            # Split time points for first and second half
            t_first_dense = t_dense[t_dense <= t_transition_opt]
            t_second_dense = t_dense[t_dense > t_transition_opt]
            
            # Recalculate normalized time for second half
            t_poly_norm_dense = (t_second_dense - t_transition_opt) / (t_max - t_transition_opt)
            t_cheb_norm_dense = -1 + 2 * (t_second_dense - t_transition_opt) / (t_max - t_transition_opt)
            
            # Get optimal parameters for this method and transition point
            best_idx = np.argmax([r['r_squared_overall'] for r in fine_results[method]])
            fourier_fraction = fine_results[method][best_idx]['fourier_fraction']
            
            # First, we need to recreate the optimal fit at this transition point
            # Split data at optimal transition point
            first_part_mask = t_clean <= t_transition_opt
            second_part_mask = t_clean > t_transition_opt
            
            t_first_opt = t_clean[first_part_mask]
            v_first_opt = v_clean[first_part_mask]
            t_second_opt = t_clean[second_part_mask]
            v_second_opt = v_clean[second_part_mask]
            
            # Define Fourier function for first part
            def fourier_func(t, *params):
                a0 = params[0]
                result = a0
                for i in range(1, len(params)//2 + 1):
                    a = params[2*i-1]
                    b = params[2*i]
                    scaled_period = period * fourier_fraction
                    result += a * np.cos(2*np.pi*i*(t-t_min)/scaled_period) + b * np.sin(2*np.pi*i*(t-t_min)/scaled_period)
                return result
            
            # Fit Fourier to first part
            initial_fourier_params = [np.mean(v_first_opt)]
            for _ in range(n_harmonics):
                initial_fourier_params.extend([0.0, 0.0])
                
            fourier_params, _ = optimize.curve_fit(fourier_func, t_first_opt, v_first_opt, p0=initial_fourier_params)
            v_transition_opt = fourier_func(t_transition_opt, *fourier_params)
            
            # Calculate first half values
            v_first_dense = fourier_func(t_first_dense, *fourier_params)
            
            # Calculate second half values based on method
            if method == 'polynomial':
                def poly_func(t_norm, *params):
                    result = v_transition_opt
                    for i, coef in enumerate(params):
                        result += coef * (t_norm ** (i+1))
                    return result
                    
                poly_params, _ = optimize.curve_fit(
                    poly_func, (t_second_opt - t_transition_opt) / (t_max - t_transition_opt), 
                    v_second_opt, p0=[0.0] * 5
                )
                
                v_second_dense = poly_func(t_poly_norm_dense, *poly_params)
                
            elif method == 'exponential':
                def exp_decay_func(t, a, b, c, d):
                    offset = v_transition_opt - a - c
                    return a * np.exp(-b * t) + c * np.exp(-d * t) + offset
                
                p0 = [
                    (v_transition_opt - np.min(v_second_opt)) * 0.5,
                    5.0,
                    (v_transition_opt - np.min(v_second_opt)) * 0.5,
                    20.0
                ]
                
                exp_params, _ = optimize.curve_fit(
                    exp_decay_func, (t_second_opt - t_transition_opt) / (t_max - t_transition_opt), 
                    v_second_opt, p0=p0,
                    bounds=([-np.inf, 0, -np.inf, 0], [np.inf, 100, np.inf, 100])
                )
                
                v_second_dense = exp_decay_func(t_poly_norm_dense, *exp_params)
                
            elif method == 'chebyshev':
                coeffs = chebfit(-1 + 2 * (t_second_opt - t_transition_opt) / (t_max - t_transition_opt), 
                                v_second_opt, deg=5)
                v_at_trans = chebval(-1, coeffs)
                coeffs[0] += v_transition_opt - v_at_trans
                
                v_second_dense = chebval(t_cheb_norm_dense, coeffs)
                
            elif method == 'windkessel':
                def windkessel_func(t, a, b, c, d, k):
                    c_mod = c * np.sin(k)
                    a_adjusted = v_transition_opt / (1 + c_mod)
                    return a_adjusted * np.exp(-b * t) * (1 + c * np.sin(d * t + k))
                
                wk_p0 = [v_transition_opt, 5.0, 0.1, 20.0, 0.0]
                
                wk_params, _ = optimize.curve_fit(
                    windkessel_func, (t_second_opt - t_transition_opt) / (t_max - t_transition_opt), 
                    v_second_opt, p0=wk_p0,
                    bounds=([0, 0, -0.5, 0, -np.pi], [np.inf, 50, 0.5, 100, np.pi])
                )
                
                v_second_dense = windkessel_func(t_poly_norm_dense, *wk_params)
            
            # Combine first and second half
            v_dense = np.concatenate([v_first_dense, v_second_dense])
            
            # Plot this method's best fit
            plt.plot(t_dense, v_dense, '-', linewidth=3, color=method_colors[method], 
                     label=f"{method.capitalize()} (R²={result['r_squared']:.6f})")
            
            # Mark transition point
            plt.axvline(x=t_transition_opt, color=method_colors[method], linestyle='--', alpha=0.5)
        
        # Add vertical line for best overall transition point
        best_method_overall = max(best_results.items(), key=lambda x: x[1]['r_squared'])[0]
        best_transition_overall = best_results[best_method_overall]['transition'] / 100
        t_transition_best = t_min + best_transition_overall * period
        
        # Add annotations
        plt.title(f'Comparison of Best Fitting Methods for {profile_name} Velocity Profile', fontsize=14)
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Velocity (normalized)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f"{profile_name}_best_fits_comparison.png", dpi=300)
        plt.show()
        
        # Create a cycle extension plot for the best method
        plt.figure(figsize=(16, 8))
        
        # Get the best method
        best_method_overall = max(best_results.items(), key=lambda x: x[1]['r_squared'])[0]
        best_r2 = best_results[best_method_overall]['r_squared']
        best_transition_percent = best_results[best_method_overall]['transition']
        
        # Create extended time domain for 3 cycles
        t_extended = np.linspace(t_min, t_min + 3*period, 3000)
        
        # Calculate values for the best method over multiple cycles
        v_extended = np.zeros_like(t_extended)
        
        for i in range(len(t_extended)):
            # Get position within cycle
            t_cycle = t_min + ((t_extended[i] - t_min) % period)
            
            # Determine if in first or second part
            if t_cycle <= t_transition_best:
                # First part - Fourier
                v_extended[i] = fourier_func(t_cycle, *fourier_params)
            else:
                # Second part - Best method
                # Calculate normalized time
                t_norm = (t_cycle - t_transition_best) / (t_max - t_transition_best)
                
                if best_method_overall == 'polynomial':
                    v_extended[i] = poly_func(t_norm, *poly_params)
                elif best_method_overall == 'exponential':
                    v_extended[i] = exp_decay_func(t_norm, *exp_params)
                elif best_method_overall == 'chebyshev':
                    t_cheb = -1 + 2 * t_norm
                    v_extended[i] = chebval(t_cheb, coeffs)
                elif best_method_overall == 'windkessel':
                    v_extended[i] = windkessel_func(t_norm, *wk_params)
        
        # Plot the extended profile
        plt.plot(t_extended, v_extended, '-', linewidth=2, 
                 color=method_colors[best_method_overall],
                 label=f'Best Method: {best_method_overall.capitalize()} (R²={best_r2:.6f})')
                 
        # Mark cycle boundaries
        plt.axvline(x=t_min + period, color='black', linestyle='--', alpha=0.5, label='Cycle Boundary')
        plt.axvline(x=t_min + 2*period, color='black', linestyle='--', alpha=0.5)
        
        # Mark transition points in each cycle
        for i in range(3):
            transition_x = t_min + (i * period) + (best_transition_percent/100 * period)
            plt.axvline(x=transition_x, color=method_colors[best_method_overall], 
                      linestyle=':', alpha=0.7, linewidth=1.5, 
                      label='Transition Point' if i==0 else None)
        
        plt.title(f'Best Method ({best_method_overall.capitalize()}) for {profile_name} Extended Over Multiple Cycles', fontsize=14)
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Velocity (normalized)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f"{profile_name}_{best_method_overall}_extended_cycles.png", dpi=300)
        plt.show()

        # Export the best model parameters to JSON
        print(f"\nExporting best model parameters for {profile_name}...")
        
        # Create a dictionary with all the necessary parameters
        model_params = {
            'profile_name': profile_name,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model_type': 'hybrid_fourier_' + best_method_overall,
            'period': float(period),
            'transition_fraction': float(best_transition_percent / 100),
            'r_squared': float(best_r2),
            'fourier_params': {
                'n_harmonics': n_harmonics,
                'coefficients': [float(p) for p in fourier_params]
            },
            't_min': float(t_min),
            't_max': float(t_max)
        }
        
        # Add method-specific parameters
        if best_method_overall == 'polynomial':
            model_params['polynomial_params'] = {
                'degree': len(poly_params),
                'coefficients': [float(p) for p in poly_params]
            }
        elif best_method_overall == 'exponential':
            model_params['exponential_params'] = {
                'a': float(exp_params[0]),
                'b': float(exp_params[1]),
                'c': float(exp_params[2]),
                'd': float(exp_params[3]),
                'offset': float(v_transition_opt - exp_params[0] - exp_params[2])
            }
        elif best_method_overall == 'chebyshev':
            model_params['chebyshev_params'] = {
                'degree': len(coeffs) - 1,
                'coefficients': [float(c) for c in coeffs]
            }
        elif best_method_overall == 'windkessel':
            model_params['windkessel_params'] = {
                'a': float(wk_params[0]),
                'b': float(wk_params[1]),
                'c': float(wk_params[2]),
                'd': float(wk_params[3]),
                'k': float(wk_params[4]),
                'c_mod': float(wk_params[2] * np.sin(wk_params[4])),
                'a_adjusted': float(v_transition_opt / (1 + wk_params[2] * np.sin(wk_params[4])))
            }
        
        # Save to JSON file
        output_file = f"params/{profile_name}_hybrid_model_params.json"
        with open(output_file, 'w') as f:
            json.dump(model_params, f, indent=4)
        
        print(f"Parameters saved to {output_file}")
        
        # Also export a Python module for easy import
        python_file = f"params/{profile_name.lower()}_model.py"
        with open(python_file, 'w') as f:
            f.write(f"# Auto-generated {profile_name} velocity profile model\n")
            f.write(f"# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("import numpy as np\n")
            
            if best_method_overall == 'chebyshev':
                f.write("from numpy.polynomial.chebyshev import chebval\n")
            
            f.write("\n# Model parameters\n")
            f.write(f"PROFILE_NAME = '{profile_name}'\n")
            f.write(f"MODEL_TYPE = 'hybrid_fourier_{best_method_overall}'\n")
            f.write(f"PERIOD = {period}\n")
            f.write(f"T_MIN = {t_min}\n")
            f.write(f"T_MAX = {t_max}\n")
            f.write(f"TRANSITION_FRACTION = {best_transition_percent / 100}\n")
            f.write(f"R_SQUARED = {best_r2}\n")
            
            f.write("\n# Fourier parameters (first part)\n")
            f.write(f"N_HARMONICS = {n_harmonics}\n")
            f.write(f"FOURIER_PARAMS = {[float(p) for p in fourier_params]}\n")
            
            f.write("\n# Second part parameters\n")
            if best_method_overall == 'polynomial':
                f.write(f"POLY_DEGREE = {len(poly_params)}\n")
                f.write(f"POLY_COEFFS = {[float(p) for p in poly_params]}\n")
            elif best_method_overall == 'exponential':
                f.write(f"EXP_A = {float(exp_params[0])}\n")
                f.write(f"EXP_B = {float(exp_params[1])}\n")
                f.write(f"EXP_C = {float(exp_params[2])}\n")
                f.write(f"EXP_D = {float(exp_params[3])}\n")
                f.write(f"EXP_OFFSET = {float(v_transition_opt - exp_params[0] - exp_params[2])}\n")
            elif best_method_overall == 'chebyshev':
                f.write(f"CHEBY_DEGREE = {len(coeffs) - 1}\n")
                f.write(f"CHEBY_COEFFS = {[float(c) for c in coeffs]}\n")
            elif best_method_overall == 'windkessel':
                f.write(f"WK_A = {float(wk_params[0])}\n")
                f.write(f"WK_B = {float(wk_params[1])}\n")
                f.write(f"WK_C = {float(wk_params[2])}\n")
                f.write(f"WK_D = {float(wk_params[3])}\n")
                f.write(f"WK_K = {float(wk_params[4])}\n")
                f.write(f"WK_C_MOD = {float(wk_params[2] * np.sin(wk_params[4]))}\n")
                f.write(f"WK_A_ADJUSTED = {float(v_transition_opt / (1 + wk_params[2] * np.sin(wk_params[4])))}\n")
            
            # Add a function for easy evaluation of the model
            f.write("\n# Function to evaluate the velocity at any time\n")
            f.write("def velocity(t):\n")
            f.write("    # Make time cyclic\n")
            f.write("    t_cycle = T_MIN + ((t - T_MIN) % PERIOD)\n")
            f.write("    t_transition = T_MIN + TRANSITION_FRACTION * PERIOD\n")
            f.write("    \n")
            f.write("    # First part (Fourier)\n")
            f.write("    if t_cycle <= t_transition:\n")
            f.write("        a0 = FOURIER_PARAMS[0]\n")
            f.write("        result = a0\n")
            f.write("        scaled_period = PERIOD * TRANSITION_FRACTION\n")
            f.write("        for i in range(1, N_HARMONICS + 1):\n")
            f.write("            a = FOURIER_PARAMS[2*i-1]\n")
            f.write("            b = FOURIER_PARAMS[2*i]\n")
            f.write("            result += a * np.cos(2*np.pi*i*(t_cycle-T_MIN)/scaled_period) + \\\n")
            f.write("                     b * np.sin(2*np.pi*i*(t_cycle-T_MIN)/scaled_period)\n")
            f.write("        return result\n")
            f.write("    \n")
            f.write("    # Second part\n")
            f.write("    else:\n")
            f.write("        t_norm = (t_cycle - t_transition) / (T_MAX - t_transition)\n")
            
            if best_method_overall == 'polynomial':
                f.write("        # Polynomial method\n")
                f.write("        result = FOURIER_PARAMS[0]\n")
                f.write("        for i, coef in enumerate(POLY_COEFFS):\n")
                f.write("            result += coef * (t_norm ** (i+1))\n")
            elif best_method_overall == 'exponential':
                f.write("        # Exponential method\n")
                f.write("        result = EXP_A * np.exp(-EXP_B * t_norm) + \\\n")
                f.write("                EXP_C * np.exp(-EXP_D * t_norm) + EXP_OFFSET\n")
            elif best_method_overall == 'chebyshev':
                f.write("        # Chebyshev method\n")
                f.write("        t_cheb = -1 + 2 * t_norm\n")
                f.write("        result = chebval(t_cheb, CHEBY_COEFFS)\n")
            elif best_method_overall == 'windkessel':
                f.write("        # Windkessel method\n")
                f.write("        result = WK_A_ADJUSTED * np.exp(-WK_B * t_norm) * \\\n")
                f.write("                (1 + WK_C * np.sin(WK_D * t_norm + WK_K))\n")
            
            f.write("        return result\n")
        
        print(f"Python module saved to {python_file}")