import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

# Function to apply Gaussian smoothing
def gaussian_smooth(data, sigma=1.0):
    """Apply Gaussian smoothing"""
    return gaussian_filter1d(data, sigma)

# Main function to process the CSV file
def process_velocity_profiles(csv_path):
    """Process and smooth velocity profiles from CSV file using Gaussian smoothing"""
    
    # Read the CSV file
    try:
        data = pd.read_csv(csv_path)
        print(f"Successfully read {csv_path}")
        print(f"Columns: {list(data.columns)}")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    # Assume first column is time, others are velocity profiles
    time_col = data.columns[0]
    t_values = data[time_col].values
    
    # Process each velocity profile column
    for col in data.columns[1:]:
        print(f"\nProcessing velocity profile: {col}")
        
        # Get clean data
        v_values = data[col].values
        mask = ~(np.isnan(t_values) | np.isnan(v_values))
        t_clean = t_values[mask]
        v_clean = v_values[mask]
        
        # Apply Gaussian smoothing with different sigma values
        try:
            # Try different sigma values for comparison
            v_gauss_light = gaussian_smooth(v_clean, sigma=0.8)
            
            # Medium smoothing (sigma=1.5) provides the best balance:
            # - Maintains important peaks and key features of the velocity profile
            # - Effectively irons out small bumps and noise in the data
            # - Preserves the overall shape of the cardiac cycle
            v_gauss_medium = gaussian_smooth(v_clean, sigma=1.5)
            
            v_gauss_heavy = gaussian_smooth(v_clean, sigma=3.0)
            
            # Create plots to compare results
            plt.figure(figsize=(14, 10))
            
            # Plot original data
            plt.subplot(2, 2, 1)
            plt.plot(t_clean, v_clean, 'o-', alpha=0.5, label='Original')
            plt.title(f'Original Data: {col}')
            plt.grid(alpha=0.3)
            plt.legend()
            
            # Plot light smoothing
            plt.subplot(2, 2, 2)
            plt.plot(t_clean, v_clean, 'o', alpha=0.3, label='Original')
            plt.plot(t_clean, v_gauss_light, '-', linewidth=2, label='Gaussian (σ=0.8)')
            plt.title('Light Smoothing')
            plt.grid(alpha=0.3)
            plt.legend()
            
            # Plot medium smoothing - this is the preferred option
            plt.subplot(2, 2, 3)
            plt.plot(t_clean, v_clean, 'o', alpha=0.3, label='Original')
            plt.plot(t_clean, v_gauss_medium, '-', linewidth=2, label='Gaussian (σ=1.5)')
            plt.title('Medium Smoothing (Preferred) - Maintains Peaks, Removes Bumps')
            plt.grid(alpha=0.3)
            plt.legend()
            
            # Plot heavy smoothing
            plt.subplot(2, 2, 4)
            plt.plot(t_clean, v_clean, 'o', alpha=0.3, label='Original')
            plt.plot(t_clean, v_gauss_heavy, '-', linewidth=2, label='Gaussian (σ=3.0)')
            plt.title('Heavy Smoothing')
            plt.grid(alpha=0.3)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(f"{col}_gaussian_smoothing.png", dpi=300)
            plt.show()
            
            # Create a secondary plot showing extended cycles
            plt.figure(figsize=(15, 10))
            
            # Create extended time domain covering multiple cycles
            period = t_clean[-1] - t_clean[0]
            
            # Create a continuous function across cycles with interpolation
            f_orig = interp1d(t_clean % period, v_clean, kind='linear', 
                             fill_value=(v_clean[-1], v_clean[0]), bounds_error=False)
            
            # We use medium smoothing (sigma=1.5) for the final result as it:
            # - Provides optimal balance between smoothness and feature preservation
            # - Maintains the physiologically important peaks in blood velocity
            # - Creates a clean profile suitable for simulation while staying true to the original data
            f_gauss = interp1d(t_clean % period, v_gauss_medium, kind='linear', 
                              fill_value=(v_gauss_medium[-1], v_gauss_medium[0]), bounds_error=False)
            
            # Create smooth multi-cycle time domain
            t_extended = np.linspace(t_clean[0], t_clean[0] + 3*period, 3000)
            
            # Generate properly continuous multi-cycle data
            extended_original = np.array([f_orig(t % period) for t in t_extended])
            extended_gauss = np.array([f_gauss(t % period) for t in t_extended])
            
            # Plot extended cycles
            plt.subplot(2, 1, 1)
            plt.plot(t_extended, extended_original, '-', alpha=0.5, label='Original (tiled)')
            plt.axvline(x=t_clean[0] + period, color='r', linestyle='--', alpha=0.5, label='Period boundary')
            plt.axvline(x=t_clean[0] + 2*period, color='r', linestyle='--', alpha=0.5)
            plt.title('Original Data (Multiple Cycles)')
            plt.grid(alpha=0.3)
            plt.legend()
            
            plt.subplot(2, 1, 2)
            plt.plot(t_extended, extended_gauss, '-', linewidth=2, label='Gaussian Smoothed')
            plt.axvline(x=t_clean[0] + period, color='r', linestyle='--', alpha=0.5, label='Period boundary')
            plt.axvline(x=t_clean[0] + 2*period, color='r', linestyle='--', alpha=0.5)
            plt.title('Gaussian Smoothed (σ=1.5, Multiple Cycles) - Maintains Peaks, Removes Bumps')
            plt.grid(alpha=0.3)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(f"{col}_gaussian_multicycle.png", dpi=300)
            plt.show()
            
            # Save the smoothed data for later use
            # We save the medium smoothing results as our preferred solution
            smoothed_data = pd.DataFrame({
                time_col: t_clean,
                f"{col}_original": v_clean,
                f"{col}_smoothed": v_gauss_medium  # Medium smoothing preserves peaks while removing noise
            })
            smoothed_data.to_csv(f"{col}_smoothed_data.csv", index=False)
            print(f"Saved smoothed data with optimal medium smoothing to {col}_smoothed_data.csv")
            
        except Exception as e:
            print(f"Error processing {col}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    # Path to the CSV file
    csv_path = "params/velocity_profile_normalized.csv"
    
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"Error: File {csv_path} does not exist")
        # Try alternative path
        alt_path = "params/normalized_velocity_profiles.csv"
        if os.path.exists(alt_path):
            print(f"Found alternative file at {alt_path}")
            csv_path = alt_path
        else:
            print(f"Error: Alternative file {alt_path} also does not exist")
            print("Please check the file path or create the file first")
            exit(1)
    
    # Process the velocity profiles
    process_velocity_profiles(csv_path)
    
    print("Processing complete!")