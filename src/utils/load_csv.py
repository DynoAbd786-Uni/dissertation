import os
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from glob import glob
from utils.constants import PARAMS_DIR



def load_csv_data(vessel_radius_mm=None, dx=None, dt=None, plot=False, normalize_time=True, resample_points=64):
    """
    Load multiple CSV files and convert flow rates to appropriate units.
    
    Args:
        vessel_radius_mm: Radius of the vessel in mm (needed for conversion)
                        If None, data is kept as raw ml/s.
        dx: Grid spacing in meters. If provided with dt, converts velocity to LU/step.
        dt: Time step in seconds. If provided with dx, converts velocity to LU/step.
        plot: Boolean flag to enable plotting of the graphs.
        normalize_time: If True, normalize time values to span 1 second.
        resample_points: Number of points to resample to (default: 64)

    Returns:
        dict: Nested dictionary with structure:
            {
                'file1': {
                    'x': np.array(...),  # Time/x values (normalized if requested)
                    'y': {
                        'col1': np.array(...),  # Velocity in lattice units if dx/dt provided
                        'col2': np.array(...),
                        ...
                    },
                    'original_duration': float,  # Original time span in seconds
                    'original_points': int,      # Original number of data points
                    'units': str,                # Units of the y values
                    'max_velocity': float,       # Maximum velocity in m/s before lattice conversion
                    'max_lattice_velocity': float,       # Maximum velocity in lattice units per step
                },
                'file2': {
                    ...
                }
            }
    """
    file_paths = glob(PARAMS_DIR + "/*.csv")
    print("file_paths", file_paths)
    result = {}
    
    for file_path in file_paths:
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        df = pd.read_csv(file_path)
        
        # Identify the x (time) column
        x_column = None
        for candidate in ['time', 'Time', 't', 'T', 'x', 'X']:
            if candidate in df.columns:
                x_column = candidate
                break
        
        if x_column is None:
            x_column = df.columns[0]  # Default to first column if no explicit time found
            print(f"No explicit time column found in {file_name}. Using {x_column} as x-axis.")
        
        # Extract time values
        time_values = df[x_column].to_numpy()
        time_min = time_values.min()
        time_max = time_values.max()
        original_duration = time_max - time_min
        original_points = len(time_values)
        
        # Normalize time
        if normalize_time:
            # Shift to start from 0 and scale to 1 second period
            normalized_time = (time_values - time_min) / original_duration
        else:
            normalized_time = time_values
            
        file_data = {
            'x': normalized_time, 
            'y': {}, 
            'original_duration': original_duration,
            'original_points': original_points,
            'units': 'ml/s' if vessel_radius_mm is None else 'm/s'
        }
        
        if normalize_time:
            print(f"Normalizing time for {file_name} from {original_duration:.4f}s to 1.0s")
        
        # Process y values and keep track of max velocity
        max_velocity = 0.0
        
        if vessel_radius_mm is not None:
            vessel_radius_m = vessel_radius_mm * 0.001
            cross_section_area = math.pi * vessel_radius_m**2
            conversion_factor = 1e-6 / cross_section_area  # ml/s to m³/s, then to m/s

            print(f"Converting flow rate (ml/s) to velocity (m/s) with vessel radius {vessel_radius_mm} mm")
            print(f"Conversion factor: {conversion_factor:.6e}")
            
            for column in df.columns:
                if column != x_column:
                    raw_values = df[column].to_numpy()
                    velocity_values = raw_values * conversion_factor
                    file_data['y'][column] = velocity_values
                    
                    # Store maximum velocity before lattice unit conversion
                    col_max = np.max(velocity_values)
                    if col_max > max_velocity:
                        max_velocity = col_max
                        
                    print(f"  Column {column}: min={velocity_values.min():.4f} m/s, max={col_max:.4f} m/s")
        else:
            # If we don't have vessel radius, we're still in flow rate units
            for column in df.columns:
                if column != x_column:
                    flow_values = df[column].to_numpy()
                    file_data['y'][column] = flow_values
                    
                    # Store maximum flow rate
                    col_max = np.max(flow_values)
                    if col_max > max_velocity:
                        max_velocity = col_max
        
        # Store the max velocity/flow rate in the file_data
        file_data['max_velocity'] = max_velocity
        file_data['max_lattice_velocity'] = max_velocity * conversion_factor if conversion_factor else None
        
        # Convert to lattice units if dx and dt are provided
        if dx is not None and dt is not None:
            conversion_factor = dt / dx
            file_data['units'] = 'LU/step'
            print(f"Converting velocity to lattice units with dx={dx} m and dt={dt} s")
            print(f"Conversion factor: {conversion_factor:.6e}")
            
            # Store max lattice velocity
            file_data['max_lattice_velocity'] = max_velocity * conversion_factor
            
            for column in file_data['y']:
                file_data['y'][column] *= conversion_factor
                print(f"  Column {column}: min={file_data['y'][column].min():.4f} LU/step, max={file_data['y'][column].max():.4f} LU/step")
        
        # Resample to exact number of points if requested
        if resample_points > 0:
            if original_points != resample_points:
                print(f"Resampling {file_name} from {original_points} to {resample_points} points")
        
        result[file_name] = file_data
        
        # Plot if `plot=True`
        if plot:
            plt.figure(figsize=(8, 5))
            for col, y_values in file_data['y'].items():
                plt.plot(file_data['x'], y_values, label=col)
            plt.xlabel(x_column)
            plt.ylabel("Velocity (LU/step)" if dx and dt else ("Velocity (m/s)" if vessel_radius_mm else "Flow rate (ml/s)"))
            plt.title(f"Flow Profile: {file_name}")
            plt.legend()
            plt.grid(True)
            plt.show()
        
        # Optional: Print the maximum velocity information
        print(f"Maximum {'velocity' if vessel_radius_mm else 'flow rate'}: {max_velocity:.4f} {'m/s' if vessel_radius_mm else 'ml/s'}")
    
    return result

# Example usage:
if __name__ == "__main__":
    data = load_csv_data(vessel_radius_mm=2.0, plot=True)  # Enable plotting

    # Access data
    for file_name, file_data in data.items():
        print(f"File: {file_name}")
        print(f"  X range: {file_data['x'].min()} to {file_data['x'].max()}")
        print(f"  Y columns: {list(file_data['y'].keys())}")
        print()