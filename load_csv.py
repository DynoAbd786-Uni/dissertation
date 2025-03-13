import os
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from glob import glob

PARAMS_DIR = "params"

def load_csv_data(vessel_radius_mm=None, plot=False):
    """
    Load multiple CSV files and convert flow rates (ml/s) to velocity (m/s).
    
    Args:
        vessel_radius_mm: Radius of the vessel in mm (needed for conversion)
                        If None, data is kept as raw ml/s.
        plot: Boolean flag to enable plotting of the graphs.

    Returns:
        dict: Nested dictionary with structure:
            {
                'file1': {
                    'x': np.array(...),  # Time/x values
                    'y': {
                        'col1': np.array(...),  # Velocity in m/s if vessel_radius_mm provided
                        'col2': np.array(...),
                        ...
                    }
                },
                'file2': {
                    ...
                }
            }
    """
    file_paths = glob(PARAMS_DIR + "/*.csv")
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
        
        file_data = {'x': df[x_column].to_numpy(), 'y': {}}
        
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
                    print(f"  Column {column}: min={velocity_values.min():.4f} m/s, max={velocity_values.max():.4f} m/s")
        else:
            for column in df.columns:
                if column != x_column:
                    file_data['y'][column] = df[column].to_numpy()
        
        result[file_name] = file_data
        
        # ✅ Plot if `plot=True`
        if plot:
            plt.figure(figsize=(8, 5))
            for col, y_values in file_data['y'].items():
                plt.plot(file_data['x'], y_values, label=col)
            plt.xlabel(x_column)
            plt.ylabel("Velocity (m/s)" if vessel_radius_mm else "Flow rate (ml/s)")
            plt.title(f"Flow Profile: {file_name}")
            plt.legend()
            plt.grid(True)
            plt.show()
    
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