#!/usr/bin/env python3
import os
import sys
import glob
import re
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib.pyplot as plt
from pathlib import Path

def natural_sort_key(s):
    """Sort strings with embedded numbers naturally"""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

def read_vtk_to_numpy(vtk_file_path):
    """
    Read a VTK file and convert its data to numpy arrays
    Returns a dictionary with the array name as key and numpy array as value
    """
    print(f"Reading VTK file: {vtk_file_path}")
    reader = vtk.vtkStructuredPointsReader()
    reader.SetFileName(vtk_file_path)
    reader.Update()
    data = reader.GetOutput()
    
    # Get dimensions for reshaping
    dimensions = data.GetDimensions()
    print(f"VTK dimensions: {dimensions}")
    
    # Get point data
    point_data = data.GetPointData()
    
    # Convert all arrays to numpy
    result = {}
    for i in range(point_data.GetNumberOfArrays()):
        array_name = point_data.GetArrayName(i)
        vtk_array = point_data.GetArray(array_name)
        numpy_array = vtk_to_numpy(vtk_array)
        
        # Reshape to match the image dimensions (height, width)
        reshaped_array = numpy_array.reshape(dimensions[1], dimensions[0])
        result[array_name] = reshaped_array
        print(f"  - Converted field '{array_name}', shape: {reshaped_array.shape}, range: [{np.min(reshaped_array):.4f}, {np.max(reshaped_array):.4f}]")
    
    return result, dimensions

def batch_convert_vtk_to_npy(vtk_folder, output_folder):
    """
    Batch convert all VTK files in a folder to NumPy .npy files
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Find all VTK files
    vtk_files = sorted(glob.glob(os.path.join(vtk_folder, "*.vtk")), key=natural_sort_key)
    print(f"Found {len(vtk_files)} VTK files")
    
    if not vtk_files:
        print(f"No VTK files found in {vtk_folder}")
        return
    
    # Process first file to determine fields
    first_data, dimensions = read_vtk_to_numpy(vtk_files[0])
    fields = list(first_data.keys())
    print(f"Available fields: {fields}")
    
    # Create a separate folder for each field
    field_folders = {}
    for field in fields:
        field_folder = os.path.join(output_folder, field)
        os.makedirs(field_folder, exist_ok=True)
        field_folders[field] = field_folder
    
    # Process each VTK file
    for i, vtk_file in enumerate(vtk_files):
        print(f"Processing file {i+1}/{len(vtk_files)}: {os.path.basename(vtk_file)}")
        data, _ = read_vtk_to_numpy(vtk_file)
        
        # Extract the frame number from the filename
        frame_number = os.path.splitext(os.path.basename(vtk_file))[0].split('_')[-1]
        
        # Save each field as a separate .npy file
        for field, array in data.items():
            output_file = os.path.join(field_folders[field], f"{field}_{frame_number}.npy")
            np.save(output_file, array)
    
    print(f"Conversion complete. Numpy arrays saved to {output_folder}/")
    return field_folders, fields, dimensions

def visualize_sample(field_folders, fields, dimensions):
    """
    Create a sample visualization of the first file in each field
    """
    rows = len(fields)
    fig, axes = plt.subplots(rows, 1, figsize=(10, 4 * rows))
    if rows == 1:
        axes = [axes]
    
    for i, field in enumerate(fields):
        field_folder = field_folders[field]
        npy_files = sorted(glob.glob(os.path.join(field_folder, "*.npy")), key=natural_sort_key)
        
        if npy_files:
            # Load the first file
            data = np.load(npy_files[0])
            
            # Plot the data
            im = axes[i].imshow(data, origin='lower')
            axes[i].set_title(f"{field} - {os.path.basename(npy_files[0])}")
            fig.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "sample_visualization.png"), dpi=150)
    print(f"Sample visualization saved to {os.path.join(output_folder, 'sample_visualization.png')}")

if __name__ == "__main__":
    # Default locations
    default_vtk_folder = os.path.join(os.getcwd(), "aneurysm_simulation_results", "vtk")
    output_folder = os.path.join(os.getcwd(), "aneurysm_numpy_data")
    
    # Check command line arguments
    if len(sys.argv) > 1:
        default_vtk_folder = sys.argv[1]
    if len(sys.argv) > 2:
        output_folder = sys.argv[2]
    
    print(f"VTK folder: {default_vtk_folder}")
    print(f"Output folder: {output_folder}")
    
    # Check if VTK folder exists
    if not os.path.exists(default_vtk_folder):
        print(f"Error: VTK folder {default_vtk_folder} does not exist.")
        sys.exit(1)
    
    # Convert VTK files to NumPy arrays
    field_folders, fields, dimensions = batch_convert_vtk_to_npy(default_vtk_folder, output_folder)
    
    # Create a sample visualization
    visualize_sample(field_folders, fields, dimensions)
    
    print("\nExample code to load and use the NumPy arrays:")
    print("-----------------------------------------------")
    print("import numpy as np")
    print("import matplotlib.pyplot as plt")
    print(f"# Load a specific field (e.g., {fields[0]})")
    print(f"data = np.load('aneurysm_numpy_data/{fields[0]}/{fields[0]}_000000.npy')")
    print("# Visualize the data")
    print("plt.figure(figsize=(10, 8))")
    print("plt.imshow(data, origin='lower')")
    print("plt.colorbar()")
    print("plt.title('2D Visualization')")
    print("plt.show()")