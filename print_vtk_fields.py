#!/usr/bin/env python3
import os
import sys
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import glob
import re

def natural_sort_key(s):
    """Sort strings with embedded numbers naturally"""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

def print_vtk_fields(vtk_file_path):
    """
    Read a VTK file and print all fields contained in it
    """
    print(f"Analyzing VTK file: {vtk_file_path}")
    
    # Try different VTK readers to ensure compatibility
    readers = [
        (vtk.vtkStructuredPointsReader(), "Structured Points"),
        (vtk.vtkUnstructuredGridReader(), "Unstructured Grid"),
        (vtk.vtkPolyDataReader(), "Poly Data"),
        (vtk.vtkStructuredGridReader(), "Structured Grid"),
        (vtk.vtkRectilinearGridReader(), "Rectilinear Grid")
    ]
    
    for reader, reader_type in readers:
        try:
            reader.SetFileName(vtk_file_path)
            reader.Update()
            data = reader.GetOutput()
            
            # If we got any data from this reader, process it
            if data:
                # Get dimensions if available
                dimensions = None
                if hasattr(data, 'GetDimensions'):
                    dimensions = data.GetDimensions()
                    print(f"VTK dimensions: {dimensions}")
                
                # Get number of points and cells
                num_points = data.GetNumberOfPoints()
                num_cells = data.GetNumberOfCells()
                print(f"Reader type: {reader_type}")
                print(f"Number of points: {num_points}")
                print(f"Number of cells: {num_cells}")
                
                # Print point data fields
                point_data = data.GetPointData()
                num_point_arrays = point_data.GetNumberOfArrays()
                print(f"\nPoint Data Fields ({num_point_arrays} fields):")
                
                for i in range(num_point_arrays):
                    array_name = point_data.GetArrayName(i)
                    vtk_array = point_data.GetArray(array_name)
                    numpy_array = vtk_to_numpy(vtk_array)
                    
                    # Get array information
                    array_type = vtk_array.GetDataTypeAsString()
                    num_components = vtk_array.GetNumberOfComponents()
                    num_tuples = vtk_array.GetNumberOfTuples()
                    array_shape = numpy_array.shape
                    
                    # Calculate min, max for the array
                    array_min = numpy_array.min()
                    array_max = numpy_array.max()
                    array_mean = numpy_array.mean()
                    
                    print(f"  {i+1}. {array_name}:")
                    print(f"     - Type: {array_type}")
                    print(f"     - Shape: {array_shape}")
                    print(f"     - Components: {num_components}")
                    print(f"     - Tuples: {num_tuples}")
                    print(f"     - Range: [{array_min:.6f}, {array_max:.6f}]")
                    print(f"     - Mean: {array_mean:.6f}")
                    
                    # Try to reshape array if dimensions are available
                    if dimensions and len(dimensions) >= 2:
                        try:
                            # For structured data, try to reshape to grid dimensions
                            expected_size = dimensions[0] * dimensions[1]
                            if num_components == 1 and num_tuples == expected_size:
                                reshaped = numpy_array.reshape(dimensions[1], dimensions[0])
                                print(f"     - Reshaped: {reshaped.shape} (height, width)")
                        except Exception as e:
                            print(f"     - Reshape failed: {e}")
                
                # Print cell data fields
                cell_data = data.GetCellData()
                num_cell_arrays = cell_data.GetNumberOfArrays()
                print(f"\nCell Data Fields ({num_cell_arrays} fields):")
                
                for i in range(num_cell_arrays):
                    array_name = cell_data.GetArrayName(i)
                    vtk_array = cell_data.GetArray(array_name)
                    numpy_array = vtk_to_numpy(vtk_array)
                    
                    # Get array information
                    array_type = vtk_array.GetDataTypeAsString()
                    num_components = vtk_array.GetNumberOfComponents()
                    num_tuples = vtk_array.GetNumberOfTuples()
                    array_shape = numpy_array.shape
                    
                    # Calculate min, max for the array
                    array_min = numpy_array.min()
                    array_max = numpy_array.max()
                    array_mean = numpy_array.mean()
                    
                    print(f"  {i+1}. {array_name}:")
                    print(f"     - Type: {array_type}")
                    print(f"     - Shape: {array_shape}")
                    print(f"     - Components: {num_components}")
                    print(f"     - Tuples: {num_tuples}")
                    print(f"     - Range: [{array_min:.6f}, {array_max:.6f}]")
                    print(f"     - Mean: {array_mean:.6f}")
                
                # We found a valid reader, no need to try others
                return True
                
        except Exception as e:
            continue
    
    print(f"Error: Could not read {vtk_file_path} with any VTK reader")
    return False

def find_vtk_files():
    """Find VTK files in the project directory"""
    vtk_dir = os.path.join(os.getcwd(), "aneurysm_simulation_results", "vtk")
    
    if not os.path.exists(vtk_dir):
        print(f"VTK directory not found: {vtk_dir}")
        return []
    
    vtk_files = sorted(glob.glob(os.path.join(vtk_dir, "*.vtk")), key=natural_sort_key)
    return vtk_files

if __name__ == "__main__":
    # Check if a specific file was provided
    if len(sys.argv) > 1:
        vtk_file = sys.argv[1]
        if os.path.exists(vtk_file):
            print_vtk_fields(vtk_file)
        else:
            print(f"File not found: {vtk_file}")
    else:
        # Find all VTK files
        vtk_files = find_vtk_files()
        
        if not vtk_files:
            print("No VTK files found. Please provide a path to a VTK file.")
            sys.exit(1)
        
        # Print fields from the first file by default
        first_file = vtk_files[0]
        print(f"Found {len(vtk_files)} VTK files. Analyzing first file.")
        print_vtk_fields(first_file)
        
        # Offer to print fields for all files
        if len(vtk_files) > 1:
            print("\nOther VTK files found:")
            for i, file in enumerate(vtk_files[:5]):  # Show first 5 files
                print(f"  {i+1}. {os.path.basename(file)}")
            
            if len(vtk_files) > 5:
                print(f"  ... and {len(vtk_files) - 5} more files")
            
            print("\nTo analyze a specific file, run:")
            print(f"python {sys.argv[0]} <path_to_vtk_file>")