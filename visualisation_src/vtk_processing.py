#!/usr/bin/env python3
"""
Simulation VTK Data Processor

This script loads .vtk files from LBM simulations (aneurysm, pipe, etc.), extracts the specific 
fields (rho, velocity, wall shear stress, etc.), and organizes them into dictionaries of NumPy 
arrays for further analysis and visualization. Each field will be accessible by frame number.

This version has been modified to automatically search for VTK files recursively starting from 
one directory above the current working directory.
"""

import os
import numpy as np
import vtk
from vtk.util import numpy_support
from pathlib import Path
import re
import tqdm
import time
from datetime import timedelta
import argparse


def natural_sort_key(s):
    """Sort strings with embedded numbers naturally"""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', str(s))]


def extract_frame_number(filename):
    """Extract the frame number from the VTK filename"""
    # Extract all digits from the filename
    digits = ''.join(re.findall(r'\d+', filename))
    if digits:
        return int(digits)
    return None


def read_vtk_file(file_path):
    """Read a VTK file and return the data object"""
    reader = vtk.vtkStructuredPointsReader()
    reader.SetFileName(str(file_path))
    reader.Update()
    return reader.GetOutput()


def extract_vtk_fields(vtk_data):
    """Extract all fields from a VTK data object"""
    fields = {}
    
    # Extract cell data (where most of our fields are located)
    cell_data = vtk_data.GetCellData()
    num_arrays = cell_data.GetNumberOfArrays()
    
    for i in range(num_arrays):
        array_name = cell_data.GetArrayName(i)
        vtk_array = cell_data.GetArray(array_name)
        numpy_array = numpy_support.vtk_to_numpy(vtk_array)
        fields[array_name] = numpy_array
    
    # Extract point data (if any)
    point_data = vtk_data.GetPointData()
    num_point_arrays = point_data.GetNumberOfArrays()
    
    for i in range(num_point_arrays):
        array_name = point_data.GetArrayName(i)
        vtk_array = point_data.GetArray(array_name)
        numpy_array = numpy_support.vtk_to_numpy(vtk_array)
        fields[f"point_{array_name}"] = numpy_array
    
    # Store dimensions for future reference
    dims = vtk_data.GetDimensions()
    fields['dimensions'] = np.array(dims)
    
    return fields


def process_all_vtk_files(vtk_files, max_files=None):
    """Process all VTK files and organize their fields by name and frame"""
    # Initialize dictionaries to store each field
    field_data = {}
    frame_numbers = []
    
    # Limit the number of files to process if specified
    if max_files is not None and max_files > 0:
        vtk_files = vtk_files[:max_files]
    
    # Process each VTK file
    for file_path in tqdm.tqdm(vtk_files, desc="Processing VTK files"):
        try:
            # Extract frame number from filename
            frame_num = extract_frame_number(file_path.name)
            if frame_num is None:
                continue
                
            frame_numbers.append(frame_num)
            
            # Read and extract fields
            vtk_data = read_vtk_file(file_path)
            fields = extract_vtk_fields(vtk_data)
            
            # Organize fields by name, with frames stored sequentially
            for field_name, field_array in fields.items():
                if field_name not in field_data:
                    field_data[field_name] = {}
                
                field_data[field_name][frame_num] = field_array
                
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
    
    # Get list of unique field names
    all_fields = list(field_data.keys())
    print(f"\nProcessed {len(frame_numbers)} VTK files with {len(all_fields)} fields per file")
    print(f"Field names: {', '.join(all_fields)}")
    
    return field_data, sorted(frame_numbers)


def stack_fields_into_arrays(field_data, frame_numbers):
    """Stack all frames for each field into 3D NumPy arrays"""
    stacked_fields = {}
    
    for field_name, frames_data in field_data.items():
        if field_name == 'dimensions':
            # Just store one copy of the dimensions
            stacked_fields[field_name] = next(iter(frames_data.values()))
            continue
        
        # Check if all frames have the same shape
        shapes = {frame: data.shape for frame, data in frames_data.items()}
        if len(set(shapes.values())) > 1:
            print(f"Warning: Field '{field_name}' has inconsistent shapes across frames")
            stacked_fields[field_name] = frames_data  # Store as dictionary
            continue
        
        # All frames have the same shape, stack them into a 3D array
        try:
            # Create arrays in order of frame numbers
            sorted_frames = sorted(frames_data.keys())
            arrays_to_stack = [frames_data[frame] for frame in sorted_frames]
            
            # Stack along a new first dimension (frame)
            stacked_array = np.stack(arrays_to_stack, axis=0)
            
            # Store the stacked array and frame mapping
            stacked_fields[field_name] = stacked_array
            
            print(f"Stacked '{field_name}' into array with shape {stacked_array.shape}")
            
        except Exception as e:
            print(f"Error stacking {field_name}: {e}")
            # If stacking fails, store as a dictionary
            stacked_fields[field_name] = frames_data
    
    # Create a mapping from frame numbers to indices
    frame_to_index = {frame: i for i, frame in enumerate(sorted(frame_numbers))}
    
    return stacked_fields, frame_to_index


def reshape_all_fields_to_2d(stacked_fields, frame_to_index=None):
    """Reshape all 1D field data into 2D arrays for easier analysis and visualization."""
    if 'dimensions' not in stacked_fields:
        print("Dimensions not available for reshaping data")
        return stacked_fields
    
    # Get dimensions for reshaping
    dimensions = stacked_fields['dimensions']
    real_dims = [d for d in dimensions if d > 1]
    
    if len(real_dims) < 2:
        print("Cannot reshape data - insufficient dimensions")
        return stacked_fields
    
    # Dimensions for reshaping - need to subtract 1 because VTK dimensions are cell counts + 1
    ny, nx = real_dims[1]-1, real_dims[0]-1
    print(f"Reshaping fields to dimensions: ({ny}, {nx})")
    
    # Create a new dictionary to store the reshaped fields
    reshaped_fields = {'dimensions': stacked_fields['dimensions']}
    
    # Process each field
    for field_name, field_data in stacked_fields.items():
        if field_name == 'dimensions':
            continue
            
        print(f"Reshaping field: {field_name}")
        
        # Handle 3D arrays (stacked frames)
        if isinstance(field_data, np.ndarray) and field_data.ndim > 1:
            # Get the number of frames
            n_frames = field_data.shape[0]
            
            # Create a new array with reshaped dimensions
            reshaped_array = np.zeros((n_frames, ny, nx), dtype=field_data.dtype)
            
            # Reshape each frame
            for i in range(n_frames):
                frame_data = field_data[i]
                # Reshape the 1D array to 2D
                try:
                    reshaped_array[i] = np.flipud(frame_data.reshape(ny, nx))
                except Exception as e:
                    print(f"Error reshaping frame {i} for {field_name}: {e}")
                    # Keep original data if reshaping fails
                    reshaped_array[i] = frame_data
            
            # Store the reshaped array
            reshaped_fields[field_name] = reshaped_array
            
        # Handle dictionary of frames
        elif isinstance(field_data, dict):
            reshaped_dict = {}
            
            # Reshape each frame in the dictionary
            for frame_num, frame_data in field_data.items():
                try:
                    reshaped_dict[frame_num] = frame_data.reshape(ny, nx)
                except Exception as e:
                    print(f"Error reshaping frame {frame_num} for {field_name}: {e}")
                    # Keep original data if reshaping fails
                    reshaped_dict[frame_num] = frame_data
            
            # Store the reshaped dictionary
            reshaped_fields[field_name] = reshaped_dict
        # Handle single arrays (not per-frame)
        elif isinstance(field_data, np.ndarray) and field_data.ndim == 1 and field_name != 'dimensions':
            try:
                reshaped_fields[field_name] = field_data.reshape(ny, nx)
            except Exception as e:
                print(f"Error reshaping {field_name}: {e}")
                # Keep original data if reshaping fails
                reshaped_fields[field_name] = field_data
        else:
            # For any other type of data, keep as is
            reshaped_fields[field_name] = field_data
    
    print(f"Reshaped {len(reshaped_fields)-1} fields (excluding dimensions)")
    
    # Apply vertical flip correction to the reshaped data
    print("\nApplying vertical flip orientation correction to reshaped data...")
    corrected_fields = {'dimensions': reshaped_fields['dimensions']}
    
    for field_name, field_data in reshaped_fields.items():
        if field_name == 'dimensions':
            continue
            
        # Handle 3D arrays (stacked frames)
        if isinstance(field_data, np.ndarray) and field_data.ndim == 3:
            # Create a new array with the same shape
            corrected_array = np.zeros_like(field_data)
            
            # Flip each frame
            for i in range(field_data.shape[0]):
                # Apply vertical flip
                corrected_array[i] = np.flipud(field_data[i])
                
                # Negate velocity components if needed
                if field_name in ['u_x', 'wss_x']:
                    corrected_array[i] = -corrected_array[i]
            
            corrected_fields[field_name] = corrected_array
            
        # Handle dictionary of frames
        elif isinstance(field_data, dict):
            corrected_dict = {}
            
            for frame_num, frame_data in field_data.items():
                if frame_data.ndim == 2:  # Only process 2D arrays
                    corrected = np.flipud(frame_data)
                    
                    # Negate velocity components if needed
                    if field_name in ['u_x', 'wss_x']:
                        corrected = -corrected
                        
                    corrected_dict[frame_num] = corrected
                else:
                    corrected_dict[frame_num] = frame_data
            
            corrected_fields[field_name] = corrected_dict
            
        # Handle 2D arrays (single frame)
        elif isinstance(field_data, np.ndarray) and field_data.ndim == 2:
            corrected = np.flipud(field_data)
            
            # Negate velocity components if needed
            if field_name in ['u_x', 'wss_x']:
                corrected = -corrected
                
            corrected_fields[field_name] = corrected
        else:
            # For any other type of data, keep as is
            corrected_fields[field_name] = field_data
    
    return corrected_fields


def save_stacked_fields_as_npz(fields_to_save, frame_to_index, output_dir=None, simulation_folder=None):
    """Save each field from stacked_fields as individual .npz files.
    
    Parameters:
    -----------
    fields_to_save : dict
        Dictionary containing the fields to save. These can be either the original stacked fields
        or reshaped fields, depending on what's passed in.
    frame_to_index : dict
        Mapping from frame numbers to indices in the arrays.
    output_dir : Path, optional
        Directory to save the .npz files. If None, will save to processed_data/raw_fields in simulation_folder.
    simulation_folder : Path, optional
        The simulation folder containing the VTK files.
    """
    # Create output directory based on simulation folder
    if output_dir is None and simulation_folder is not None:
        # Save to processed_data/raw_fields in the selected simulation folder
        output_dir = simulation_folder / 'processed_data' / 'raw_fields'
    elif output_dir is None:
        # Fallback to standard location
        base_dir = Path('/home/abdua786/code/uni/3/dissertation/dissertation')
        output_dir = base_dir / 'processed_data' / 'raw_fields'
    else:
        output_dir = Path(output_dir)
    
    # Create directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving fields to {output_dir}")
    
    # Convert frame_to_index to arrays for storage
    frame_numbers = np.array(list(frame_to_index.keys()))
    frame_indices = np.array(list(frame_to_index.values()))
    
    # Save frame mapping for reference
    frame_mapping_file = output_dir / 'frame_mapping.npz'
    np.savez_compressed(
        frame_mapping_file, 
        frame_numbers=frame_numbers, 
        frame_indices=frame_indices
    )
    print(f"Saved frame mapping to {frame_mapping_file}")
    
    # Store paths of saved files
    saved_files = {'frame_mapping': frame_mapping_file}
    
    # Also save the dimensions.npz file
    if 'dimensions' in fields_to_save:
        dimensions_file = output_dir / 'dimensions.npz'
        np.savez_compressed(dimensions_file, data=fields_to_save['dimensions'])
        saved_files['dimensions'] = dimensions_file
        print(f"Saved dimensions to {dimensions_file}")
    
    # Save each field as a separate npz file
    for field_name, field_data in fields_to_save.items():
        if field_name == 'dimensions':
            continue  # Already saved separately
            
        if isinstance(field_data, np.ndarray):
            # Create filename based on field name
            output_file = output_dir / f"{field_name}.npz"
            
            # Save as compressed npz file - each field gets its own file
            np.savez_compressed(output_file, data=field_data)
            
            # Record file size for reporting
            file_size = os.path.getsize(output_file) / (1024**2)  # size in MB
            saved_files[field_name] = output_file
            
            print(f"Saved field '{field_name}' to {output_file} ({file_size:.2f} MB)")
        else:
            print(f"Skipping field '{field_name}' as it's not a NumPy array")
    
    # Create a README file with information about the data
    readme_path = output_dir / "README.txt"
    with open(readme_path, 'w') as f:
        simulation_type = simulation_folder.name if simulation_folder is not None else "Simulation"
        f.write(f"{simulation_type} Field Data\n")
        f.write("="* (len(simulation_type) + 11) + "\n\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of frames: {len(frame_numbers)}\n")
        f.write("\nAvailable fields:\n")
        
        for field_name in fields_to_save.keys():
            if field_name in saved_files:
                field_data = fields_to_save[field_name]
                if isinstance(field_data, np.ndarray):
                    shape_str = str(field_data.shape)
                    f.write(f"- {field_name}: {shape_str}\n")
        
        f.write("\nNotes:\n")
        # Check if any of the fields are 3D arrays (frames, height, width)
        multi_dim = any(isinstance(data, np.ndarray) and data.ndim > 2 
                        for field_name, data in fields_to_save.items() 
                        if field_name != 'dimensions')
        if multi_dim:
            f.write("- These files contain 3D arrays with dimensions [frames, height, width]\n")
        else:
            f.write("- These files contain the raw field data without reshaping\n")
        f.write("- The frame_mapping.npz file contains the mapping between frame numbers and indices\n")
        f.write("- Each field is stored as a separate .npz file for easier loading\n")
        f.write("- To load a field: data = np.load('field_name.npz')['data']\n")
    
    print(f"\nSaved {len(saved_files)-2} fields to {output_dir}")
    print(f"Created README at {readme_path}")
    
    return saved_files


def find_vtk_collections(root_dir):
    """Function to find VTK collections recursively"""
    vtk_collections = []
    
    # Walk through all directories under the root
    for dirpath, dirnames, filenames in os.walk(root_dir):
        path = Path(dirpath)
        
        # Check if this directory contains VTK files
        vtk_files = list(path.glob('*.vtk'))
        if vtk_files:
            # Found a collection of VTK files
            vtk_collections.append({
                'vtk_dir': path,
                'config_dir': path.parent,  # Directory containing the VTK collection is the pipe config
                'vtk_files': vtk_files
            })
            print(f"Found VTK collection in: {path}")
            print(f"  Config directory: {path.parent}")
            print(f"  Number of VTK files: {len(vtk_files)}")
    
    return vtk_collections


def process_vtk_collection(collection, max_files_to_process=None):
    """Process a single VTK collection and convert its VTK files to NPZ format"""
    vtk_dir = collection['vtk_dir']
    config_dir = collection['config_dir']
    vtk_files = collection['vtk_files']
    
    print(f"\n{'='*80}")
    print(f"Processing VTK collection: {vtk_dir}")
    print(f"Config directory: {config_dir.name}")
    print(f"{'='*80}\n")
    
    # Create processed_data directory within the config directory
    processed_data_folder = config_dir / 'processed_data'
    processed_data_folder.mkdir(exist_ok=True, parents=True)
    
    # Sort VTK files
    vtk_files = sorted(vtk_files, key=natural_sort_key)
    if not vtk_files:
        print(f"Error: No VTK files found in {vtk_dir}")
        return False
    
    print(f"Found {len(vtk_files)} VTK files")
    
    # Extract simulation type (prefix) from the first VTK file
    filename = vtk_files[0].name
    match = re.match(r'([a-zA-Z_]+)\d+', filename)
    if match:
        simulation_prefix = match.group(1)
        print(f"Detected simulation type: {simulation_prefix}")
    else:
        simulation_prefix = "simulation_"
        print(f"Could not detect simulation prefix, using default: {simulation_prefix}")
    
    # Display some example filenames
    print("Example VTK files:")
    for file in vtk_files[:3]:  # Show first 3 files
        print(f"- {file.name}")
    if len(vtk_files) > 3:
        print(f"... and {len(vtk_files) - 3} more files")
    
    # Process the VTK files
    print(f"\nProcessing VTK files...")
    if max_files_to_process:
        print(f"Limiting to {max_files_to_process} files")
    
    # Process the files
    vtk_field_data, frame_numbers = process_all_vtk_files(vtk_files, max_files_to_process)
    
    # Print information about the processed data
    print(f"\nProcessed frames: {min(frame_numbers)} to {max(frame_numbers)}")
    
    # Check data dimensions
    for field_name, frames_data in vtk_field_data.items():
        if field_name != 'dimensions':
            # Get the shape of the first frame's data
            first_frame = frames_data[frame_numbers[0]]
            print(f"Field '{field_name}': {len(frames_data)} frames, shape per frame: {first_frame.shape}")
    
    # Stack the fields
    print("\nStacking fields into 3D arrays...")
    stacked_fields, frame_to_index = stack_fields_into_arrays(vtk_field_data, frame_numbers)
    
    # Print information about the stacked data
    print("\nStacked fields:")
    for field_name, field_array in stacked_fields.items():
        if isinstance(field_array, np.ndarray):
            if field_name != 'dimensions':
                print(f"- {field_name}: shape {field_array.shape}, type {field_array.dtype}, memory usage {field_array.nbytes / (1024**2):.2f} MB")
            else:
                print(f"- {field_name}: {field_array}")
        else:
            print(f"- {field_name}: dictionary with {len(field_array)} frames")
    
    # Reshape the fields
    print("\nReshaping all fields to 2D arrays...")
    reshaped_fields = reshape_all_fields_to_2d(stacked_fields, frame_to_index)
    
    # Print information about the reshaped data
    print("\nReshaped fields:")
    for field_name, field_array in reshaped_fields.items():
        if isinstance(field_array, np.ndarray):
            if field_name != 'dimensions':
                print(f"- {field_name}: shape {field_array.shape}, type {field_array.dtype}, memory usage {field_array.nbytes / (1024**2):.2f} MB")
            else:
                print(f"- {field_name}: {field_array}")
        else:
            print(f"- {field_name}: dictionary with {len(field_array)} frames")
    
    # Save the reshaped fields (3D arrays with proper dimensions) to NPZ files
    print("\nSaving reshaped fields as individual .npz files...")
    output_dir = processed_data_folder / 'raw_fields'
    print(f"Output directory: {output_dir}")
    saved_files = save_stacked_fields_as_npz(reshaped_fields, frame_to_index, output_dir, config_dir)
    
    # Also save the original stacked fields if needed (as backup)
    save_original = False  # Set to True if you want to save both versions
    if save_original:
        print("\nSaving original stacked fields (flattened) as individual .npz files...")
        original_output_dir = processed_data_folder / 'flattened_fields'
        print(f"Original output directory: {original_output_dir}")
        saved_original_files = save_stacked_fields_as_npz(stacked_fields, frame_to_index, original_output_dir, config_dir)
    
    print(f"\nProcessing completed for {config_dir.name}")
    print(f"NPZ files saved to: {output_dir}")
    
    return True


def process_all_vtk_collections(collections, max_files_to_process=None):
    """Process all VTK collections sequentially"""
    results = {}
    
    for collection in collections:
        config_name = collection['config_dir'].name
        start_time = time.time()
        success = process_vtk_collection(collection, max_files_to_process)
        end_time = time.time()
        results[config_name] = {
            'success': success,
            'processing_time': end_time - start_time
        }
    
    # Print summary of processing results
    print("\nProcessing Summary:")
    print("-" * 80)
    for config_name, result in results.items():
        status = "✓ Success" if result['success'] else "✗ Failed"
        time_taken = str(timedelta(seconds=int(result['processing_time'])))
        print(f"{status} - {config_name} - Time: {time_taken}")
    
    # Count successes and failures
    successes = sum(1 for result in results.values() if result['success'])
    failures = len(results) - successes
    print("-" * 80)
    print(f"Total: {len(results)} collections processed, {successes} succeeded, {failures} failed")
    
    return results


def main():
    """Main function to run the VTK processing pipeline"""
    parser = argparse.ArgumentParser(description='Process VTK files from LBM simulations')
    parser.add_argument('--max-files', type=int, default=None, 
                        help='Maximum number of files to process per collection (default: all)')
    parser.add_argument('--results-dir', type=str, default=None,
                        help='Results directory to search for VTK files (default: ../results)')
    
    args = parser.parse_args()
    
    # Get the current working directory and go one directory up
    current_directory = Path.cwd()
    if args.results_dir:
        parent_directory = Path(args.results_dir)
    else:
        parent_directory = current_directory.parent / 'results'

    print(f"Starting search from: {parent_directory}")
    
    # Find all VTK collections
    vtk_collections = find_vtk_collections(parent_directory)

    if not vtk_collections:
        print("Error: No VTK collections found.")
        return
    else:
        print(f"\nFound {len(vtk_collections)} VTK collections to process:")
        for i, collection in enumerate(vtk_collections):
            print(f"{i+1}. {collection['config_dir'].name} - {len(collection['vtk_files'])} files")

    # Process all VTK collections
    if len(vtk_collections) > 0:
        print(f"Starting automatic processing of {len(vtk_collections)} VTK collections")
        processing_results = process_all_vtk_collections(vtk_collections, args.max_files)
    else:
        print("No VTK collections found to process")

    print("\nProcessing complete!")


# Allow this module to be used both as a script and as a library
if __name__ == "__main__":
    main()
else:
    # When imported as a module, you can use the functions directly
    # Example: from vtk_processing import process_vtk_collection
    pass
