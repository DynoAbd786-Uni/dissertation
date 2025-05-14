"""
Utility functions for directory operations used in simulation scripts.
"""
import os
import shutil


def delete_directory_if_exists(directory_path):
    """
    Delete a directory and all its contents if it exists
    
    Args:
        directory_path (str): Path to the directory to delete
    """
    if os.path.exists(directory_path):
        print(f"Deleting existing output directory: {directory_path}")
        try:
            shutil.rmtree(directory_path)
            print(f"Successfully deleted {directory_path}")
        except Exception as e:
            print(f"Warning: Failed to delete directory {directory_path}: {e}")