"""
File handling module for the VTK Player application
"""

import os
import sys
import glob
import re
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import pyvista as pv

def auto_find_vtk_files(self):
    """Automatically search for VTK files in predefined locations"""
    from utils.constants import DEFAULT_FOLDERS
    
    vtk_files = []
    found_folder = None
    
    # First check command line arguments for a folder path
    if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
        folder = sys.argv[1]
        files = glob.glob(os.path.join(folder, "*.vtk"))
        if files:
            vtk_files = files
            found_folder = folder
    
    # If no files found, try the predefined folders
    if not vtk_files:
        for folder in DEFAULT_FOLDERS:
            if os.path.isdir(folder):
                # Check this folder and all immediate subdirectories
                for root, dirs, files in os.walk(folder):
                    if root.count(os.sep) <= folder.count(os.sep) + 1:  # Only check immediate subdirectories
                        files = glob.glob(os.path.join(root, "*.vtk"))
                        if files:
                            vtk_files = files
                            found_folder = root
                            break
                    if vtk_files:
                        break
            if vtk_files:
                break
    
    # If we found files, load them
    if vtk_files:
        print(f"Found {len(vtk_files)} VTK files in {found_folder}")
        self.load_files(vtk_files)
        
        # Also update window title with folder name
        folder_name = os.path.basename(found_folder)
        if not folder_name:
            folder_name = os.path.basename(os.path.dirname(found_folder))
        self.setWindowTitle(f"VTK Animation Player - {folder_name}")
    else:
        # If no files found, show a message and make the open folder button more prominent
        self.open_folder_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 6px")
        print("No VTK files found in the default locations. Please use 'Open Folder' to select a directory.")

def open_files(self):
    """Open VTK files using a file dialog"""
    files, _ = QFileDialog.getOpenFileNames(
        self, 
        "Select VTK Files", 
        "", 
        "VTK Files (*.vtk)"
    )
    
    if files:
        self.load_files(files)

def open_folder(self):
    """Open a folder containing VTK files"""
    folder = QFileDialog.getExistingDirectory(
        self, 
        "Select Folder Containing VTK Files"
    )
    
    if folder:
        # Get all vtk files in the folder
        vtk_files = glob.glob(os.path.join(folder, "*.vtk"))
        self.load_files(vtk_files)

def load_files(self, files):
    """Load VTK files and prepare visualization"""
    if not files:
        return
        
    # Sort files numerically by their timestep
    def extract_timestep(filename):
        # Extract number from filename (assuming format like "aneurysm_0001000.vtk")
        match = re.search(r'_(\d+)\.vtk$', os.path.basename(filename))
        if match:
            return int(match.group(1))
        return 0
        
    self.vtk_files = sorted(files, key=extract_timestep)
    self.current_frame = 0
    
    # Update the slider
    self.frame_slider.setMaximum(len(self.vtk_files) - 1)
    self.frame_slider.setValue(0)
    
    # Update the frame label
    self.frame_label.setText(f"Frame: {self.current_frame + 1}/{len(self.vtk_files)}")
    
    # Load the first file to get available fields
    if self.vtk_files:
        try:
            mesh = pv.read(self.vtk_files[0])
            self.field_combo.clear()
            self.field_combo.addItems(mesh.array_names)
            
            # Determine if this is 2D data
            self.is_2d_data = mesh.bounds[5] - mesh.bounds[4] < 1e-6
            
            # Setup camera orientation based on data dimensions
            if self.is_2d_data:
                # For 2D data, set view to look at XY plane
                self.plotter.camera_position = 'xy'
                self.plotter.camera.zoom(1.2)
            else:
                # For 3D data, use a more typical isometric view
                self.plotter.camera_position = 'iso'
            
            if self.field_combo.count() > 0:
                # Set default field to u_magnitude if available (common for flow visualization)
                default_field_index = self.field_combo.findText("u_magnitude")
                if default_field_index >= 0:
                    self.field_combo.setCurrentIndex(default_field_index)
                
                # Display the first frame
                self.display_frame(self.current_frame)
        except Exception as e:
            QMessageBox.critical(self, "Error Loading Data", f"Error loading VTK file: {str(e)}")