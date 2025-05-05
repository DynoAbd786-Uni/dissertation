#!/usr/bin/env python3
import os
import sys
import glob
import re
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QSlider, QLabel, 
                             QComboBox, QFileDialog, QMessageBox)
from PyQt6.QtCore import Qt, QTimer, pyqtSlot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.cm as cm
import traceback


class VTKPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VTK File Player")
        self.setGeometry(100, 100, 1000, 800)
        
        # Data attributes
        self.vtk_files = []
        self.current_index = 0
        self.is_playing = False
        self.play_speed = 100  # ms between frames
        self.current_field = "u_magnitude"  # Default field to visualize
        self.available_fields = []
        self.cmap = "viridis"
        self.numpy_data = {}  # Will store numpy arrays from VTK files
        
        # Try both relative and absolute paths for VTK directory
        self.possible_vtk_dirs = [
            os.path.join(os.getcwd(), "aneurysm_simulation_results", "vtk"),
            os.path.join(os.getcwd(), "..", "aneurysm_simulation_results", "vtk"),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "aneurysm_simulation_results", "vtk")),
            "/home/abdua786/code/uni/3/dissertation/dissertation/aneurysm_simulation_results/vtk"
        ]
        
        self.default_vtk_dir = self.find_valid_vtk_dir()
        print(f"Default VTK directory: {self.default_vtk_dir}")
        
        # Setup UI
        self.setup_ui()
        
        # Create timer for animation
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        
        # If we have a default directory, try to load it
        if self.default_vtk_dir:
            print(f"Attempting to load VTK files from: {self.default_vtk_dir}")
            self.load_vtk_folder(self.default_vtk_dir)
        else:
            print("No valid VTK directory found automatically")
            # Show message to user
            QMessageBox.information(self, "VTK Directory Not Found", 
                                   "No VTK directory found automatically. Please use 'Load VTK Folder' to select the directory containing your VTK files.")
        
    def find_valid_vtk_dir(self):
        """Find the first valid VTK directory from the possible paths"""
        for path in self.possible_vtk_dirs:
            print(f"Checking path: {path}")
            if os.path.exists(path) and os.path.isdir(path):
                # Check if it contains VTK files
                vtk_files = glob.glob(os.path.join(path, "*.vtk"))
                if vtk_files:
                    print(f"Found {len(vtk_files)} VTK files in {path}")
                    return path
                else:
                    print(f"Directory exists but contains no VTK files: {path}")
            else:
                print(f"Directory does not exist: {path}")
        return None
    
    def setup_ui(self):
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        
        # Create matplotlib figure and canvas
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        
        # Add canvas to layout
        main_layout.addWidget(self.canvas)
        
        # Control panel layout
        control_layout = QHBoxLayout()
        
        # Add Open Folder button
        self.load_btn = QPushButton("Load VTK Folder")
        self.load_btn.clicked.connect(lambda: self.load_vtk_folder())
        control_layout.addWidget(self.load_btn)
        
        # Field selection
        self.field_combo = QComboBox()
        self.field_combo.currentTextChanged.connect(self.change_field)
        control_layout.addWidget(QLabel("Field:"))
        control_layout.addWidget(self.field_combo)
        
        # Colormap selection
        self.cmap_combo = QComboBox()
        for cmap_name in ['viridis', 'plasma', 'inferno', 'magma', 'jet', 'hot', 'cool']:
            self.cmap_combo.addItem(cmap_name)
        self.cmap_combo.currentTextChanged.connect(self.change_colormap)
        control_layout.addWidget(QLabel("Colormap:"))
        control_layout.addWidget(self.cmap_combo)
        
        # Debug button - added for troubleshooting
        self.debug_btn = QPushButton("Debug Info")
        self.debug_btn.clicked.connect(self.show_debug_info)
        control_layout.addWidget(self.debug_btn)
        
        # Add control layout to main layout
        main_layout.addLayout(control_layout)
        
        # Playback controls
        playback_layout = QHBoxLayout()
        
        # Previous button
        self.prev_btn = QPushButton("Previous")
        self.prev_btn.clicked.connect(self.prev_frame)
        playback_layout.addWidget(self.prev_btn)
        
        # Play/Pause button
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.toggle_play)
        playback_layout.addWidget(self.play_btn)
        
        # Next button
        self.next_btn = QPushButton("Next")
        self.next_btn.clicked.connect(self.next_frame)
        playback_layout.addWidget(self.next_btn)
        
        # Frame slider
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.valueChanged.connect(self.slider_changed)
        playback_layout.addWidget(self.slider)
        
        # Current frame label
        self.frame_label = QLabel("Frame: 0/0")
        playback_layout.addWidget(self.frame_label)
        
        # Speed control
        playback_layout.addWidget(QLabel("Speed:"))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setMinimum(10)  # 10ms between frames (fast)
        self.speed_slider.setMaximum(1000)  # 1000ms between frames (slow)
        self.speed_slider.setValue(self.play_speed)
        self.speed_slider.valueChanged.connect(self.change_speed)
        playback_layout.addWidget(self.speed_slider)
        
        # Status bar for information
        self.status_label = QLabel("Ready")
        playback_layout.addWidget(self.status_label)
        
        # Add playback controls to main layout
        main_layout.addLayout(playback_layout)
        
        # Set main widget
        self.setCentralWidget(main_widget)
        
        # Set initial message in plot area
        self.ax.text(0.5, 0.5, "No data loaded. Use 'Load VTK Folder' to select a directory with VTK files.",
                    ha='center', va='center', transform=self.ax.transAxes, fontsize=12)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.canvas.draw()
    
    def show_debug_info(self):
        """Show debug info in a message box"""
        info = [
            f"Current working directory: {os.getcwd()}",
            f"Default VTK directory: {self.default_vtk_dir}",
            f"Number of VTK files loaded: {len(self.vtk_files)}",
            f"Available fields: {', '.join(self.available_fields) if self.available_fields else 'None'}",
            f"Current field: {self.current_field}",
            f"Number of numpy arrays: {sum(len(arrays) for arrays in self.numpy_data.values()) if self.numpy_data else 0}"
        ]
        
        QMessageBox.information(self, "Debug Information", "\n".join(info))
    
    def natural_sort_key(self, s):
        """Sort strings with embedded numbers naturally"""
        return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]
    
    def load_vtk_folder(self, folder=None):
        """Load VTK files from folder and convert to numpy arrays"""
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, "Select VTK Files Directory", 
                                                     self.default_vtk_dir if self.default_vtk_dir else os.getcwd())
        
        if folder and os.path.exists(folder):
            self.status_label.setText(f"Loading VTK files from {folder}...")
            QApplication.processEvents()  # Update the UI
            
            # Find all VTK files in the folder
            vtk_pattern = os.path.join(folder, "*.vtk")
            self.vtk_files = sorted(glob.glob(vtk_pattern), key=self.natural_sort_key)
            
            print(f"Found {len(self.vtk_files)} VTK files in {folder}")
            
            if not self.vtk_files:
                self.status_label.setText(f"No VTK files found in {folder}")
                QMessageBox.warning(self, "No VTK Files", f"No VTK files found in {folder}")
                return
            
            # Process first file to get fields and initialize structures
            try:
                print(f"Processing first VTK file: {self.vtk_files[0]}")
                self.numpy_data = {}
                self.available_fields = []
                
                # Load first file to determine structure
                first_file_data = self.read_vtk_to_numpy(self.vtk_files[0])
                
                if not first_file_data:
                    self.status_label.setText("Error: First VTK file doesn't contain any data fields")
                    QMessageBox.critical(self, "Error Loading VTK", "First VTK file doesn't contain any data fields")
                    return
                
                print(f"Available fields in first file: {list(first_file_data.keys())}")
                
                # Set up data structures for all fields
                for field_name in first_file_data.keys():
                    self.available_fields.append(field_name)
                    self.numpy_data[field_name] = []
                
                # Process all files and convert to numpy arrays
                for i, vtk_file in enumerate(self.vtk_files):
                    self.status_label.setText(f"Processing file {i+1}/{len(self.vtk_files)}")
                    QApplication.processEvents()  # Update the UI
                    
                    data = self.read_vtk_to_numpy(vtk_file)
                    
                    # Store numpy arrays for each field
                    for field_name, array in data.items():
                        if field_name in self.numpy_data:
                            self.numpy_data[field_name].append(array)
                        else:
                            print(f"Warning: Field {field_name} not found in all files")
                
                # Setup UI based on loaded data
                self.setup_ui_for_data()
                
                # Load and display first frame
                self.current_index = 0
                self.load_frame(self.current_index)
                
                self.status_label.setText(f"Loaded {len(self.vtk_files)} VTK files")
            
            except Exception as e:
                error_text = f"Error loading VTK files: {str(e)}\n\n{traceback.format_exc()}"
                print(error_text)
                self.status_label.setText(f"Error: {str(e)}")
                QMessageBox.critical(self, "Error Loading VTK", error_text)
    
    def read_vtk_to_numpy(self, vtk_file_path):
        """Read a VTK file and convert its data to numpy arrays"""
        try:
            reader = vtk.vtkStructuredPointsReader()
            reader.SetFileName(vtk_file_path)
            reader.Update()
            data = reader.GetOutput()
            
            # Get dimensions for reshaping
            dimensions = data.GetDimensions()
            print(f"VTK dimensions: {dimensions}")
            
            # Get point data
            point_data = data.GetPointData()
            num_arrays = point_data.GetNumberOfArrays()
            print(f"Number of arrays in VTK: {num_arrays}")
            
            # If there are no arrays, try to use vtkUnstructuredGridReader
            if num_arrays == 0:
                print(f"No arrays found with vtkStructuredPointsReader, trying vtkUnstructuredGridReader")
                reader = vtk.vtkUnstructuredGridReader()
                reader.SetFileName(vtk_file_path)
                reader.Update()
                data = reader.GetOutput()
                point_data = data.GetPointData()
                num_arrays = point_data.GetNumberOfArrays()
                print(f"Number of arrays with vtkUnstructuredGridReader: {num_arrays}")
                
                # If still no arrays, try other readers
                if num_arrays == 0:
                    print(f"No arrays found with vtkUnstructuredGridReader, trying vtkPolyDataReader")
                    reader = vtk.vtkPolyDataReader()
                    reader.SetFileName(vtk_file_path)
                    reader.Update()
                    data = reader.GetOutput()
                    point_data = data.GetPointData()
                    num_arrays = point_data.GetNumberOfArrays()
                    print(f"Number of arrays with vtkPolyDataReader: {num_arrays}")
            
            # Convert all arrays to numpy
            result = {}
            for i in range(point_data.GetNumberOfArrays()):
                array_name = point_data.GetArrayName(i)
                vtk_array = point_data.GetArray(array_name)
                numpy_array = vtk_to_numpy(vtk_array)
                
                print(f"Found array '{array_name}' with shape {numpy_array.shape}")
                
                # Reshape to match the image dimensions (height, width)
                try:
                    reshaped_array = numpy_array.reshape(dimensions[1], dimensions[0])
                    result[array_name] = reshaped_array
                    print(f"  Reshaped to {reshaped_array.shape}")
                except ValueError as ve:
                    print(f"  Error reshaping array: {ve}")
                    # Try transposing
                    try:
                        reshaped_array = numpy_array.reshape(dimensions[0], dimensions[1]).T
                        result[array_name] = reshaped_array
                        print(f"  Reshaped with transpose to {reshaped_array.shape}")
                    except ValueError:
                        print(f"  Could not reshape to any valid dimensions, using original array")
                        result[array_name] = numpy_array
            
            return result
            
        except Exception as e:
            print(f"Error reading VTK file {vtk_file_path}: {str(e)}")
            print(traceback.format_exc())
            raise
    
    def setup_ui_for_data(self):
        """Update UI elements based on loaded data"""
        # Update field selection combo box
        self.field_combo.clear()
        for field in self.available_fields:
            self.field_combo.addItem(field)
        
        # Set default field if available
        if "u_magnitude" in self.available_fields:
            self.field_combo.setCurrentText("u_magnitude")
            self.current_field = "u_magnitude"
        elif self.available_fields:
            self.field_combo.setCurrentText(self.available_fields[0])
            self.current_field = self.available_fields[0]
        
        # Setup slider
        self.slider.setMaximum(len(self.vtk_files) - 1)
        self.frame_label.setText(f"Frame: 1/{len(self.vtk_files)}")
    
    def load_frame(self, index):
        """Display the specified frame"""
        if not self.numpy_data or not self.available_fields or index < 0 or index >= len(self.vtk_files):
            return
        
        try:
            # Update current index and slider
            self.current_index = index
            self.slider.setValue(index)
            self.frame_label.setText(f"Frame: {index+1}/{len(self.vtk_files)}")
            
            # Get the image data for the current field and index
            if self.current_field not in self.numpy_data:
                print(f"Error: Field {self.current_field} not found in numpy_data")
                return
                
            if index >= len(self.numpy_data[self.current_field]):
                print(f"Error: Index {index} out of range for field {self.current_field} (length: {len(self.numpy_data[self.current_field])})")
                return
                
            image_data = self.numpy_data[self.current_field][index]
            
            # Clear the plot
            self.ax.clear()
            
            # Plot the data
            im = self.ax.imshow(image_data, cmap=self.cmap, origin='lower')
            
            # Add a colorbar
            self.figure.colorbar(im, ax=self.ax)
            
            # Set title
            title = f"{os.path.basename(self.vtk_files[index])} - {self.current_field}"
            self.ax.set_title(title)
            
            # Add data range to status
            data_min = np.min(image_data)
            data_max = np.max(image_data)
            self.status_label.setText(f"Frame {index+1}/{len(self.vtk_files)} | {self.current_field} | Range: [{data_min:.4f}, {data_max:.4f}]")
            
            # Update the canvas
            self.canvas.draw()
            
        except Exception as e:
            error_text = f"Error displaying frame {index}: {str(e)}\n\n{traceback.format_exc()}"
            print(error_text)
            self.status_label.setText(f"Error: {str(e)}")
    
    @pyqtSlot()
    def next_frame(self):
        """Display the next frame"""
        if self.numpy_data:
            next_index = (self.current_index + 1) % len(self.vtk_files)
            self.load_frame(next_index)
    
    @pyqtSlot()
    def prev_frame(self):
        """Display the previous frame"""
        if self.numpy_data:
            prev_index = (self.current_index - 1) % len(self.vtk_files)
            self.load_frame(prev_index)
    
    @pyqtSlot()
    def toggle_play(self):
        """Toggle playback"""
        if not self.numpy_data:
            return
        
        self.is_playing = not self.is_playing
        
        if self.is_playing:
            self.play_btn.setText("Pause")
            self.timer.start(self.play_speed)
        else:
            self.play_btn.setText("Play")
            self.timer.stop()
    
    @pyqtSlot(int)
    def slider_changed(self, value):
        """Respond to slider changes"""
        if self.numpy_data and value != self.current_index:
            self.load_frame(value)
    
    @pyqtSlot(int)
    def change_speed(self, value):
        """Change playback speed"""
        self.play_speed = value
        if self.is_playing:
            self.timer.start(self.play_speed)
    
    @pyqtSlot(str)
    def change_field(self, field_name):
        """Change the visualization field"""
        self.current_field = field_name
        if self.numpy_data:
            self.load_frame(self.current_index)
    
    @pyqtSlot(str)
    def change_colormap(self, cmap_name):
        """Change the colormap"""
        self.cmap = cmap_name
        if self.numpy_data:
            self.load_frame(self.current_index)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VTKPlayer()
    window.show()
    sys.exit(app.exec())