#!/usr/bin/env python3
import os
import sys
import glob
import re
import numpy as np
import vtk
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QSlider, QLabel, 
                             QComboBox, QFileDialog, QMessageBox)
from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


class VTKNativePlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VTK Native Player")
        self.setGeometry(100, 100, 1200, 900)
        
        # Data attributes
        self.vtk_files = []
        self.current_index = 0
        self.is_playing = False
        self.play_speed = 100  # ms between frames
        self.current_field = "u_magnitude"  # Default field to visualize
        self.available_fields = []
        self.value_range = (0.0, 1.0)  # Default value range
        
        # VTK attributes
        self.vtk_data = None
        self.vtk_mapper = None
        self.vtk_actor = None
        
        # Search for VTK directory
        self.default_vtk_dir = self.find_vtk_directory()
        
        # Setup UI
        self.setup_ui()
        
        # Create timer for animation
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        
        # If default directory exists, load it
        if self.default_vtk_dir and os.path.exists(self.default_vtk_dir):
            print(f"Loading VTK files from {self.default_vtk_dir}")
            self.load_vtk_folder(self.default_vtk_dir)
        else:
            print("No default VTK directory found")
    
    def find_vtk_directory(self):
        """Find the directory containing VTK files"""
        possible_dirs = [
            os.path.join(os.getcwd(), "aneurysm_simulation_results", "vtk"),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "aneurysm_simulation_results", "vtk")),
            "/home/abdua786/code/uni/3/dissertation/dissertation/aneurysm_simulation_results/vtk"
        ]
        
        for path in possible_dirs:
            if os.path.exists(path) and os.path.isdir(path):
                vtk_files = glob.glob(os.path.join(path, "*.vtk"))
                if vtk_files:
                    return path
        
        return os.getcwd()
        
    def setup_ui(self):
        """Setup the user interface"""
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        
        # Create VTK widget
        self.vtk_widget = QVTKRenderWindowInteractor(main_widget)
        
        # Create VTK renderer
        self.renderer = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        
        # Add VTK widget to layout
        main_layout.addWidget(self.vtk_widget)
        
        # Control panel layout
        control_layout = QHBoxLayout()
        
        # Add Open Folder button
        self.load_btn = QPushButton("Load VTK Folder")
        self.load_btn.clicked.connect(lambda: self.load_vtk_folder())
        control_layout.addWidget(self.load_btn)
        
        # Field selection
        control_layout.addWidget(QLabel("Field:"))
        self.field_combo = QComboBox()
        self.field_combo.currentTextChanged.connect(self.change_field)
        control_layout.addWidget(self.field_combo)
        
        # Colormap selection
        control_layout.addWidget(QLabel("Colormap:"))
        self.colormap_combo = QComboBox()
        for colormap in ["Rainbow", "Jet", "HSV", "Hot", "Cool"]:
            self.colormap_combo.addItem(colormap)
        self.colormap_combo.currentTextChanged.connect(self.change_colormap)
        control_layout.addWidget(self.colormap_combo)
        
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
        
        # Add playback controls to main layout
        main_layout.addLayout(playback_layout)
        
        # Status bar
        self.status_label = QLabel("Ready")
        main_layout.addWidget(self.status_label)
        
        # Set central widget
        self.setCentralWidget(main_widget)
        
        # Initialize the interactor
        self.interactor.Initialize()
        
        # Add axes to the renderer
        axes = vtk.vtkAxesActor()
        self.renderer.AddActor(axes)
        
        # Set background color
        self.renderer.SetBackground(0.2, 0.2, 0.2)
        
        # Add colorbar
        self.setup_colorbar()
    
    def setup_colorbar(self):
        """Setup the colorbar for the visualization"""
        # Create a scalar bar (colorbar)
        self.scalar_bar = vtk.vtkScalarBarActor()
        self.scalar_bar.SetTitle("Value")
        self.scalar_bar.SetNumberOfLabels(5)
        self.scalar_bar.SetOrientationToHorizontal()
        self.scalar_bar.SetWidth(0.8)
        self.scalar_bar.SetHeight(0.1)
        self.scalar_bar.SetPosition(0.1, 0.01)
        
        # Add to renderer
        self.renderer.AddActor2D(self.scalar_bar)
    
    def natural_sort_key(self, s):
        """Sort strings with embedded numbers naturally"""
        return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]
    
    def load_vtk_folder(self, folder=None):
        """Load VTK files from a folder"""
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, "Select VTK Files Directory", self.default_vtk_dir)
        
        if folder and os.path.exists(folder):
            try:
                self.status_label.setText(f"Loading VTK files from {folder}...")
                QApplication.processEvents()
                
                # Find all VTK files
                vtk_pattern = os.path.join(folder, "*.vtk")
                self.vtk_files = sorted(glob.glob(vtk_pattern), key=self.natural_sort_key)
                
                if not self.vtk_files:
                    self.status_label.setText(f"No VTK files found in {folder}")
                    return
                
                # Load first file to get available fields
                self.load_vtk_fields(self.vtk_files[0])
                
                # Setup slider
                self.slider.setMaximum(len(self.vtk_files) - 1)
                
                # Load the first frame
                self.current_index = 0
                self.load_frame(self.current_index)
                
                self.status_label.setText(f"Loaded {len(self.vtk_files)} VTK files from {folder}")
            
            except Exception as e:
                self.status_label.setText(f"Error loading VTK files: {str(e)}")
                import traceback
                traceback.print_exc()
    
    def load_vtk_fields(self, vtk_file):
        """Load the fields available in the VTK file"""
        try:
            # Try different readers
            for reader_class in [vtk.vtkStructuredPointsReader, vtk.vtkUnstructuredGridReader, vtk.vtkPolyDataReader]:
                reader = reader_class()
                reader.SetFileName(vtk_file)
                reader.Update()
                data = reader.GetOutput()
                
                point_data = data.GetPointData()
                if point_data.GetNumberOfArrays() > 0:
                    # Found valid data
                    self.available_fields = []
                    self.field_combo.clear()
                    
                    # Get all fields
                    for i in range(point_data.GetNumberOfArrays()):
                        field_name = point_data.GetArrayName(i)
                        self.available_fields.append(field_name)
                        self.field_combo.addItem(field_name)
                    
                    # Set default field
                    if "u_magnitude" in self.available_fields:
                        self.field_combo.setCurrentText("u_magnitude")
                        self.current_field = "u_magnitude"
                    else:
                        self.field_combo.setCurrentText(self.available_fields[0])
                        self.current_field = self.available_fields[0]
                    
                    return
            
            raise Exception("No valid data found in VTK file")
        
        except Exception as e:
            self.status_label.setText(f"Error loading VTK fields: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def load_frame(self, index):
        """Load and display a specific frame"""
        if not self.vtk_files or index < 0 or index >= len(self.vtk_files):
            return
        
        try:
            self.status_label.setText(f"Loading frame {index+1}/{len(self.vtk_files)}...")
            QApplication.processEvents()
            
            # Update current index and slider
            self.current_index = index
            self.slider.setValue(index)
            self.frame_label.setText(f"Frame: {index+1}/{len(self.vtk_files)}")
            
            # Try different readers
            reader = None
            for reader_class in [vtk.vtkStructuredPointsReader, vtk.vtkUnstructuredGridReader, vtk.vtkPolyDataReader]:
                try:
                    reader = reader_class()
                    reader.SetFileName(self.vtk_files[index])
                    reader.Update()
                    data = reader.GetOutput()
                    
                    if data.GetPointData().GetNumberOfArrays() > 0:
                        break
                except:
                    continue
            
            if not reader:
                raise Exception("Could not read VTK file")
            
            # Get field data
            data = reader.GetOutput()
            data.GetPointData().SetActiveScalars(self.current_field)
            
            # Get value range for colormap
            scalar_range = data.GetScalarRange()
            self.value_range = scalar_range
            
            # Create new visualization if needed
            if not self.vtk_mapper:
                # For 2D data (structured points), use image actor
                if isinstance(reader, vtk.vtkStructuredPointsReader):
                    # Create a color lookup table
                    lookup_table = self.create_lookup_table()
                    
                    # Create image mapper
                    self.vtk_mapper = vtk.vtkImageMapToColors()
                    self.vtk_mapper.SetInputConnection(reader.GetOutputPort())
                    self.vtk_mapper.SetLookupTable(lookup_table)
                    
                    # Create image actor
                    self.vtk_actor = vtk.vtkImageActor()
                    self.vtk_actor.GetMapper().SetInputConnection(self.vtk_mapper.GetOutputPort())
                    
                    # Add actor to renderer
                    self.renderer.AddActor(self.vtk_actor)
                
                # For unstructured grid or polydata
                else:
                    self.vtk_mapper = vtk.vtkDataSetMapper()
                    self.vtk_mapper.SetInputConnection(reader.GetOutputPort())
                    self.vtk_mapper.SetScalarRange(scalar_range)
                    
                    # Create lookup table
                    lookup_table = self.create_lookup_table()
                    self.vtk_mapper.SetLookupTable(lookup_table)
                    
                    # Create actor
                    self.vtk_actor = vtk.vtkActor()
                    self.vtk_actor.SetMapper(self.vtk_mapper)
                    
                    # Add actor to renderer
                    self.renderer.AddActor(self.vtk_actor)
            
            # Update existing visualization
            else:
                # For 2D data
                if isinstance(reader, vtk.vtkStructuredPointsReader) and isinstance(self.vtk_actor, vtk.vtkImageActor):
                    self.vtk_mapper.SetInputConnection(reader.GetOutputPort())
                    
                    # Update lookup table
                    lookup_table = self.create_lookup_table()
                    self.vtk_mapper.SetLookupTable(lookup_table)
                
                # For unstructured grid or polydata
                else:
                    self.vtk_mapper.SetInputConnection(reader.GetOutputPort())
                    self.vtk_mapper.SetScalarRange(scalar_range)
                    
                    # Update lookup table
                    lookup_table = self.create_lookup_table()
                    self.vtk_mapper.SetLookupTable(lookup_table)
            
            # Update scalar bar
            self.scalar_bar.SetLookupTable(lookup_table)
            self.scalar_bar.SetTitle(self.current_field)
            
            # Update camera view for first frame
            if index == 0:
                self.renderer.ResetCamera()
            
            # Render view
            self.vtk_widget.GetRenderWindow().Render()
            
            self.status_label.setText(f"Displaying frame {index+1}/{len(self.vtk_files)} - Field: {self.current_field}")
        
        except Exception as e:
            self.status_label.setText(f"Error loading frame: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def create_lookup_table(self):
        """Create a color lookup table for the visualization"""
        lut = vtk.vtkLookupTable()
        
        colormap = self.colormap_combo.currentText()
        
        # Set colormap
        if colormap == "Rainbow":
            lut.SetHueRange(0.667, 0.0)  # Blue to red
        elif colormap == "Jet":
            lut.SetNumberOfColors(256)
            for i in range(256):
                if i < 64:  # Blue to cyan
                    r, g, b = 0, i*4/255, 1
                elif i < 128:  # Cyan to green
                    r, g, b = 0, 1, 1-(i-64)*4/255
                elif i < 192:  # Green to yellow
                    r, g, b = (i-128)*4/255, 1, 0
                else:  # Yellow to red
                    r, g, b = 1, 1-(i-192)*4/255, 0
                lut.SetTableValue(i, r, g, b, 1.0)
        elif colormap == "Hot":
            lut.SetHueRange(0.0, 0.167)  # Red to yellow
        elif colormap == "Cool":
            lut.SetHueRange(0.5, 0.833)  # Cyan to purple
        else:  # HSV
            lut.SetHueRange(0.0, 1.0)  # Full rainbow
        
        lut.SetRange(self.value_range)
        lut.Build()
        
        return lut
    
    @pyqtSlot()
    def next_frame(self):
        """Display the next frame"""
        if self.vtk_files:
            next_index = (self.current_index + 1) % len(self.vtk_files)
            self.load_frame(next_index)
    
    @pyqtSlot()
    def prev_frame(self):
        """Display the previous frame"""
        if self.vtk_files:
            prev_index = (self.current_index - 1) % len(self.vtk_files)
            self.load_frame(prev_index)
    
    @pyqtSlot()
    def toggle_play(self):
        """Toggle playback"""
        if not self.vtk_files:
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
        if self.vtk_files and value != self.current_index:
            self.load_frame(value)
    
    @pyqtSlot(int)
    def change_speed(self, value):
        """Change playback speed"""
        self.play_speed = value
        if self.is_playing:
            self.timer.start(self.play_speed)
    
    @pyqtSlot(str)
    def change_field(self, field_name):
        """Change the active scalar field"""
        if field_name and field_name != self.current_field:
            self.current_field = field_name
            if self.vtk_files:
                self.load_frame(self.current_index)
    
    @pyqtSlot(str)
    def change_colormap(self, colormap_name):
        """Change the colormap"""
        if self.vtk_files:
            self.load_frame(self.current_index)
    
    def closeEvent(self, event):
        """Clean up VTK objects when closing"""
        self.vtk_widget.GetRenderWindow().Finalize()
        self.interactor.TerminateApp()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VTKNativePlayer()
    window.show()
    sys.exit(app.exec())