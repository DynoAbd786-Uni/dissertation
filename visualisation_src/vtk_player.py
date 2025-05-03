#!/usr/bin/env python3
"""
VTK Animation Player

A PyQt and PyVista-based application for visualizing VTK files as a video with playback controls.
"""

import os
import sys
import glob
from pathlib import Path
import re
import numpy as np
import pyvista as pv
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QSlider, QFileDialog, QLabel, QComboBox, QCheckBox,
    QSpinBox, QDoubleSpinBox, QStyle, QGridLayout, QGroupBox
)
from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtGui import QIcon

class VTKPlayer(QMainWindow):
    """
    Main window for the VTK animation player application.
    """
    def __init__(self):
        super().__init__()
        
        # Application settings
        self.setWindowTitle("VTK Animation Player")
        self.resize(1200, 800)
        
        # Data attributes
        self.vtk_files = []
        self.current_frame = 0
        self.playing = False
        self.fps = 10
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        
        # UI setup
        self.setup_ui()
        
        # PyVista plotter
        self.setup_plotter()
        
    def setup_ui(self):
        """Set up the user interface"""
        # Main layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        
        # Create visualization area (placeholder for PyVista)
        self.vis_widget = QWidget()
        main_layout.addWidget(self.vis_widget, 1)  # 1 is the stretch factor
        
        # Control panel
        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        
        # Playback controls
        playback_group = QGroupBox("Playback Controls")
        playback_layout = QHBoxLayout()
        
        # Open file/folder buttons
        self.open_file_btn = QPushButton("Open Files")
        self.open_file_btn.clicked.connect(self.open_files)
        self.open_folder_btn = QPushButton("Open Folder")
        self.open_folder_btn.clicked.connect(self.open_folder)
        
        # Playback buttons with icons
        self.play_btn = QPushButton()
        self.play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_btn.clicked.connect(self.toggle_play)
        
        self.stop_btn = QPushButton()
        self.stop_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
        self.stop_btn.clicked.connect(self.stop)
        
        self.prev_btn = QPushButton()
        self.prev_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaSkipBackward))
        self.prev_btn.clicked.connect(self.prev_frame)
        
        self.next_btn = QPushButton()
        self.next_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaSkipForward))
        self.next_btn.clicked.connect(self.next_frame)
        
        # Frame slider
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.valueChanged.connect(self.slider_changed)
        
        # Frame counter
        self.frame_label = QLabel("Frame: 0/0")
        
        # FPS control
        fps_label = QLabel("FPS:")
        self.fps_spinbox = QSpinBox()
        self.fps_spinbox.setRange(1, 60)
        self.fps_spinbox.setValue(self.fps)
        self.fps_spinbox.valueChanged.connect(self.set_fps)
        
        # Add widgets to playback layout
        playback_layout.addWidget(self.open_file_btn)
        playback_layout.addWidget(self.open_folder_btn)
        playback_layout.addWidget(self.prev_btn)
        playback_layout.addWidget(self.play_btn)
        playback_layout.addWidget(self.stop_btn)
        playback_layout.addWidget(self.next_btn)
        playback_layout.addWidget(fps_label)
        playback_layout.addWidget(self.fps_spinbox)
        
        playback_group.setLayout(playback_layout)
        controls_layout.addWidget(playback_group)
        
        # Frame navigation
        frame_group = QGroupBox("Frame Navigation")
        frame_layout = QHBoxLayout()
        frame_layout.addWidget(self.frame_slider)
        frame_layout.addWidget(self.frame_label)
        frame_group.setLayout(frame_layout)
        controls_layout.addWidget(frame_group)
        
        # Visualization settings
        vis_group = QGroupBox("Visualization Settings")
        vis_layout = QGridLayout()
        
        # Field selection
        field_label = QLabel("Display Field:")
        self.field_combo = QComboBox()
        self.field_combo.currentTextChanged.connect(self.change_field)
        
        # Color map selection
        colormap_label = QLabel("Color Map:")
        self.colormap_combo = QComboBox()
        # Add some common colormaps
        colormaps = ["viridis", "plasma", "inferno", "magma", "cividis", 
                    "turbo", "jet", "rainbow", "ocean", "hot", "cool"]
        self.colormap_combo.addItems(colormaps)
        self.colormap_combo.currentTextChanged.connect(self.change_colormap)
        
        # Add range control
        range_min_label = QLabel("Min Value:")
        self.range_min = QDoubleSpinBox()
        self.range_min.setRange(-1000, 1000)
        self.range_min.setValue(0)
        self.range_min.valueChanged.connect(self.update_visualization)
        
        range_max_label = QLabel("Max Value:")
        self.range_max = QDoubleSpinBox()
        self.range_max.setRange(-1000, 1000)
        self.range_max.setValue(1)
        self.range_max.valueChanged.connect(self.update_visualization)
        
        # Auto range checkbox
        self.auto_range = QCheckBox("Auto Range")
        self.auto_range.setChecked(True)
        self.auto_range.stateChanged.connect(self.toggle_auto_range)
        
        # Add widgets to visualization layout
        vis_layout.addWidget(field_label, 0, 0)
        vis_layout.addWidget(self.field_combo, 0, 1)
        vis_layout.addWidget(colormap_label, 1, 0)
        vis_layout.addWidget(self.colormap_combo, 1, 1)
        vis_layout.addWidget(range_min_label, 2, 0)
        vis_layout.addWidget(self.range_min, 2, 1)
        vis_layout.addWidget(range_max_label, 3, 0)
        vis_layout.addWidget(self.range_max, 3, 1)
        vis_layout.addWidget(self.auto_range, 4, 0, 1, 2)
        
        vis_group.setLayout(vis_layout)
        controls_layout.addWidget(vis_group)
        
        # Add controls to main layout
        main_layout.addWidget(controls)
        
        # Set main widget
        self.setCentralWidget(main_widget)
    
    def setup_plotter(self):
        """Set up the PyVista plotter"""
        # Create the PyVista widget and add it to our widget
        self.plotter = pv.QtInteractor(self.vis_widget)
        layout = QVBoxLayout(self.vis_widget)
        layout.addWidget(self.plotter.interactor)
        
        # Set background color
        self.plotter.set_background("white")
        
        # Add a text display for the frame number in the top left
        self.frame_text = self.plotter.add_text("Frame: 0/0", position="upper_left", font_size=12, color="black")
        
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
            mesh = pv.read(self.vtk_files[0])
            self.field_combo.clear()
            self.field_combo.addItems(mesh.array_names)
            
            if self.field_combo.count() > 0:
                # Display the first field by default
                self.display_frame(self.current_frame)
    
    def display_frame(self, frame_idx):
        """Display the specified VTK frame"""
        if not self.vtk_files or frame_idx < 0 or frame_idx >= len(self.vtk_files):
            return
            
        # Load the mesh for the current frame
        mesh = pv.read(self.vtk_files[frame_idx])
        
        # Clear the previous plot
        self.plotter.clear()
        
        # Check if we have a field to display
        if self.field_combo.currentText() and self.field_combo.currentText() in mesh.array_names:
            field = self.field_combo.currentText()
            
            # Get colormap name
            cmap = self.colormap_combo.currentText()
            
            # Determine range
            if self.auto_range.isChecked():
                rng = mesh.get_data_range(field)
                self.range_min.setValue(rng[0])
                self.range_max.setValue(rng[1])
            else:
                rng = (self.range_min.value(), self.range_max.value())
            
            # Add the mesh to the plotter with the selected field
            self.plotter.add_mesh(mesh, scalars=field, cmap=cmap, clim=rng, show_scalar_bar=True)
        else:
            # If no field is available, just show the mesh
            self.plotter.add_mesh(mesh, color="lightblue")
        
        # Update the frame text
        self.plotter.add_text(f"Frame: {frame_idx + 1}/{len(self.vtk_files)}", 
                            position="upper_left", font_size=12, color="black")
        
        # Update the view
        self.plotter.reset_camera()
        self.plotter.update()
        
        # Update frame counter
        self.frame_label.setText(f"Frame: {frame_idx + 1}/{len(self.vtk_files)}")
    
    def next_frame(self):
        """Go to the next frame"""
        if not self.vtk_files:
            return
            
        self.current_frame = (self.current_frame + 1) % len(self.vtk_files)
        self.frame_slider.setValue(self.current_frame)
        self.display_frame(self.current_frame)
    
    def prev_frame(self):
        """Go to the previous frame"""
        if not self.vtk_files:
            return
            
        self.current_frame = (self.current_frame - 1) % len(self.vtk_files)
        self.frame_slider.setValue(self.current_frame)
        self.display_frame(self.current_frame)
    
    def slider_changed(self, value):
        """Handle slider value change"""
        if value != self.current_frame:
            self.current_frame = value
            self.display_frame(self.current_frame)
    
    def toggle_play(self):
        """Toggle playback state"""
        self.playing = not self.playing
        
        if self.playing:
            self.play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
            self.timer.start(1000 // self.fps)  # milliseconds
        else:
            self.play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            self.timer.stop()
    
    def stop(self):
        """Stop playback and reset to first frame"""
        self.playing = False
        self.play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.timer.stop()
        
        self.current_frame = 0
        self.frame_slider.setValue(0)
        self.display_frame(0)
    
    def set_fps(self, fps):
        """Set frames per second for playback"""
        self.fps = fps
        if self.playing:
            self.timer.stop()
            self.timer.start(1000 // self.fps)  # milliseconds
    
    def change_field(self, field_name):
        """Change the displayed field"""
        self.display_frame(self.current_frame)
    
    def change_colormap(self, cmap_name):
        """Change the color map"""
        self.display_frame(self.current_frame)
    
    def toggle_auto_range(self, state):
        """Handle auto range checkbox state change"""
        # Enable/disable manual range inputs based on auto range state
        manual_enabled = not bool(state)
        self.range_min.setEnabled(manual_enabled)
        self.range_max.setEnabled(manual_enabled)
        
        # Update visualization
        self.display_frame(self.current_frame)
    
    def update_visualization(self):
        """Update visualization with current settings"""
        self.display_frame(self.current_frame)

def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    window = VTKPlayer()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()