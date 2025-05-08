"""
VTK Player Class Module - Main application class
"""

import os
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtCore import QTimer

# Import modules for functionality with updated paths
from utils.constants import DEFAULT_FOLDERS
from ui.ui_setup import setup_ui, setup_plotter
from core.file_handling import auto_find_vtk_files, open_files, open_folder, load_files
from core.playback_controls import toggle_play, stop, next_frame, prev_frame, slider_changed, set_fps
from core.visualization_logic import display_frame, update_visualization, change_field, change_colormap, toggle_auto_range

class VTKPlayer(QMainWindow):
    """
    Main window for the VTK animation player application.
    """
    def __init__(self):
        super().__init__()
        
        # Application settings
        self.setWindowTitle("VTK Animation Player - 2D Aneurysm Simulation")
        self.resize(1200, 800)
        
        # Data attributes
        self.vtk_files = []
        self.current_frame = 0
        self.playing = False
        self.fps = 10
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.is_2d_data = True  # Assume 2D data by default
        
        # Add method references to make the modules work
        self.setup_ui = setup_ui.__get__(self)
        self.setup_plotter = setup_plotter.__get__(self)
        self.auto_find_vtk_files = auto_find_vtk_files.__get__(self)
        self.open_files = open_files.__get__(self)
        self.open_folder = open_folder.__get__(self)
        self.load_files = load_files.__get__(self)
        self.toggle_play = toggle_play.__get__(self)
        self.stop = stop.__get__(self)
        self.next_frame = next_frame.__get__(self)
        self.prev_frame = prev_frame.__get__(self)
        self.slider_changed = slider_changed.__get__(self)
        self.set_fps = set_fps.__get__(self)
        self.display_frame = display_frame.__get__(self)
        self.update_visualization = update_visualization.__get__(self)
        self.change_field = change_field.__get__(self)
        self.change_colormap = change_colormap.__get__(self)
        self.toggle_auto_range = toggle_auto_range.__get__(self)
        
        # UI setup
        self.setup_ui()
        
        # PyVista plotter
        self.setup_plotter()
        
        # Automatically try to find and load VTK files
        self.auto_find_vtk_files()