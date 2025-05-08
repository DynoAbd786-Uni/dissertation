#!/usr/bin/env python3
"""
VTK Animation Player - Main Entry Point

This file serves as the entry point for the VTK Animation Player application.
"""

import sys
import os

# Configure GPU usage before importing PyVista components
os.environ["MESA_LOADER_DRIVER_OVERRIDE"] = "i965"  # Try dedicated GPU first
try:
    # For NVIDIA GPUs
    os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
    os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
except Exception:
    print("Could not set NVIDIA environment variables, falling back to integrated GPU")

# Print available GPU information (helpful for debugging)
print("GPU Information:")
try:
    import GPUtil
    gpus = GPUtil.getGPUs()
    if gpus:
        for i, gpu in enumerate(gpus):
            print(f"GPU {i}: {gpu.name} (Load: {gpu.load*100:.1f}%, Memory: {gpu.memoryUsed}/{gpu.memoryTotal} MB)")
    else:
        print("No dedicated GPUs detected")
except ImportError:
    print("GPUtil not installed, cannot detect GPUs")
except Exception as e:
    print(f"Error detecting GPUs: {e}")

import pyvista as pv
from PyQt5.QtWidgets import QApplication
from core.vtk_player_class import VTKPlayer  # Updated import path

# Configure PyVista rendering settings - using safer settings compatible with older versions
pv.global_theme.background = 'white'
pv.global_theme.show_scalar_bar = True

# Use try-except blocks for settings that might not be available in all versions
try:
    # For newer PyVista versions, anti_aliasing needs to be "ssaa", "msaa", "fxaa", or None
    pv.global_theme.anti_aliasing = "fxaa"  # Fast approximate anti-aliasing
except (AttributeError, TypeError) as e:
    print(f"Could not set anti-aliasing: {e}")
    # If that fails, try other approaches but don't break the application

def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    window = VTKPlayer()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()