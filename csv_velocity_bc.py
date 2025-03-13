from custom_time_depenadant_zouhe_bc_class import TimeDependentZouHeBC
from xlb.compute_backend import ComputeBackend
from xlb.operator import Operator
import warp as wp
import numpy as np
import jax.numpy as jnp
from jax import jit, lax
from functools import partial
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, Union, List, Dict, Optional
import os

class CSVVelocityZouHeBC(TimeDependentZouHeBC):
    """
    ZouHe boundary condition that loads velocity profiles directly from CSV files.
    This class extends TimeDependentZouHeBC to provide easier CSV-based workflows.
    """
    
    def __init__(self, 
             bc_type: str, 
             csv_filepath: str,
             column_name: str = None,
             time_column: str = 'time',
             speed_multiplier: float = 1.0,
             dt: float = 0.01,
             dx: float = 1e-3,  # Physical length of one lattice unit (m)
             period: float = None,
             num_segments: int = 10,
             vessel_radius_mm: float = None, 
             convert_to_lattice: bool = True,  # Whether to convert to lattice units
             indices=None, 
             **kwargs):
        """Initialize the boundary condition with data from a CSV file."""
        # Store parameters
        self.csv_filepath = csv_filepath
        self.column_name = column_name
        self.time_column = time_column
        self.speed_multiplier = speed_multiplier
        self.dt = dt
        self.dx = dx
        self.user_period = period
        self.num_segments = num_segments
        
        # Load and process CSV data
        self.processed_data = self._load_csv_data()
        
        # Convert from ml/s to appropriate units if vessel radius is provided
        if vessel_radius_mm is not None:
            if convert_to_lattice:
                # Convert from ml/s to lattice units/step
                self.processed_data = self.convert_flow_to_lattice_velocity(vessel_radius_mm, dx, dt)
            else:
                # Convert from ml/s to m/s only
                self.processed_data = self.convert_flow_to_velocity(vessel_radius_mm)
        
        # Initialize current velocity value arrays for WARP and JAX
        self.current_velocity = 0.0
        self.warp_velocity = wp.zeros(1, dtype=wp.float32)
        
        # Create velocity profile function based on the data
        if bc_type == "velocity":
            profile_func = self._create_velocity_profile
        elif bc_type == "pressure":
            profile_func = self._create_pressure_profile
        else:
            raise ValueError(f"bc_type must be 'velocity' or 'pressure', got '{bc_type}'")
        
        # Call parent constructor with the generated profile function
        super().__init__(bc_type, indices=indices, **kwargs)
        
        print(f"Created CSV-based {bc_type} boundary condition from {os.path.basename(csv_filepath)}")
        print(f"Profile period: {self.processed_data['period']:.4f}s")

            
    def _construct_kernel(self, functional):
        """Override with timestep-aware kernel for WARP"""
        _id = wp.uint8(self.id)
        _q = self.velocity_set.q
        
        # Get a reference to the velocity array that will be updated each timestep
        _velocity_array = self.warp_velocity
        
        @wp.kernel
        def kernel(
            f_pre: wp.array(dtype=self.compute_dtype),
            f_post: wp.array(dtype=self.compute_dtype),
            bc_mask: wp.array(dtype=wp.uint8),
            missing_mask: wp.array(dtype=wp.uint8),
        ):
            i, j, k = wp.tid()
            
            # Check if this is a boundary cell for this BC
            if bc_mask[i, j, k] == _id:
                # Create copies of the distribution functions at this cell
                _f_pre = wp.array(shape=_q, dtype=self.compute_dtype)
                _f_post = wp.array(shape=_q, dtype=self.compute_dtype)
                _missing_mask = wp.array(shape=_q, dtype=wp.uint8)
                
                # Copy values for this cell
                for q in range(_q):
                    _f_pre[q] = f_pre[q, i, j, k]
                    _f_post[q] = f_post[q, i, j, k]
                    _missing_mask[q] = missing_mask[q, i, j, k]
                
                # Get the current velocity value from the global array 
                current_timestep = wp.int32(0)  # Not used since velocity is pre-calculated
                current_velocity = _velocity_array[0]
                
                # Apply the boundary condition
                _f_out = functional(
                    wp.vec3i(i, j, k),
                    current_timestep,
                    current_velocity,  # Pass velocity directly here
                    _missing_mask,
                    f_pre,
                    f_post,
                    _f_pre,
                    _f_post,
                )
                
                # Copy back the modified values
                for q in range(_q):
                    f_post[q, i, j, k] = _f_out[q]
        
        return kernel        

    # Add this method to create a WARP-compatible functional for auxiliary data
    def _create_aux_data_func(self):
        """Create WARP-compatible functional for auxiliary data"""
        # This function will be used during initialization only
        @wp.func
        def aux_func(index: wp.vec3i):
            # Return zero velocity vector - will be updated later by update_timestep
            return wp.vec(0.0, length=1)
        return aux_func
        
    # Override initialization of auxiliary data
    @Operator.register_backend(ComputeBackend.WARP)
    def auxiliary_data_initialization(self):
        """Override auxiliary data initialization to use our function"""
        # Create our simple auxiliary data
        if self.indices is None:
            return wp.zeros(1, dtype=wp.vec(length=1))
            
        if len(self.indices.shape) == 1:
            aux_data = wp.zeros(self.indices.shape[0], dtype=wp.vec(length=1))
        else:
            aux_data = wp.zeros((self.indices.shape[0], self.indices.shape[1]), 
                             dtype=wp.vec(length=1))
                             
        # Create custom kernel for initialization
        @wp.kernel
        def init_aux_data_kernel(aux_data: wp.array(dtype=wp.vec(length=1))):
            i = wp.tid()
            # Just initialize with zero - will be set properly by update_timestep
            aux_data[i] = wp.vec(0.0, length=1)
            
        # Launch the kernel
        if aux_data.ndim == 1:
            wp.launch(init_aux_data_kernel, dim=aux_data.shape[0], inputs=[aux_data])
        else:
            # Need a different kernel for 2D array
            @wp.kernel
            def init_aux_data_kernel_2d(
                aux_data: wp.array2d(dtype=wp.vec(length=1))
            ):
                i, j = wp.tid()
                aux_data[i, j] = wp.vec(0.0, length=1)
                
            wp.launch(init_aux_data_kernel_2d, dim=aux_data.shape, inputs=[aux_data])
            
        return aux_data
        
    # Completely override the base class's aux_data_init method
    def aux_data_init(self, f_0, f_1, bc_mask, missing_mask):
        """Completely bypass the standard initialization process"""
        # Just return the input fields unchanged
        return f_0, f_1

    def update_timestep(self, timestep):
        """
        Override to update both timestep and pre-calculate velocity value
        """
        # Update the parent's timestep tracking
        super().update_timestep(timestep)
        
        # Pre-calculate velocity for this timestep
        self.current_velocity = self.calculate_velocity_at_timestep(timestep)
        
        # Update WARP array with current velocity
        temp_array = wp.array([float(self.current_velocity)], dtype=wp.float32)
        wp.copy(self.warp_velocity, temp_array)
        
        # Debug output if needed
        # print(f"Step {timestep}: Velocity = {self.current_velocity:.6f}")
        # Print periodic debug info
        prev_velocity = getattr(self, 'current_velocity', None)
        
        if timestep % 5000 == 0:
            print(f"Step {timestep}: Velocity = {self.current_velocity:.6f} LU/step")
            print(f"  Time: {timestep * self.dt:.4f}s, Period: {self.processed_data['period']:.4f}s")
            if prev_velocity is not None and abs(self.current_velocity - prev_velocity) < 1e-10:
                print("  WARNING: Velocity not changing significantly!")

    # DIFFERENCE: Override warp_implementation is identical in structure to original
    # But relies on custom kernel with timestep awareness
    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f_pre, f_post, bc_mask, missing_mask):
        wp.launch(
            self.warp_kernel,
            inputs=[f_pre, f_post, bc_mask, missing_mask],
            dim=f_pre.shape[1:],
        )
        return f_post
    
    def _load_csv_data(self) -> Dict:
        """Load and process CSV data"""
        try:
            # Read the CSV file
            data = pd.read_csv(self.csv_filepath)
            
            # Get column names
            if self.column_name is None:
                # If column name not specified, use the second column (assuming first is time)
                columns = data.columns.tolist()
                if len(columns) < 2:
                    raise ValueError("CSV file must have at least 2 columns")
                self.column_name = columns[1]
                print(f"Using column '{self.column_name}' for values")
            
            # Extract time and value columns
            if self.time_column not in data.columns:
                raise ValueError(f"Time column '{self.time_column}' not found in CSV")
            if self.column_name not in data.columns:
                raise ValueError(f"Value column '{self.column_name}' not found in CSV")
                
            # Extract values as numpy arrays
            time_values = data[self.time_column].values.astype(np.float32)
            value_values = data[self.column_name].values.astype(np.float32)
            
            # Clean data (remove NaNs)
            valid_mask = ~(np.isnan(time_values) | np.isnan(value_values))
            time_clean = time_values[valid_mask]
            value_clean = value_values[valid_mask]
            
            if len(time_clean) < 2:
                raise ValueError("Not enough valid data points after cleaning")
                
            # Calculate cycle parameters
            t_min = float(time_clean[0])
            t_max = float(time_clean[-1])
            period = self.user_period if self.user_period is not None else t_max - t_min
            
            # Create piecewise linear segments for WARP
            segment_positions = np.linspace(0, 1.0, self.num_segments + 1)
            segment_times = t_min + segment_positions * period
            segment_values = np.interp(segment_times, time_clean, value_clean)
            
            # Return processed data
            return {
                'time_values': time_clean,
                'value_values': value_clean,
                't_min': t_min,
                't_max': t_max,
                'period': period,
                'segment_positions': segment_positions.tolist(),
                'segment_values': segment_values.tolist(),
                'num_segments': self.num_segments
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise ValueError(f"Error processing CSV file: {str(e)}")
    
    def _create_velocity_profile(self):
        """Create a velocity profile function from the CSV data"""
        # Core parameters - velocity will be updated via current_velocity/warp_velocity
        
        # WARP implementation using pre-calculated velocity value
        @wp.func
        def velocity_warp(index: wp.vec3i, timestep: int = 0):
            """WARP implementation that uses the pre-calculated velocity"""
            # Simply use the pre-calculated velocity value
            velocity = self.warp_velocity[0]
            
            # Return in the same format as your example - using length parameter
            return wp.vec(velocity, length=1)
        
        # JAX implementation with correct return format
        def velocity_jax():
            def velocity(timestep):
                # For JAX, we use the current_velocity value directly
                u_x = self.current_velocity
                
                # Match your example format exactly - return as [u_x, u_y] for 2D
                u_y = jnp.zeros_like(u_x) if isinstance(u_x, jnp.ndarray) else 0.0
                return jnp.array([u_x, u_y])
            return velocity
            
        # Return based on compute backend
        if self.compute_backend == ComputeBackend.JAX:
            return velocity_jax
        else:
            return velocity_warp
    
    def _create_pressure_profile(self):
        """Create a pressure profile function from the CSV data"""
        # Similar to velocity profile but returns pressure/density instead
        # Implementation is similar but adapted for pressure boundary conditions
        # For brevity, focusing on the velocity case in this example
        raise NotImplementedError("Pressure profiles not yet implemented")
    
    def plot_profile(self, num_seconds=2.0, num_points=1000):
        """Visualize the CSV velocity profile"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot original data points
        ax.scatter(self.processed_data['time_values'], 
                  self.processed_data['value_values'] * self.speed_multiplier, 
                  label="Original data", alpha=0.7, s=20, color='red')
        
        # Plot piecewise segments
        segment_times = self.processed_data['t_min'] + np.array(self.processed_data['segment_positions']) * self.processed_data['period']
        segment_values = np.array(self.processed_data['segment_values']) * self.speed_multiplier
        ax.plot(segment_times, segment_values, 'o-', 
               label=f"Piecewise approx ({self.num_segments} segments)",
               color='blue', linewidth=2)
        
        # Generate a high-resolution visualization
        times = np.linspace(self.processed_data['t_min'], 
                          self.processed_data['t_min'] + num_seconds, 
                          num_points)
        values = []
        
        # Use numpy for visualization - simulate the profile behavior
        for t in times:
            t_cycle = self.processed_data['t_min'] + (
                (t - self.processed_data['t_min']) % self.processed_data['period'])
            v = np.interp(t_cycle, 
                         self.processed_data['time_values'], 
                         self.processed_data['value_values'])
            values.append(float(v * self.speed_multiplier))
        
        # Plot the interpolated profile
        ax.plot(times, values, '-', linewidth=1.5,
               label="Interpolated profile", color='green', alpha=0.8)
        
        # Add cycle boundary markers
        for i in range(int(num_seconds / self.processed_data['period']) + 1):
            cycle_time = self.processed_data['t_min'] + i * self.processed_data['period']
            if cycle_time <= times[-1]:
                ax.axvline(cycle_time, color='gray', linestyle='--', alpha=0.5)
                if i > 0:
                    ax.text(cycle_time, 0, f"Cycle {i}", 
                           ha='center', va='bottom', alpha=0.7, fontsize=8)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'{self.bc_type.capitalize()} magnitude')
        ax.set_title(f'CSV {self.bc_type} profile: {self.column_name}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        return fig, ax

    def calculate_velocity_at_timestep(self, timestep):
        """
        Calculate the velocity value for a given timestep.
        
        Args:
            timestep: The simulation timestep (integer)
            
        Returns:
            The velocity value at the given timestep (scaled by speed_multiplier)
        """
        # Extract needed parameters
        time_values = self.processed_data['time_values']
        velocity_values = self.processed_data['value_values']
        t_min = self.processed_data['t_min']
        period = self.processed_data['period']
        
        # Calculate raw time
        t_raw = self.dt * timestep
        
        # Calculate time within cycle (with wrapping)
        t_shifted = t_raw - t_min
        t_cycle = t_min + (t_shifted % period)
        
        # Interpolate to find velocity at this time
        v = np.interp(
            t_cycle,
            time_values,
            velocity_values,
            left=velocity_values[0],
            right=velocity_values[-1]
        )

        # print("time as secs: ", t_raw)
        
        # Scale by speed_multiplier and return
        return v * self.speed_multiplier

    def calculate_velocity_at_time(self, time):
        """
        Calculate the velocity value for a given physical time.
        
        Args:
            time: The physical time in seconds
            
        Returns:
            The velocity value at the given time (scaled by speed_multiplier)
        """
        # Calculate equivalent timestep
        timestep = int(time / self.dt)
        return self.calculate_velocity_at_timestep(timestep)

    def debug_velocity(self, start_step=0, num_steps=100, step_size=10):
        """
        Print velocity values for a range of timesteps for debugging.
        
        Args:
            start_step: First timestep to evaluate
            num_steps: Number of steps to evaluate
            step_size: Interval between timesteps
            
        Returns:
            List of (timestep, time, velocity) tuples
        """
        results = []
        
        for i in range(num_steps):
            timestep = start_step + i * step_size
            time = timestep * self.dt
            velocity = self.calculate_velocity_at_timestep(timestep)
            
            results.append((timestep, time, velocity))
            print(f"Step {timestep}: t={time:.4f}s, v={velocity:.6f}")
            
        return results

    def convert_flow_to_velocity(self, vessel_radius_mm):
        """
        Convert flow rate data (ml/s) to uniform velocity (m/s) for a vessel inlet.
        
        Args:
            vessel_radius_mm: The vessel radius in millimeters
            
        Returns:
            The raw CSV data with values converted to m/s
        """
        # Calculate vessel cross-sectional area in m²
        vessel_radius_m = vessel_radius_mm / 1000
        vessel_area_m2 = np.pi * (vessel_radius_m ** 2)
        
        # Convert values from ml/s to m³/s (1 ml = 10^-6 m³)
        # Then divide by area to get velocity in m/s
        m3_per_ml = 1e-6
        conversion_factor = m3_per_ml / vessel_area_m2
        
        # Make a copy of the processed data
        converted_data = self.processed_data.copy()
        
        # Apply conversion to all velocity values
        converted_data['value_values'] = converted_data['value_values'] * conversion_factor
        converted_data['segment_values'] = [v * conversion_factor for v in converted_data['segment_values']]
        
        # Save the maximum velocity value
        max_physical_velocity = max(converted_data['value_values'])
        self.u_max = max_physical_velocity * self.speed_multiplier
        
        print(f"Converted flow rate (ml/s) to velocity (m/s)")
        print(f"Vessel radius: {vessel_radius_mm} mm")
        print(f"Vessel area: {vessel_area_m2:.8f} m²")
        print(f"Conversion factor: {conversion_factor:.8f}")
        print(f"Peak flow: {max(self.processed_data['value_values']):.2f} ml/s")
        print(f"Peak velocity: {max_physical_velocity:.6f} m/s")
        print(f"Speed multiplier: {self.speed_multiplier}")
        print(f"Final peak velocity: {self.u_max:.6f} m/s")
        
        # Return the converted data
        return converted_data

    def convert_flow_to_lattice_velocity(self, vessel_radius_mm, dx=None, dt=None):
        """
        Convert flow rate data (ml/s) to lattice velocity (LU/step).
        
        Args:
            vessel_radius_mm: The vessel radius in millimeters
            dx: Physical length of one lattice unit (m/LU), if None uses self.dx
            dt: Physical time of one lattice step (s/step), if None uses self.dt
            
        Returns:
            The processed data with values converted to lattice velocity
        """
        # Step 1: Convert flow rate (ml/s) to physical velocity (m/s)
        converted_data = self.convert_flow_to_velocity(vessel_radius_mm)
        
        # Step 2: Convert physical velocity (m/s) to lattice velocity (LU/step)
        # Get conversion factors
        dx = dx or getattr(self, 'dx', 1e-3)  # Default: 1mm per lattice unit
        dt = dt or self.dt                    # Use simulation timestep
        
        # The velocity conversion factor: (dt/dx) converts m/s to LU/step
        # v_lattice = v_physical * (dt/dx)
        conversion_factor = dt / dx
        
        # Apply conversion to all velocity values
        for key in ['value_values', 'segment_values']:
            if isinstance(converted_data[key], list):
                converted_data[key] = [v * conversion_factor for v in converted_data[key]]
            else:
                converted_data[key] = converted_data[key] * conversion_factor
        
        # Save the maximum velocity in lattice units
        max_lattice_velocity = max(converted_data['value_values'])
        
        # CRITICAL FIX: Make sure u_max is properly set to the maximum lattice velocity
        self.u_max = max_lattice_velocity
        
        print(f"Converted physical velocity (m/s) to lattice velocity (LU/step)")
        print(f"dx: {dx:.3e} m/LU, dt: {dt:.3e} s/step")
        print(f"Velocity conversion factor: {conversion_factor:.6f}")
        print(f"Max physical velocity: {max_lattice_velocity/conversion_factor:.6f} m/s")
        print(f"Max lattice velocity: {max_lattice_velocity:.6f} LU/step")
        print(f"Final u_max set to: {self.u_max:.6f} LU/step")
        
        return converted_data
