import os
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from typing import Dict, Callable, Any, List, Union, Optional
import inspect
import warp as wp
import jax.numpy as jnp
from xlb.compute_backend import ComputeBackend


class VelocityProfileRegistry:
    """
    Registry for storing and retrieving velocity profile functions with different backends.
    """
    
    def __init__(self, default_backend, default_precision_policy):
        """
        Initialize registry with default backend.
        
        Parameters:
            default_backend: Default backend to use if not specified
            default_precision_policy: Default precision policy
        """
        self.profiles = {}  # Dictionary to store all registered profiles
        self.default_backend = default_backend
        self.default_precision_policy = default_precision_policy
        
        # Register built-in profiles
        self._register_built_in_profiles()
        # self.import_profiles_from_csv("params/velocity_profile_normalized.csv", profile_columns=["ICA", "CCA"], preview_plot=True)
        
    def _register_built_in_profiles(self):
        """Register all built-in velocity profiles."""
        self.register("sinusoidal", self.create_sinusoidal_profile, 
                    "Sinusoidal velocity profile with adjustable frequency")
        
        # Add these lines to register the hybrid carotid profiles
        try:
            self.register("hybrid_ica", lambda **kwargs: self.create_hybrid_velocity_profile(profile_type='ICA', **kwargs),
                        "Hybrid Fourier/analytical model for Internal Carotid Artery")
            self.register("hybrid_cca", lambda **kwargs: self.create_hybrid_velocity_profile(profile_type='CCA', **kwargs), 
                        "Hybrid Fourier/analytical model for Common Carotid Artery")
            print("Registered hybrid carotid velocity profiles")
        except ImportError as e:
            print(f"Note: Could not register hybrid profiles: {e}")
        
    def register(self, name: str, profile_factory: Callable, description: str = None):
        """
        Register a new velocity profile function factory.
        
        Parameters:
            name: Name to identify the velocity profile
            profile_factory: Function that creates backend-specific profile functions
            description: Optional description of the profile
        """
        if name in self.profiles:
            print(f"Warning: Overwriting existing profile '{name}'")
            
        self.profiles[name] = {
            'factory': profile_factory,
            'description': description or "No description provided"
        }
        print(f"Registered velocity profile: '{name}'")
    
    def get(self, name: str, params: Dict[str, Any] = None, backend=None):
        """
        Get a velocity profile function by name.
        
        Parameters:
            name: Name of the registered profile
            params: Parameters for the profile
            backend: Backend to use (JAX or WARP)
            
        Returns:
            Profile function for the specified backend
        """
        if params is None:
            params = {}
            
        if backend is None:
            backend = self.default_backend
            
        if name not in self.profiles:
            raise ValueError(f"Profile '{name}' not found in registry")
            
        # Get the profile factory
        profile_factory = self.profiles[name]['factory']
        
        # Call the factory with parameters to get backend-specific implementation
        return profile_factory(**params, backend=backend)
        
    def list_profiles(self):
        """List all registered profiles."""
        return list(self.profiles.keys())
        
    def get_description(self, name: str):
        """Get the description of a profile."""
        if name not in self.profiles:
            raise ValueError(f"Profile '{name}' not found in registry")
            
        return self.profiles[name]['description']

    def plot_profile(self, name: str, params: Dict = None, num_seconds: float = 1.0,
                   num_points: int = None, ax: plt.Axes = None, compare_backends: bool = True):
        """
        Plot a velocity profile over time.
        
        Parameters:
            name: Name of the profile to plot
            params: Parameters to pass to the profile function
            num_seconds: Number of seconds to plot (default: 1.0)
            num_points: Number of points to plot (higher means better precision)
            ax: Optional matplotlib axes to plot on
            compare_backends: Whether to show both JAX and WARP implementations
        """
        if params is None:
            params = {}
            
        # Create axes if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            
        # Set number of points if not specified
        if num_points is None:
            num_points = int(1000 * num_seconds)  # 1000 points per second by default
            
        # Generate time points evenly spread over the requested seconds
        dt = params.get('dt', 0.01)  # Get dt from params or use default
        times = np.linspace(0, num_seconds, num_points)  # Actual times in seconds
        
        backends_to_plot = [ComputeBackend.JAX]
        if compare_backends:
            backends_to_plot.append(ComputeBackend.WARP)
        
        for backend in backends_to_plot:
            try:
                # Calculate velocities for this backend
                velocities = []
                
                # For JAX backend - use the regular approach
                if backend == ComputeBackend.JAX:
                    # Get the profile function for JAX backend
                    profile_func = self.get(name, params, backend=backend)
                    
                    # If profile_func is a function that returns a function (typical for JAX)
                    if callable(profile_func) and not inspect.signature(profile_func).parameters:
                        profile_func = profile_func()
                        
                    # Calculate velocities at each timestep
                    timesteps = times / dt  # Convert time to timesteps
                    for t in timesteps:
                        # Handle different return types from the profile function
                        result = profile_func(t)
                        
                        # Adapt based on return type
                        if isinstance(result, np.ndarray) or isinstance(result, jnp.ndarray):
                            if len(result.shape) > 0:  # It's an array
                                velocities.append(float(result[0]))  # Assume first component is x velocity
                            else:
                                velocities.append(float(result))  # It's a scalar
                        elif hasattr(result, 'x'):
                            velocities.append(float(result.x))  # Assuming vec-like object with x attribute
                        else:
                            velocities.append(float(result))  # Assume scalar velocity
                
                # For WARP backend - use NumPy equivalent for visualization only
                else:
                    # Handle different profile types with equivalent NumPy implementations
                    if name == "sinusoidal":
                        # Extract parameters
                        u_max = params.get('u_max', 1.0)
                        frequency = params.get('frequency', 1.0)
                        offset = params.get('offset', 0.5)
                        amplitude = params.get('amplitude', 0.5)
                        omega = 2.0 * np.pi * frequency
                        
                        # Simulate the WARP function using NumPy (only for visualization)
                        for t in times:
                            u_x = u_max * (offset + amplitude * np.sin(omega * t))
                            velocities.append(float(u_x))
                    
                    elif name == "boundary_condition":
                        # Extract parameters
                        u_max = params.get('u_max', 1.0)
                        omega = 2.0 * np.pi * 12345
                        
                        # Simulate the WARP function using NumPy (only for visualization)
                        for t in times:
                            u_x = u_max * (1.5 + 1.5 * np.sin(omega * t))
                            velocities.append(float(u_x))
                    
                    else:
                        print(f"Warning: Unknown WARP profile '{name}' for visualization")
                        continue
                
                # Set style based on backend
                if backend == ComputeBackend.JAX:
                    style = {'linestyle': '-', 'linewidth': 2, 'alpha': 0.8}
                    label = f"{name} (JAX)"
                    color = '#1f77b4'  # Blue
                else:  # WARP
                    style = {'linestyle': '-', 'linewidth': 2, 'alpha': 0.8}
                    label = f"{name} (WARP)"
                    color = '#ff7f0e'  # Orange
                    
                # Plot the velocity profile using actual times instead of timesteps
                ax.plot(times, velocities, label=label, color=color, **style)
                
            except Exception as e:
                print(f"Error plotting {name} with {backend}: {e}")
                import traceback
                traceback.print_exc()
                
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Velocity')
        ax.set_title(f'Velocity Profile: {name} ({num_seconds}s)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return ax

    # Built-in profile implementation - only keeping sinusoidal
    def create_sinusoidal_profile(self, u_max=1.0, dt=0.01, frequency=1.0, 
                                offset=0.5, amplitude=0.5, backend=None, **kwargs):
        """
        Create a sinusoidal velocity profile.
        Compatible with both JAX and WARP backends.
        
        Parameters:
            u_max: Maximum velocity
            dt: Time step
            frequency: Oscillation frequency (Hz)
            offset: Vertical offset
            amplitude: Amplitude of oscillation
            backend: Compute backend
        
        Returns:
            A backend-specific profile function
        """
        omega = 2.0 * np.pi * frequency
        
        if backend is None:
            backend = self.default_backend

        # Store parameters as WARP static types if available
        if self.default_precision_policy is not None:
            u_max_static = wp.static(self.default_precision_policy.store_precision.wp_dtype(u_max))
            dt_static = wp.static(self.default_precision_policy.store_precision.wp_dtype(dt))
            omega_static = wp.static(self.default_precision_policy.store_precision.wp_dtype(omega))
        else:
            u_max_static = wp.static(wp.float32(u_max))
            dt_static = wp.static(wp.float32(dt))
            omega_static = wp.static(wp.float32(omega))
        
        # WARP implementation
        @wp.func
        def sinusoidal_profile_warp(index: wp.vec3i, timestep: int = 0):
            t = dt_static * wp.float32(timestep)
            u_x = u_max_static * (offset + amplitude * wp.sin(omega_static * t))
            return wp.vec(u_x, length=1)
            
        # JAX implementation
        def sinusoidal_profile_jax():
            def velocity(timestep):
                t = dt * timestep
                u_x = u_max * (offset + amplitude * jnp.sin(omega * t))
                u_y = jnp.zeros_like(u_x) if isinstance(u_x, jnp.ndarray) else 0.0
                return jnp.array([u_x, u_y])
            return velocity
        
        # Return the appropriate implementation
        if backend == ComputeBackend.JAX:
            return sinusoidal_profile_jax
        elif backend == ComputeBackend.WARP:
            return sinusoidal_profile_warp
        

    def import_profiles_from_csv(self, csv_filepath: str, x_column=None, 
                           profile_columns=None, dt=0.01, 
                           use_column_names=True, preview_plot=True):
        """
        Import velocity profiles from CSV data with proper cyclic behavior.
        """
        import pandas as pd
        import os
        
        # Read the CSV data
        try:
            data = pd.read_csv(csv_filepath)
            print(f"Successfully read CSV file: {csv_filepath}")
            print(f"CSV columns: {list(data.columns)}")
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return []
        
        # If x_column isn't specified or doesn't exist, use the first column
        if x_column is None or x_column not in data.columns:
            x_column = data.columns[0]
            print(f"Using '{x_column}' as the time/x column")
        
        # Use all columns except x_column if profile_columns not specified
        if profile_columns is None:
            profile_columns = [col for col in data.columns if col != x_column]
        
        # Create a preview plot of the raw data
        if preview_plot:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot each profile column
            for col_name in profile_columns:
                # Plot the raw data
                ax.plot(data[x_column], data[col_name], 
                        label=col_name, linewidth=2, alpha=0.8)
            
            # Add labels and legend
            ax.set_xlabel(x_column)
            ax.set_ylabel('Velocity')
            ax.set_title(f'Raw CSV Data: {os.path.basename(csv_filepath)}')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            plt.tight_layout()
            plt.savefig("csv_profiles_raw_data.png", dpi=300)
            plt.show()
        
        # Track registered profiles
        profile_names = []
        
        # Process each profile column
        for col_name in profile_columns:
            # Extract data
            time_values = data[x_column].values.astype(np.float32)
            velocity_values = data[col_name].values.astype(np.float32)
            
            # Remove any NaN values
            valid_mask = ~(np.isnan(time_values) | np.isnan(velocity_values))
            time_clean = time_values[valid_mask]
            velocity_clean = velocity_values[valid_mask]
            
            if len(time_clean) < 2:
                print(f"Warning: Not enough valid data points for {col_name}")
                continue
            
            # Calculate cycle length for repetition
            min_time = float(time_clean[0])
            max_time = float(time_clean[-1])
            cycle_length = max_time - min_time
            print(f"Profile '{col_name}' cycle length: {cycle_length:.4f} seconds")
            
            # Generate a name for this profile
            profile_name = col_name if use_column_names else f"csv_profile_{len(profile_names)}"
            
            # Create the profile factory - passing raw data instead of WARP arrays
            def create_csv_profile(time_data=time_clean.tolist(), 
                                 velocity_data=velocity_clean.tolist(),
                                 t_min=min_time,
                                 t_max=max_time, 
                                 cycle_len=cycle_length,
                                 dt=dt,
                                 backend=None):
                """Factory function for CSV-based profile"""
                
                # WARP implementation using pre-computed values
                @wp.func
                def csv_profile_warp(index: wp.vec3i, timestep: int = 0):
                    """WARP implementation using linear interpolation"""
                    # Current time from timestep
                    t_raw = wp.float32(timestep) * wp.float32(dt)
                    
                    # Make cyclic by modulo with cycle length using manual mod calculation
                    # First shift to zero-based
                    t_shifted = t_raw - wp.float32(t_min)
                    
                    # Calculate modulo using division and floor
                    # t_cycle = t_shifted - wp.float32(cycle_len) * wp.float(int(t_shifted / cycle_len))
                    # For simplicity, let's use a different approach
                    
                    # Compute whole cycles completed
                    cycles = wp.int32(t_shifted / wp.float32(cycle_len))
                    
                    # Compute time within current cycle
                    t_within_cycle = t_shifted - wp.float32(cycles) * wp.float32(cycle_len)
                    
                    # Shift back to original time range
                    t_cycle = t_within_cycle + wp.float32(t_min)
                    
                    # Default result
                    u_x = wp.float32(velocity_data[0])
                    
                    # Simple linear search through time data points
                    for i in range(len(time_data) - 1):
                        if wp.float32(time_data[i]) <= t_cycle and t_cycle < wp.float32(time_data[i+1]):
                            # Linear interpolation
                            t0 = wp.float32(time_data[i])
                            t1 = wp.float32(time_data[i+1])
                            v0 = wp.float32(velocity_data[i])
                            v1 = wp.float32(velocity_data[i+1])
                            
                            alpha = (t_cycle - t0) / (t1 - t0)
                            u_x = v0 + alpha * (v1 - v0)
                            break
                    
                    return wp.vec(u_x, length=1)
                
                # JAX implementation
                def csv_profile_jax():
                    """JAX implementation of the CSV profile"""
                    # Create JAX arrays for interpolation
                    x_jax = jnp.array(time_data)
                    y_jax = jnp.array(velocity_data)
                    
                    def velocity(timestep):
                        t_raw = dt * timestep
                        # Make cyclic by modulo with cycle length
                        t_cycle = jnp.mod(t_raw - t_min, cycle_len) + t_min
                        # Interpolate to get velocity
                        u_x = jnp.interp(t_cycle, x_jax, y_jax)
                        u_y = jnp.zeros_like(u_x) if isinstance(u_x, jnp.ndarray) else 0.0
                        return jnp.array([u_x, u_y])
                        
                    return velocity
                    
                # Return appropriate implementation
                if backend == ComputeBackend.JAX:
                    return csv_profile_jax
                elif backend == ComputeBackend.WARP:
                    return csv_profile_warp
            
            # Add function to the registry
            self.register(profile_name, create_csv_profile, 
                         f"CSV-based cyclic velocity profile from column '{col_name}'")
            profile_names.append(profile_name)
            
            print(f"Registered cyclic profile '{profile_name}' from CSV column '{col_name}'")
        
        return profile_names

    # Add this method to the VelocityProfileRegistry class
    def create_hybrid_velocity_profile(self, profile_type='ICA', u_max=1.0, dt=0.01, backend=None, **kwargs):
        """
        Create a hybrid Fourier/analytical model velocity profile based on optimized parameters.
        Compatible with both JAX and WARP backends.
        
        Parameters:
            profile_type: 'ICA' or 'CCA'
            u_max: Maximum velocity scaling factor
            dt: Time step for simulation
            backend: Computing backend
        
        Returns:
            A backend-specific profile function
        """
        if backend is None:
            backend = self.default_backend
        
        # Load appropriate model parameters
        try:
            if profile_type == 'ICA':
                from params.ica_model import (
                    PERIOD, T_MIN, TRANSITION_FRACTION, MODEL_TYPE,
                    FOURIER_PARAMS, N_HARMONICS
                )
                
                # Load method-specific parameters
                if 'polynomial' in MODEL_TYPE:
                    from params.ica_model import POLY_COEFFS
                    method_params = POLY_COEFFS
                    method_type = 'polynomial'
                elif 'exponential' in MODEL_TYPE:
                    from params.ica_model import EXP_A, EXP_B, EXP_C, EXP_D, EXP_OFFSET
                    method_params = [EXP_A, EXP_B, EXP_C, EXP_D, EXP_OFFSET]
                    method_type = 'exponential'
                elif 'chebyshev' in MODEL_TYPE:
                    from params.ica_model import CHEBY_COEFFS
                    method_params = CHEBY_COEFFS
                    method_type = 'chebyshev'
                elif 'windkessel' in MODEL_TYPE:
                    from params.ica_model import WK_A_ADJUSTED, WK_B, WK_C, WK_D, WK_K
                    method_params = [WK_A_ADJUSTED, WK_B, WK_C, WK_D, WK_K]
                    method_type = 'windkessel'
                else:
                    print(f"Warning: Unknown model type {MODEL_TYPE}, using default")
                    method_type = 'default'
                    method_params = []
                    
            elif profile_type == 'CCA':
                from params.cca_model import (
                    PERIOD, T_MIN, TRANSITION_FRACTION, MODEL_TYPE,
                    FOURIER_PARAMS, N_HARMONICS
                )
                
                # Load method-specific parameters
                if 'polynomial' in MODEL_TYPE:
                    from params.cca_model import POLY_COEFFS
                    method_params = POLY_COEFFS
                    method_type = 'polynomial'
                elif 'exponential' in MODEL_TYPE:
                    from params.cca_model import EXP_A, EXP_B, EXP_C, EXP_D, EXP_OFFSET
                    method_params = [EXP_A, EXP_B, EXP_C, EXP_D, EXP_OFFSET]
                    method_type = 'exponential'
                elif 'chebyshev' in MODEL_TYPE:
                    from params.cca_model import CHEBY_COEFFS
                    method_params = CHEBY_COEFFS
                    method_type = 'chebyshev'
                elif 'windkessel' in MODEL_TYPE:
                    from params.cca_model import WK_A_ADJUSTED, WK_B, WK_C, WK_D, WK_K
                    method_params = [WK_A_ADJUSTED, WK_B, WK_C, WK_D, WK_K]
                    method_type = 'windkessel'
                else:
                    print(f"Warning: Unknown model type {MODEL_TYPE}, using default")
                    method_type = 'default'
                    method_params = []
            else:
                raise ValueError(f"Unknown profile type: {profile_type}")
                
            print(f"Loaded {profile_type} hybrid model: {MODEL_TYPE}")
            
        except ImportError:
            raise ImportError(f"{profile_type} model not found. Run optimization first to generate the model.")
    
        # Store parameters as WARP static types if needed
        if self.default_precision_policy is not None:
            u_max_static = wp.static(self.default_precision_policy.store_precision.wp_dtype(u_max))
            dt_static = wp.static(self.default_precision_policy.store_precision.wp_dtype(dt))
            period_static = wp.static(self.default_precision_policy.store_precision.wp_dtype(PERIOD))
            t_min_static = wp.static(self.default_precision_policy.store_precision.wp_dtype(T_MIN))
            transition_fraction_static = wp.static(self.default_precision_policy.store_precision.wp_dtype(TRANSITION_FRACTION))
            a0_static = wp.static(self.default_precision_policy.store_precision.wp_dtype(FOURIER_PARAMS[0]))
        else:
            u_max_static = wp.static(wp.float32(u_max))
            dt_static = wp.static(wp.float32(dt))
            period_static = wp.static(wp.float32(PERIOD))
            t_min_static = wp.static(wp.float32(T_MIN))
            transition_fraction_static = wp.static(wp.float32(TRANSITION_FRACTION))
            a0_static = wp.static(wp.float32(FOURIER_PARAMS[0]))
        
        # WARP implementation
        if method_type == 'polynomial':
            # Create static versions of polynomial coefficients
            poly_params_static = []
            for i in range(min(5, len(method_params))):  # Limit to 5 coefficients for WARP
                if self.default_precision_policy is not None:
                    poly_params_static.append(wp.static(self.default_precision_policy.store_precision.wp_dtype(method_params[i])))
                else:
                    poly_params_static.append(wp.static(wp.float32(method_params[i])))
            
            # Pad with zeros if needed
            while len(poly_params_static) < 5:
                poly_params_static.append(wp.static(wp.float32(0.0)))
                
            @wp.func
            def hybrid_profile_warp(index: wp.vec3i, timestep: int = 0):
                # Calculate time with cycle wrapping
                t_raw = t_min_static + dt_static * wp.float32(timestep)
                t_shifted = t_raw - t_min_static  # Shift to zero-based
                cycles = wp.int32(t_shifted / period_static)  # Number of complete cycles
                t_within_cycle = t_shifted - wp.float32(cycles) * period_static  # Time within current cycle
                t_cycle = t_within_cycle + t_min_static  # Shift back to original time range
                
                # Determine transition point
                t_transition = t_min_static + transition_fraction_static * period_static
                
                # Normalize time for Fourier calculation
                scaled_period = period_static * transition_fraction_static
                
                # First part (Fourier)
                if t_cycle <= t_transition:
                    # Fourier series calculation
                    result = a0_static
                    
                    # Manually unroll the Fourier series calculation for first few harmonics
                    if len(FOURIER_PARAMS) >= 3:
                        a1 = wp.static(wp.float32(FOURIER_PARAMS[1]))
                        b1 = wp.static(wp.float32(FOURIER_PARAMS[2]))
                        result += a1 * wp.cos(2.0*wp.pi*1.0*(t_cycle-t_min_static)/scaled_period)
                        result += b1 * wp.sin(2.0*wp.pi*1.0*(t_cycle-t_min_static)/scaled_period)
                    
                    if len(FOURIER_PARAMS) >= 5:
                        a2 = wp.static(wp.float32(FOURIER_PARAMS[3]))
                        b2 = wp.static(wp.float32(FOURIER_PARAMS[4]))
                        result += a2 * wp.cos(2.0*wp.pi*2.0*(t_cycle-t_min_static)/scaled_period)
                        result += b2 * wp.sin(2.0*wp.pi*2.0*(t_cycle-t_min_static)/scaled_period)
                    
                    if len(FOURIER_PARAMS) >= 7:
                        a3 = wp.static(wp.float32(FOURIER_PARAMS[5]))
                        b3 = wp.static(wp.float32(FOURIER_PARAMS[6]))
                        result += a3 * wp.cos(2.0*wp.pi*3.0*(t_cycle-t_min_static)/scaled_period)
                        result += b3 * wp.sin(2.0*wp.pi*3.0*(t_cycle-t_min_static)/scaled_period)
                    
                    return wp.vec(u_max_static * result, length=1)
                    
                # Second part (Polynomial)
                else:
                    # Calculate normalized time
                    t_norm = (t_cycle - t_transition) / (t_min_static + period_static - t_transition)
                    
                    # Start with transition value
                    result = a0_static
                    
                    # Manually unroll polynomial calculation
                    result += poly_params_static[0] * t_norm
                    result += poly_params_static[1] * t_norm * t_norm
                    result += poly_params_static[2] * t_norm * t_norm * t_norm
                    result += poly_params_static[3] * t_norm * t_norm * t_norm * t_norm
                    result += poly_params_static[4] * t_norm * t_norm * t_norm * t_norm * t_norm
                    
                    return wp.vec(u_max_static * result, length=1)
                    
        elif method_type == 'exponential':
            # Extract and convert exponential parameters
            if self.default_precision_policy is not None:
                exp_a_static = wp.static(self.default_precision_policy.store_precision.wp_dtype(method_params[0]))
                exp_b_static = wp.static(self.default_precision_policy.store_precision.wp_dtype(method_params[1]))
                exp_c_static = wp.static(self.default_precision_policy.store_precision.wp_dtype(method_params[2]))
                exp_d_static = wp.static(self.default_precision_policy.store_precision.wp_dtype(method_params[3]))
                exp_offset_static = wp.static(self.default_precision_policy.store_precision.wp_dtype(method_params[4]))
            else:
                exp_a_static = wp.static(wp.float32(method_params[0]))
                exp_b_static = wp.static(wp.float32(method_params[1]))
                exp_c_static = wp.static(wp.float32(method_params[2]))
                exp_d_static = wp.static(wp.float32(method_params[3]))
                exp_offset_static = wp.static(wp.float32(method_params[4]))
            
            @wp.func
            def hybrid_profile_warp(index: wp.vec3i, timestep: int = 0):
                # Calculate time with cycle wrapping
                t_raw = t_min_static + dt_static * wp.float32(timestep)
                t_shifted = t_raw - t_min_static  # Shift to zero-based
                cycles = wp.int32(t_shifted / period_static)  # Number of complete cycles
                t_within_cycle = t_shifted - wp.float32(cycles) * period_static  # Time within current cycle
                t_cycle = t_within_cycle + t_min_static  # Shift back to original time range
                
                # Determine transition point
                t_transition = t_min_static + transition_fraction_static * period_static
                
                # First part (Fourier) - similar to polynomial implementation
                if t_cycle <= t_transition:
                    # Similar Fourier implementation (code reuse)
                    scaled_period = period_static * transition_fraction_static
                    result = a0_static
                    
                    if len(FOURIER_PARAMS) >= 3:
                        a1 = wp.static(wp.float32(FOURIER_PARAMS[1]))
                        b1 = wp.static(wp.float32(FOURIER_PARAMS[2]))
                        result += a1 * wp.cos(2.0*wp.pi*1.0*(t_cycle-t_min_static)/scaled_period)
                        result += b1 * wp.sin(2.0*wp.pi*1.0*(t_cycle-t_min_static)/scaled_period)
                    
                    if len(FOURIER_PARAMS) >= 5:
                        a2 = wp.static(wp.float32(FOURIER_PARAMS[3]))
                        b2 = wp.static(wp.float32(FOURIER_PARAMS[4]))
                        result += a2 * wp.cos(2.0*wp.pi*2.0*(t_cycle-t_min_static)/scaled_period)
                        result += b2 * wp.sin(2.0*wp.pi*2.0*(t_cycle-t_min_static)/scaled_period)
                    
                    if len(FOURIER_PARAMS) >= 7:
                        a3 = wp.static(wp.float32(FOURIER_PARAMS[5]))
                        b3 = wp.static(wp.float32(FOURIER_PARAMS[6]))
                        result += a3 * wp.cos(2.0*wp.pi*3.0*(t_cycle-t_min_static)/scaled_period)
                        result += b3 * wp.sin(2.0*wp.pi*3.0*(t_cycle-t_min_static)/scaled_period)
                    
                    return wp.vec(u_max_static * result, length=1)
                
                # Second part (Exponential)
                else:
                    # Calculate normalized time
                    t_norm = (t_cycle - t_transition) / (t_min_static + period_static - t_transition)
                    
                    # Double exponential formula
                    result = exp_a_static * wp.exp(-exp_b_static * t_norm)
                    result += exp_c_static * wp.exp(-exp_d_static * t_norm) 
                    result += exp_offset_static
                    
                    return wp.vec(u_max_static * result, length=1)
                    
        elif method_type == 'windkessel':
            # Extract and convert Windkessel parameters
            if self.default_precision_policy is not None:
                wk_a_adj_static = wp.static(self.default_precision_policy.store_precision.wp_dtype(method_params[0]))
                wk_b_static = wp.static(self.default_precision_policy.store_precision.wp_dtype(method_params[1]))
                wk_c_static = wp.static(self.default_precision_policy.store_precision.wp_dtype(method_params[2]))
                wk_d_static = wp.static(self.default_precision_policy.store_precision.wp_dtype(method_params[3]))
                wk_k_static = wp.static(self.default_precision_policy.store_precision.wp_dtype(method_params[4]))
            else:
                wk_a_adj_static = wp.static(wp.float32(method_params[0]))
                wk_b_static = wp.static(wp.float32(method_params[1]))
                wk_c_static = wp.static(wp.float32(method_params[2]))
                wk_d_static = wp.static(wp.float32(method_params[3]))
                wk_k_static = wp.static(wp.float32(method_params[4]))
                
            @wp.func
            def hybrid_profile_warp(index: wp.vec3i, timestep: int = 0):
                # Calculate time with cycle wrapping
                t_raw = t_min_static + dt_static * wp.float32(timestep)
                t_cycle = t_min_static + wp.fmod(t_raw - t_min_static, period_static)
                
                # Determine transition point
                t_transition = t_min_static + transition_fraction_static * period_static
                
                # First part (Fourier)
                if t_cycle <= t_transition:
                    # Fourier series calculation
                    result = a0_static
                    
                    # Use static boolean flags instead of dynamic length checks
                    if wp.static(has_harmonic1):  # Use wp.static to make this a compile-time constant
                        a1 = wp.static(wp.float32(FOURIER_PARAMS[1]))
                        b1 = wp.static(wp.float32(FOURIER_PARAMS[2]))
                        result += a1 * wp.cos(2.0*wp.pi*1.0*(t_cycle-t_min_static)/scaled_period)
                        result += b1 * wp.sin(2.0*wp.pi*1.0*(t_cycle-t_min_static)/scaled_period)
                    
                    if wp.static(has_harmonic2):
                        a2 = wp.static(wp.float32(FOURIER_PARAMS[3]))
                        b2 = wp.static(wp.float32(FOURIER_PARAMS[4]))
                        result += a2 * wp.cos(2.0*wp.pi*2.0*(t_cycle-t_min_static)/scaled_period)
                        result += b2 * wp.sin(2.0*wp.pi*2.0*(t_cycle-t_min_static)/scaled_period)
                    
                    if wp.static(has_harmonic3):
                        a3 = wp.static(wp.float32(FOURIER_PARAMS[5]))
                        b3 = wp.static(wp.float32(FOURIER_PARAMS[6]))
                        result += a3 * wp.cos(2.0*wp.pi*3.0*(t_cycle-t_min_static)/scaled_period)
                        result += b3 * wp.sin(2.0*wp.pi*3.0*(t_cycle-t_min_static)/scaled_period)
                    
                    return wp.vec(u_max_static * result, length=1)
                
                # Second part (Windkessel)
                else:
                    # Calculate normalized time
                    t_norm = (t_cycle - t_transition) / (t_min_static + period_static - t_transition)
                    
                    # Windkessel formula: a*exp(-b*t)*(1 + c*sin(d*t + k))
                    result = wk_a_adj_static * wp.exp(-wk_b_static * t_norm) 
                    result *= (1.0 + wk_c_static * wp.sin(wk_d_static * t_norm + wk_k_static))
                    
                    return wp.vec(u_max_static * result, length=1)
                    
        else:
            # Default/fallback implementation
            @wp.func
            def hybrid_profile_warp(index: wp.vec3i, timestep: int = 0):
                # Simple Fourier calculation with just DC component
                return wp.vec(u_max_static * a0_static, length=1)
    
        # JAX implementation - more flexible since we don't need manual unrolling
        def hybrid_profile_jax():
            def velocity(timestep):
                # Calculate time with cycle wrapping
                t_raw = T_MIN + dt * timestep
                t_cycle = T_MIN + jnp.mod(t_raw - T_MIN, PERIOD)
                
                # Determine transition point
                t_transition = T_MIN + TRANSITION_FRACTION * PERIOD
                
                # Create the two parts of the function
                def fourier_part(t):
                    scaled_period = PERIOD * TRANSITION_FRACTION
                    result = FOURIER_PARAMS[0]
                    
                    for i in range(1, N_HARMONICS + 1):
                        if 2*i < len(FOURIER_PARAMS):
                            a = FOURIER_PARAMS[2*i-1]
                            b = FOURIER_PARAMS[2*i]
                            result += a * jnp.cos(2*jnp.pi*i*(t-T_MIN)/scaled_period)
                            result += b * jnp.sin(2*jnp.pi*i*(t-T_MIN)/scaled_period)
                    
                    return result
                
                def second_part(t):
                    # Calculate normalized time
                    t_norm = (t - t_transition) / (T_MIN + PERIOD - t_transition)
                    
                    if method_type == 'polynomial':
                        result = FOURIER_PARAMS[0]  # Start with DC component
                        for i, coef in enumerate(method_params):
                            result += coef * (t_norm ** (i+1))
                        return result
                    
                    elif method_type == 'exponential':
                        a, b, c, d, offset = method_params
                        return a * jnp.exp(-b * t_norm) + c * jnp.exp(-d * t_norm) + offset
                    
                    elif method_type == 'chebyshev':
                        # JAX-compatible Chebyshev polynomial evaluation
                        t_cheb = -1 + 2 * t_norm
                        
                        # Convert to standard polynomial form for JAX
                        n = len(method_params)
                        b_nm2 = 0.0
                        b_nm1 = 0.0
                        for i in range(n-1, -1, -1):
                            b_n = method_params[i] + 2*t_cheb*b_nm1 - b_nm2
                            b_nm2 = b_nm1
                            b_nm1 = b_n
                        
                        return b_nm1 - t_cheb * b_nm2
                    
                    elif method_type == 'windkessel':
                        a_adj, b, c, d, k = method_params
                        return a_adj * jnp.exp(-b * t_norm) * (1.0 + c * jnp.sin(d * t_norm + k))
                    
                    else:
                        return FOURIER_PARAMS[0]  # Default to DC component
                
                # Combine the two parts using a conditional
                result = jnp.where(t_cycle <= t_transition, 
                                 fourier_part(t_cycle), 
                                 second_part(t_cycle))
                
                # Scale by u_max and return as a velocity vector
                u_x = u_max * result
                u_y = jnp.zeros_like(u_x) if isinstance(u_x, jnp.ndarray) else 0.0
                
                return jnp.array([u_x, u_y])
            
            return velocity
        
        # Return the appropriate implementation
        if backend == ComputeBackend.JAX:
            return hybrid_profile_jax
        elif backend == ComputeBackend.WARP:
            return hybrid_profile_warp

if __name__ == "__main__":
    # Import needed XLB components 
    from xlb import PrecisionPolicy

    # Create required parameters for registry
    backend = ComputeBackend.JAX  # Use JAX for plotting
    precision_policy = PrecisionPolicy.FP32FP32  # Use single precision
    
    # Create registry with required parameters
    registry = VelocityProfileRegistry(
        default_backend=backend, 
        default_precision_policy=precision_policy
    )
    
    # Define a fully compatible sinusoidal profile for use in boundary conditions
    def bc_profile(u_max=1.0, dt=0.01, backend=None):
        """
        Create a boundary condition profile with sinusoidal variation.
        
        Parameters:
            u_max: Maximum velocity
            dt: Time step
            backend: Compute backend
        """
        omega = 2.0 * np.pi * 12345  # High frequency for example
        
        # WARP implementation
        @wp.func
        def bc_profile_warp(index: wp.vec3i, timestep: int = 0):
            t = dt * wp.float32(timestep)
            u_x = u_max * (1.5 + 1.5 * wp.sin(omega * t))
            return wp.vec(u_x, length=1)
            
        # JAX implementation
        def bc_profile_jax():
            def velocity(timestep):
                t = dt * timestep
                u_x = u_max * (0.5 + 0.5 * jnp.sin(omega * t))
                u_y = jnp.zeros_like(u_x) if isinstance(u_x, jnp.ndarray) else 0.0
                return jnp.array([u_x, u_y])
            return velocity
        
        # Return the appropriate implementation
        if backend == ComputeBackend.JAX:
            return bc_profile_jax
        elif backend == ComputeBackend.WARP:
            return bc_profile_warp
    
    # Register the custom profile
    registry.register("boundary_condition", bc_profile, "Custom boundary condition profile")
    
    # List all available profiles
    print("\nAvailable velocity profiles:")
    for name in registry.list_profiles():
        print(f"- {name}: {registry.get_description(name)}")
    
    # Create a figure with subplots to show different time ranges and backend comparisons
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot profiles with different time ranges and resolutions, comparing backends
    registry.plot_profile("sinusoidal", {"frequency": 1.0, "u_max": 1.0}, 
                        num_seconds=1.0, num_points=100, ax=axs[0, 0],
                        compare_backends=True)
    
    registry.plot_profile("sinusoidal", {"frequency": 2.0, "u_max": 1.0}, 
                        num_seconds=0.5, num_points=200, ax=axs[0, 1],
                        compare_backends=True)
    
    registry.plot_profile("boundary_condition", {"u_max": 1.0}, 
                        num_seconds=0.01, num_points=500, ax=axs[1, 0],
                        compare_backends=True)
    
    # For this high-frequency example, use a very small time window
    registry.plot_profile("boundary_condition", {"u_max": 1.0, "dt": 0.0001}, 
                        num_seconds=0.001, num_points=1000, ax=axs[1, 1],
                        compare_backends=True)
    
    plt.tight_layout()
    plt.savefig("velocity_profiles_comparison.png", dpi=300)
    plt.show()

    # Import CSV profiles for ICA and CCA
    import os

    # Import and visualize the CSV profiles
    csv_profiles = registry.import_profiles_from_csv(
        "params/velocity_profile_normalized.csv",
        x_column="time",
        profile_columns=["ICA", "CCA"],
        preview_plot=True  # This will create the plot before registration
    )

    # Plot the processed profiles if they were successfully imported
    if csv_profiles:
        # Create a new figure for the processed profiles
        fig_csv, axs_csv = plt.subplots(len(csv_profiles), 1, figsize=(12, 6*len(csv_profiles)))
        
        if len(csv_profiles) == 1:
            axs_csv = [axs_csv]  # Make it iterable if only one profile
        
        # Plot each imported profile
        for i, name in enumerate(csv_profiles):
            registry.plot_profile(
                name,
                {"dt": 0.01},
                num_seconds=1.0,  # Adjust based on your data
                num_points=200,
                ax=axs_csv[i],
                compare_backends=True
            )
            axs_csv[i].set_title(f"{name} Velocity Profile")
        
        plt.tight_layout()
        plt.savefig("carotid_profiles_processed.png", dpi=300)
        plt.show()