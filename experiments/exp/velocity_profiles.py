import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os
from typing import Dict, Callable, Any, Union, Tuple, List

class VelocityProfileLibrary:
    """
    A class to manage different velocity profiles for blood flow simulations.
    Supports analytical profiles, data-driven profiles, and custom functions.
    """
    
    def __init__(self):
        """Initialize the velocity profile library."""
        # Registry of available profiles
        self.profiles = {}
        
        # Register built-in profiles
        self._register_built_in_profiles()
    
    def _register_built_in_profiles(self):
        """Register all built-in velocity profiles."""
        # Steady profiles
        self.register_profile('parabolic', self.parabolic_profile)
        self.register_profile('plug', self.plug_profile)
        
        # Pulsatile profiles
        self.register_profile('sinusoidal', self.sinusoidal_profile)
        self.register_profile('womersley', self.womersley_profile)
        
        # Physiological profiles
        self.register_profile('carotid_cca', self.carotid_cca_profile)
        self.register_profile('carotid_ica', self.carotid_ica_profile)
    
    def register_profile(self, name: str, profile_func: Callable) -> None:
        """
        Register a new velocity profile function.
        
        Parameters:
            name: Name to identify the profile
            profile_func: Function that generates the profile
        """
        if name in self.profiles:
            print(f"Warning: Overwriting existing profile '{name}'")
        
        self.profiles[name] = profile_func
        print(f"Registered profile '{name}'")
    
    def get_profile(self, name: str) -> Callable:
        """
        Get a velocity profile function by name.
        
        Parameters:
            name: Name of the registered profile
        
        Returns:
            The profile function
        """
        if name not in self.profiles:
            raise ValueError(f"Profile '{name}' not found in registry")
        
        return self.profiles[name]
    
    def list_profiles(self) -> List[str]:
        """
        List all available profile names.
        
        Returns:
            List of profile names
        """
        return list(self.profiles.keys())
    
    def save_profile_data(self, profile_name: str, params: Dict[str, Any], 
                         output_path: str, num_points: int = 100) -> None:
        """
        Generate and save profile data to a CSV file.
        
        Parameters:
            profile_name: Name of the profile to use
            params: Parameters for the profile
            output_path: Path to save the CSV file
            num_points: Number of points to generate
        """
        profile_func = self.get_profile(profile_name)
        
        # Generate data
        data = profile_func(**params, num_points=num_points)
        
        # Convert to DataFrame if needed
        if isinstance(data, tuple):
            df = pd.DataFrame({
                'time': data[0],
                'velocity': data[1]
            })
        elif isinstance(data, dict):
            df = pd.DataFrame(data)
        else:
            df = data
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"Saved profile data to {output_path}")
    
    def load_multi_profile_csv(self, csv_path: str, time_col: str = None, 
                          velocity_cols: List[str] = None, normalize: bool = False) -> Dict[str, Callable]:
        """
        Create multiple velocity profile functions from a CSV with several columns.
        
        Parameters:
            csv_path: Path to CSV file
            time_col: Name of the time column (default: first column)
            velocity_cols: List of velocity column names (default: all non-time columns)
            normalize: Whether to normalize time values to [0,1]
            
        Returns:
            Dictionary mapping profile names to their profile functions
        """
        # Load data
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return {}
        
        # Use first column as time if not specified
        if time_col is None:
            time_col = df.columns[0]
        
        # Use all non-time columns if not specified
        if velocity_cols is None:
            velocity_cols = [col for col in df.columns if col != time_col]
        
        # Extract time data
        x = df[time_col].values
        
        # Normalize if requested
        if normalize:
            x = (x - x.min()) / (x.max() - x.min())
        
        # Create profile function for each velocity column
        profile_functions = {}
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        
        for vel_col in velocity_cols:
            # Extract velocity data
            y = df[vel_col].values
            
            # Create interpolation function
            interp_func = interp1d(x, y, kind='cubic', bounds_error=False, fill_value=(y[0], y[-1]))
            
            # Create profile function
            def create_profile_func(interp):
                def profile_func(time=None, num_points=100, **kwargs):
                    """Profile function based on CSV data."""
                    if time is None:
                        time = np.linspace(0, 1, num_points) if normalize else \
                            np.linspace(x.min(), x.max(), num_points)
                    
                    velocity = interp(time)
                    return time, velocity
                return profile_func
            
            # Create and register this profile
            profile_name = f"csv_{base_name}_{vel_col}"
            profile_func = create_profile_func(interp_func)
            self.register_profile(profile_name, profile_func)
            profile_functions[vel_col] = profile_func
        
        return profile_functions
    
    def plot_profile(self, profile_name: str, params: Dict[str, Any] = None, 
                   ax=None, **kwargs) -> plt.Figure:
        """
        Plot a velocity profile.
        
        Parameters:
            profile_name: Name of the profile to plot
            params: Parameters for the profile
            ax: Matplotlib axis to plot on (optional)
            **kwargs: Additional keyword arguments for plotting
            
        Returns:
            Matplotlib figure
        """
        if params is None:
            params = {}
        
        profile_func = self.get_profile(profile_name)
        
        # Generate data
        result = profile_func(**params)
        
        # Extract time and velocity
        if isinstance(result, tuple) and len(result) == 2:
            time, velocity = result
        elif isinstance(result, dict):
            time = result.get('time', np.arange(len(result.get('velocity', []))))
            velocity = result.get('velocity', [])
        else:
            raise ValueError("Profile function must return (time, velocity) tuple or dictionary with these keys")
        
        # Create plot
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure
        
        # Plot data
        ax.plot(time, velocity, **kwargs)
        ax.set_xlabel('Time')
        ax.set_ylabel('Velocity')
        ax.set_title(f'Velocity Profile: {profile_name}')
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def compare_profiles(self, profiles: List[Tuple[str, Dict[str, Any]]], 
                        figsize=(12, 8)) -> plt.Figure:
        """
        Compare multiple velocity profiles on the same plot.
        
        Parameters:
            profiles: List of (profile_name, params) tuples
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        for name, params in profiles:
            self.plot_profile(name, params, ax=ax, label=name)
        
        ax.legend()
        plt.tight_layout()
        return fig
    
    # Built-in profile implementations
    
    def parabolic_profile(self, radius: float, max_velocity: float, 
                         num_points: int = 100, **kwargs) -> dict:
        """
        Generate a parabolic (Poiseuille) velocity profile.
        
        Parameters:
            radius: Radius of the vessel
            max_velocity: Maximum centerline velocity
            num_points: Number of points across the radius
            
        Returns:
            Dictionary with radial position and velocity values
        """
        r = np.linspace(0, radius, num_points)
        velocity = max_velocity * (1 - (r/radius)**2)
        
        return {
            'radial_position': r,
            'velocity': velocity,
            'profile_type': 'parabolic'
        }
    
    def plug_profile(self, radius: float, velocity: float, 
                    num_points: int = 100, **kwargs) -> dict:
        """
        Generate a plug flow velocity profile.
        
        Parameters:
            radius: Radius of the vessel
            velocity: Plug flow velocity
            num_points: Number of points across the radius
            
        Returns:
            Dictionary with radial position and velocity values
        """
        r = np.linspace(0, radius, num_points)
        v = np.ones_like(r) * velocity
        
        return {
            'radial_position': r,
            'velocity': v,
            'profile_type': 'plug'
        }
        
    def sinusoidal_profile(self, mean_velocity: float, amplitude: float, 
                          frequency: float, phase: float = 0.0,
                          time_range: Tuple[float, float] = (0, 1),
                          num_points: int = 100, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a sinusoidal velocity profile.
        
        Parameters:
            mean_velocity: Mean velocity
            amplitude: Amplitude of oscillation
            frequency: Frequency in Hz
            phase: Phase offset in radians
            time_range: (start, end) time range
            num_points: Number of time points
            
        Returns:
            Tuple of (time, velocity) arrays
        """
        time = np.linspace(time_range[0], time_range[1], num_points)
        omega = 2 * np.pi * frequency
        velocity = mean_velocity + amplitude * np.sin(omega * time + phase)
        
        return time, velocity
    
    def womersley_profile(self, radius: float, pressure_amplitude: float,
                         blood_density: float, blood_viscosity: float,
                         frequency: float, time_points: int = 20,
                         radial_points: int = 30, **kwargs) -> dict:
        """
        Generate Womersley velocity profile for pulsatile flow.
        
        Parameters:
            radius: Vessel radius (m)
            pressure_amplitude: Pressure gradient amplitude (Pa/m)
            blood_density: Blood density (kg/m³)
            blood_viscosity: Blood dynamic viscosity (Pa·s)
            frequency: Frequency (Hz)
            time_points: Number of time points in one cycle
            radial_points: Number of radial points
            
        Returns:
            Dictionary with profile data
        """
        # Angular frequency
        omega = 2 * np.pi * frequency
        
        # Womersley number
        alpha = radius * np.sqrt(omega * blood_density / blood_viscosity)
        
        # Radial positions
        r = np.linspace(0, radius, radial_points)
        
        # Time points for one cycle
        t = np.linspace(0, 1/frequency, time_points)
        
        # Preallocate velocity array
        velocity = np.zeros((len(t), len(r)))
        
        # Calculate velocity profile at each time point
        for i, time in enumerate(t):
            # Simplified Womersley solution (approximation)
            for j, radial_pos in enumerate(r):
                # Normalized radial position
                rho = radial_pos / radius
                
                # Simplified Womersley profile
                velocity[i, j] = (pressure_amplitude * radius**2 / (4 * blood_viscosity)) * \
                                (1 - rho**2) * (1 + np.cos(omega * time))
        
        return {
            'time': t,
            'radius': r,
            'velocity': velocity,
            'womersley_number': alpha,
            'profile_type': 'womersley'
        }
    
    def carotid_cca_profile(self, heart_rate: float = 60, 
                           systolic_velocity: float = 0.8,
                           diastolic_velocity: float = 0.2,
                           num_points: int = 100, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a typical common carotid artery velocity profile.
        
        Parameters:
            heart_rate: Heart rate in beats per minute
            systolic_velocity: Peak systolic velocity (m/s)
            diastolic_velocity: End diastolic velocity (m/s)
            num_points: Number of time points per cardiac cycle
            
        Returns:
            Tuple of (time, velocity) arrays
        """
        # Time for one cardiac cycle
        period = 60 / heart_rate  # seconds
        time = np.linspace(0, period, num_points)
        
        # Normalized cardiac cycle (approximation)
        t_norm = time / period
        
        # Parameters for CCA waveform shape
        systolic_peak = 0.14  # Systolic peak at ~14% of cardiac cycle
        dicrotic_notch = 0.45  # Dicrotic notch at ~45% of cardiac cycle
        
        # Create basic waveform components
        systolic_wave = np.exp(-((t_norm - systolic_peak) / 0.1)**2)
        diastolic_wave = (1 - np.exp(-((t_norm - dicrotic_notch) / 0.3)**2)) * \
                         np.exp(-((t_norm - diastolic_wave)/0.5)**2)
        
        # Combine components
        waveform = systolic_wave + 0.3 * diastolic_wave
        waveform = waveform / np.max(waveform)  # Normalize to 1
        
        # Scale to velocity range
        velocity = diastolic_velocity + waveform * (systolic_velocity - diastolic_velocity)
        
        return time, velocity
    
    def carotid_ica_profile(self, heart_rate: float = 60, 
                           systolic_velocity: float = 0.6,
                           diastolic_velocity: float = 0.25,
                           num_points: int = 100, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a typical internal carotid artery velocity profile.
        
        Parameters:
            heart_rate: Heart rate in beats per minute
            systolic_velocity: Peak systolic velocity (m/s)
            diastolic_velocity: End diastolic velocity (m/s)
            num_points: Number of time points per cardiac cycle
            
        Returns:
            Tuple of (time, velocity) arrays
        """
        # Time for one cardiac cycle
        period = 60 / heart_rate  # seconds
        time = np.linspace(0, period, num_points)
        
        # Normalized cardiac cycle
        t_norm = time / period
        
        # Parameters for ICA waveform shape
        systolic_peak = 0.16  # Systolic peak slightly later than CCA
        secondary_peak = 0.5  # Secondary peak at ~50% of cardiac cycle
        
        # Create basic waveform components - ICA has less pulsatility and more continuous flow
        systolic_wave = np.exp(-((t_norm - systolic_peak) / 0.12)**2)
        continuous_component = 0.7 + 0.3 * np.sin(2 * np.pi * t_norm + np.pi)
        
        # Combine components - ICA has higher diastolic flow than CCA
        waveform = 0.7 * systolic_wave + 0.3 * continuous_component
        waveform = waveform / np.max(waveform)  # Normalize to 1
        
        # Scale to velocity range
        velocity = diastolic_velocity + waveform * (systolic_velocity - diastolic_velocity)
        
        return time, velocity
        
    def from_csv_data(self, csv_path: str, time_col: str = None, velocity_col: str = None,
                    normalize_time: bool = True, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a velocity profile from CSV data.
        
        Parameters:
            csv_path: Path to CSV file
            time_col: Name of time column (or None to use first column)
            velocity_col: Name of velocity column (or None to use second column)
            normalize_time: Whether to normalize time to [0,1]
            
        Returns:
            Tuple of (time, velocity) arrays
        """
        # Load profile from CSV and return as a function
        csv_profile = self.load_from_csv(csv_path, x_col=time_col, y_col=velocity_col,
                                        normalize=normalize_time)
        
        # Generate the profile data
        return csv_profile(**kwargs)









def plot_all_profiles():
    """
    Create a comprehensive visualization of all available velocity profiles.
    """
    # Create library
    vel_lib = VelocityProfileLibrary()
    
    # Get all profile names
    profile_names = vel_lib.list_profiles()
    
    # Define parameters for each profile type
    profile_params = {
        'parabolic': {'radius': 0.003, 'max_velocity': 0.6},
        'plug': {'radius': 0.003, 'velocity': 0.4},
        'sinusoidal': {'mean_velocity': 0.5, 'amplitude': 0.25, 'frequency': 1.0, 'num_points': 100},
        'womersley': {'radius': 0.003, 'pressure_amplitude': 1000, 'blood_density': 1056, 
                    'blood_viscosity': 0.0035, 'frequency': 1.1},
        'carotid_cca': {'heart_rate': 70, 'systolic_velocity': 0.8, 'diastolic_velocity': 0.2},
        'carotid_ica': {'heart_rate': 70, 'systolic_velocity': 0.6, 'diastolic_velocity': 0.25}
    }
    
    # Set up subplots in a grid
    n_profiles = len(profile_names)
    n_cols = 2
    n_rows = (n_profiles + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
    
    # Plot each profile
    for i, name in enumerate(profile_names):
        if i < len(axes):
            # Get parameters or use empty dict if not defined
            params = profile_params.get(name, {})
            
            try:
                # For CSV-based profiles that might not have parameters defined
                if name.startswith('csv_'):
                    vel_lib.plot_profile(name, {}, ax=axes[i], label=name, color='darkred')
                else:
                    vel_lib.plot_profile(name, params, ax=axes[i], label=name)
                
                axes[i].set_title(f"Profile: {name}")
                axes[i].legend()
            except Exception as e:
                axes[i].text(0.5, 0.5, f"Error plotting {name}:\n{str(e)}", 
                           ha='center', va='center', transform=axes[i].transAxes)
    
    # Hide any unused subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('all_velocity_profiles.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    # Create library
    vel_lib = VelocityProfileLibrary()
    
    # List available profiles
    print("Available profiles:")
    print(vel_lib.list_profiles())
    
    # Plot all available profiles
    plot_all_profiles()
    
    # Additional individual examples if needed:
    # Plot a sinusoidal profile
    params = {
        'mean_velocity': 0.5, 
        'amplitude': 0.3, 
        'frequency': 1.0,
        'num_points': 200
    }
    fig = vel_lib.plot_profile('sinusoidal', params)
    plt.show()
    
    # Load and compare CSV profiles
    try:
        csv_profile = vel_lib.load_from_csv("params/velocity_profile_normalized.csv", normalize=True)
        
        # Compare profiles
        profiles_to_compare = [
            ('sinusoidal', params),
            ('csv_velocity_profile_normalized_ICA', {})
        ]
        compare_fig = vel_lib.compare_profiles(profiles_to_compare)
        plt.show()
    except FileNotFoundError:
        print("CSV file not found. Skipping CSV comparison.")
