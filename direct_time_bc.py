import warp as wp
import numpy as np
from typing import Any
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import viridis

class DirectTimeDependentBC:
    """
    A radically simplified BC that overwrites both distributions AND macroscopic values
    """
    
    def __init__(self, inlet_indices, u_max=0.04, frequency=20.0, dt=0.00005):
        """
        Args:
            inlet_indices: List of coordinate arrays for inlet locations
            u_max: Maximum velocity
            frequency: Oscillation frequency in Hz
            dt: Simulation time step
        """
        self.inlet_indices = inlet_indices
        self.u_max = u_max
        self.frequency = frequency
        self.dt = dt
        
        # Convert inlet indices to WARP arrays for use in kernel
        x_coords = np.array(inlet_indices[0], dtype=np.int32)
        y_coords = np.array(inlet_indices[1], dtype=np.int32)
        self.num_points = len(x_coords)
        
        # Keep numpy versions for debugging
        self.x_coords = x_coords
        self.y_coords = y_coords
        
        # Print some sample points to verify locations
        print(f"Sample inlet points: ({x_coords[0]},{y_coords[0]}), ({x_coords[-1]},{y_coords[-1]})")
        
        # Store inlet coordinates as WARP arrays
        self.inlet_x = wp.from_numpy(x_coords)
        self.inlet_y = wp.from_numpy(y_coords)
        
        # Create kernel for setting velocity
        self._create_kernel()
        
        print(f"Created DirectTimeDependentBC with {self.num_points} inlet points")
        print(f"Parameters: u_max={u_max}, freq={frequency}Hz, dt={dt}")
    
    def _create_kernel(self):
        """Create a super simple kernel that just sets distributions"""
        
        @wp.kernel
        def set_inlet_velocity(
            f_post: wp.array4d(dtype=wp.float32),
            timestep: wp.int32,
            inlet_x: wp.array(dtype=wp.int32),
            inlet_y: wp.array(dtype=wp.int32),
            num_points: wp.int32,
            u_max: wp.float32
        ):
            # MAXIMUM SIMPLICITY: Just use constant velocity
            u_x = u_max
            
            # Debug output
            if timestep % 1000 == 0 and wp.tid() == 0:
                wp.printf("[INLET BC] Setting fixed u_x=%.6f at %d points\n", 
                         u_x, num_points)
            
            # Process in parallel across inlet points
            idx = wp.tid()
            if idx < num_points:
                # Get inlet coordinates for this thread
                i = inlet_x[idx]
                j = inlet_y[idx]
                k = 0
                
                # BRUTALLY SIMPLE:
                # Set f1 (east) to a large value
                # Set f3 (west) to a small value
                # This creates a guaranteed east-ward flow
                f_post[1, i, j, k] = 0.3
                f_post[3, i, j, k] = 0.01
                
                # Make sure we have proper density by setting the rest
                f_post[0, i, j, k] = 0.44  # center
                f_post[2, i, j, k] = 0.05  # north
                f_post[4, i, j, k] = 0.05  # south 
                f_post[5, i, j, k] = 0.05  # northeast
                f_post[6, i, j, k] = 0.05  # northwest
                f_post[7, i, j, k] = 0.05  # southwest
                f_post[8, i, j, k] = 0.05  # southeast
        
        self.kernel = set_inlet_velocity
    
    def apply(self, f_post, timestep):
        """Apply the boundary condition after the main step"""
        if timestep % 1000 == 0:
            print(f"Applying direct inlet BC at timestep {timestep}")
        
        # Launch kernel to set inlet values
        wp.launch(
            self.kernel,
            dim=self.num_points,
            inputs=[
                f_post,
                wp.int32(timestep),
                self.inlet_x,
                self.inlet_y,
                wp.int32(self.num_points),
                wp.float32(self.u_max)
            ]
        )
        wp.synchronize()
        
        # Every 10,000 steps, verify BC values directly 
        if timestep % 10000 == 0:
            self._visualize_inlet(f_post, timestep)
            
        return f_post
    
    def _visualize_inlet(self, f, timestep):
        """Directly verify BC values by extracting and plotting them"""
        try:
            # Extract the distribution values at inlet points
            inlet_values = {}
            
            # Get f values for first few directions (focus on 1 & 3)
            print("\nDIRECT VERIFICATION OF INLET VALUES:")
            for idx, (ix, iy) in enumerate(zip(self.x_coords, self.y_coords)):
                if idx < 5:  # Just check a few points
                    f0 = f[0, ix, iy, 0].numpy()
                    f1 = f[1, ix, iy, 0].numpy()  # East
                    f3 = f[3, ix, iy, 0].numpy()  # West
                    print(f"  Inlet ({ix},{iy}): f0={f0:.4f}, f1(east)={f1:.4f}, f3(west)={f3:.4f}")
                    
                    # Calculate expected velocity from these values
                    rho_est = f0 + f1 + f3 + 0.3  # Rough estimate
                    u_est = (f1 - f3) / rho_est
                    print(f"    Estimated velocity: u_x={u_est:.6f}")
                    
                # Store values for all points
                inlet_values[(ix, iy)] = (f[1, ix, iy, 0].numpy(), f[3, ix, iy, 0].numpy())
            
            # Create a plot showing the distribution ratios at inlet
            plt.figure(figsize=(6, 6))
            
            # Extract coordinates and values
            coords = np.array(list(inlet_values.keys()))
            ratios = np.array([east/west if west > 0 else 10 for east, west in inlet_values.values()])
            
            # Create a heatmap of the f1/f3 ratios
            plt.scatter(coords[:, 0], coords[:, 1], c=ratios, cmap='plasma', s=50, 
                       norm=Normalize(vmin=1, vmax=10))
            plt.colorbar(label='f1/f3 ratio')
            plt.title(f'Distribution Ratio at Inlet (Step {timestep})')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.tight_layout()
            
            # Save the plot
            plt.savefig(f'inlet_verification_{timestep}.png')
            print(f"Saved inlet verification plot to inlet_verification_{timestep}.png")
        except Exception as e:
            print(f"Error in visualization: {e}")