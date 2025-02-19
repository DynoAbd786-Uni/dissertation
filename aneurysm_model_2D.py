import xlb
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.grid import grid_factory
from xlb.operator.stepper import IncompressibleNavierStokesStepper
from xlb.operator.boundary_condition import FullwayBounceBackBC, ZouHeBC, ExtrapolationOutflowBC
from xlb.operator.macroscopic import Macroscopic
from xlb.utils import save_fields_vtk, save_image
import xlb.velocity_set
import warp as wp
import jax.numpy as jnp
import numpy as np
import time
from datetime import timedelta
import json
from datetime import datetime


class AneurysmSimulation2D:
    def __init__(self, omega, grid_shape, velocity_set, backend, precision_policy, resolution, input_params):
        # Store input parameters
        self.input_params = input_params
        
        # initialize backend
        xlb.init(
            velocity_set=velocity_set,
            default_backend=backend,
            default_precision_policy=precision_policy,
        )

        self.grid_shape = grid_shape
        self.velocity_set = velocity_set
        self.backend = backend
        self.precision_policy = precision_policy
        self.omega = omega
        self.resolution = resolution
        self.u_max = input_params["u_max"]
        self.dt = input_params["dt"]
        self.dx = input_params["dx"]
        self.boundary_conditions = []

        # Create grid using factory
        self.grid = grid_factory(grid_shape, compute_backend=backend)
        self._setup()

    def _setup(self):
        self.setup_boundary_conditions()
        self.setup_stepper()
        # Initialize fields using the stepper
        self.f_0, self.f_1, self.bc_mask, self.missing_mask = self.stepper.prepare_fields()

    def define_boundary_indices(self):
        x, y = self.grid_shape

        mid_horizontal = int(round(y / 2))
        pixel_width = round(0.002 / self.resolution)

        # Ovoid parameters
        x0 = x // 2   # center x position
        y0 = (y // 2) + pixel_width  # center y position
        a = x // 4                   # semi-major axis
        b = y // 6                   # semi-minor axis
        
        curve_x = []
        curve_y = []
        
        # Generate only upper half with continuous pixels
        last_y = None
        for x_coord in range(x0 - a, x0 + a + 1):
            if 0 <= x_coord < x:
                # Calculate exact y coordinate for upper curve
                y_coord = y0 + b * np.sqrt(1 - ((x_coord - x0)**2 / a**2))
                y_base = int(y_coord)
                
                # Fill gaps between consecutive y coordinates
                if last_y is not None:
                    for y_fill in range(min(last_y, y_base), max(last_y, y_base) + 1):
                        if 0 <= y_fill < y:
                            curve_x.append(x_coord)
                            curve_y.append(y_fill)
                else:
                    if 0 <= y_base < y:
                        curve_x.append(x_coord)
                        curve_y.append(y_base)
                
                last_y = y_base
        
        # Sort coordinates to maintain order
        points = list(zip(curve_x, curve_y))
        points = sorted(set(points))  # Remove duplicates while preserving order
        curve_x, curve_y = zip(*points)
        
        # Debug prints
        print("Number of curve points:", len(curve_x))
        print("X range:", min(curve_x), "to", max(curve_x))
        print("Y range:", min(curve_y), "to", max(curve_y))

        # Format for XLB
        curve_wall = [list(curve_x), list(curve_y)]
        # print("Curve wall:", curve_wall)

        loc_upper = round(mid_horizontal + pixel_width) 
        loc_lower = round(mid_horizontal - pixel_width)

        print("loc_upper:", loc_upper)
        print("loc_lower:", loc_lower)

        
        # Create straight sections of upper wall
        array_upper_left = [loc_upper for i in range(min(curve_x))]
        array_upper_right = [loc_upper for i in range(max(curve_x) + 1, x)]
        wall_width_upper_left = list(range(min(curve_x)))
        wall_width_upper_right = list(range(max(curve_x) + 1, x))

        # Create lower wall
        wall_width_lower = list(range(x))
        array_lower = [loc_lower for i in range(x)]

        # Combine all wall sections
        wall_width = (wall_width_upper_left + 
                    list(curve_x) + 
                    wall_width_upper_right + 
                    wall_width_lower)
        
        wall_height = (array_upper_left + 
                    list(curve_y) + 
                    array_upper_right + 
                    array_lower)

        # Format for XLB
        walls = [wall_width, wall_height]
        

        # Get inlet/outlet from box_no_edge (similar to how lid-driven cavity gets its lid)
        x_array_left = [0 for i in range(loc_lower + 1, loc_upper)]
        x_array_right = [x - 1 for i in range(loc_lower + 1, loc_upper)]
        y_array = [i for i in range(loc_lower + 1, loc_upper)]

        inlet = [x_array_left, y_array]
        outlet = [x_array_right, y_array]
        
        # Debug prints
        # print("Inlet:", inlet)
        # print("Outlet:", outlet)
        # print("Walls:", walls)
        
        return inlet, outlet, walls

    def setup_boundary_conditions(self):
        inlet, outlet, walls = self.define_boundary_indices()
        
        # Walls: no-slip boundary condition
        bc_walls = FullwayBounceBackBC(indices=walls)
        
        # Inlet: constant velocity profile
        bc_inlet = ZouHeBC("velocity", prescribed_value=(self.u_max, 0.0), indices=inlet)
        
        # Outlet: zero-gradient outflow
        bc_outlet = ExtrapolationOutflowBC(indices=outlet)
        
        self.boundary_conditions = [bc_walls, bc_inlet, bc_outlet]

    def setup_stepper(self):
        self.stepper = IncompressibleNavierStokesStepper(
            omega=self.omega,
            grid=self.grid,
            boundary_conditions=self.boundary_conditions,
            collision_type="KBC",
        )

    def run(self, num_steps, post_process_interval=100):
        # Save parameters before starting simulation
        self.save_simulation_parameters()
        
        # Initialize timing variables
        start_time = time.time()
        last_checkpoint = start_time
        steps_per_second = 0
        estimated_time = "calculating..."
        total_post_process_time = 0
        post_process_calls = 0
        
        # Calculate total nodes for MLUPS
        total_nodes = self.grid_shape[0] * self.grid_shape[1]
        
        for i in range(num_steps):
            self.f_0, self.f_1 = self.stepper(self.f_0, self.f_1, self.bc_mask, self.missing_mask, i)
            self.f_0, self.f_1 = self.f_1, self.f_0

            if i % post_process_interval == 0:
                if i > 0:
                    current_time = time.time()
                    elapsed = current_time - last_checkpoint
                    steps_per_second = post_process_interval / elapsed
                    mlups = (total_nodes * steps_per_second) / 1e6  # Convert to millions
                    remaining_steps = num_steps - i
                    estimated_seconds = remaining_steps / steps_per_second
                    estimated_time = str(timedelta(seconds=int(estimated_seconds)))
                    last_checkpoint = current_time
                
                    print(f"Step {i}/{num_steps} ({(i/num_steps)*100:.1f}%)")
                    print(f"Performance: {mlups:.2f} MLUPS")
                    print(f"Steps per second: {steps_per_second:.1f}")
                    print(f"Estimated time remaining: {estimated_time}")
                
                # Measure post-processing time
                post_process_time = self.post_process(i)
                total_post_process_time += post_process_time
                post_process_calls += 1
                print(f"Post-process time: {post_process_time:.3f}s")
        
        # Final timing statistics
        total_time = time.time() - start_time
        simulation_time = total_time - total_post_process_time
        average_steps_per_second = num_steps / simulation_time
        avg_post_process_time = total_post_process_time / post_process_calls if post_process_calls > 0 else 0
        avg_mlups = (total_nodes * average_steps_per_second) / 1e6
        
        print("\nSimulation completed!")
        print(f"Total time: {str(timedelta(seconds=int(total_time)))}")
        print(f"Pure simulation time: {str(timedelta(seconds=int(simulation_time)))}")
        print(f"Total post-processing time: {str(timedelta(seconds=int(total_post_process_time)))}")
        print(f"Average post-process time: {avg_post_process_time:.3f}s")
        print(f"Average performance: {avg_mlups:.2f} MLUPS")
        print(f"Grid size: {self.grid_shape[0]}x{self.grid_shape[1]} = {total_nodes} nodes")

    def post_process(self, i):
        # Tracking post processing time
        post_process_start = time.time()

        # Write the results. We'll use JAX backend for the post-processing
        if not isinstance(self.f_0, jnp.ndarray):
            # If the backend is warp, we need to drop the last dimension added by warp for 2D simulations
            f_0 = wp.to_jax(self.f_0)[..., 0]
        else:
            f_0 = self.f_0

        macro = Macroscopic(
            compute_backend=ComputeBackend.JAX,
            precision_policy=self.precision_policy,
            velocity_set=xlb.velocity_set.D2Q9(precision_policy=self.precision_policy, backend=ComputeBackend.JAX),
        )

        rho, u = macro(f_0)

        # remove boundary cells
        rho = rho[:, 1:-1, 1:-1]
        u = u[:, 1:-1, 1:-1]
        u_magnitude = (u[0] ** 2 + u[1] ** 2) ** 0.5

        fields = {"rho": rho[0], "u_x": u[0], "u_y": u[1], "u_magnitude": u_magnitude}

        save_fields_vtk(fields, timestep=i, prefix="lid_driven_cavity")
        
        # Save image with fixed colorbar range
        # Use theoretical maximum velocity as upper bound
        vmin = 0.0
        vmax = self.u_max * 1.5  # 20% margin above inlet velocity
        save_image(fields["u_magnitude"], 
                timestep=i, 
                prefix="aneurysm",
                vmin=vmin,
                vmax=vmax)
                
        post_process_time = time.time() - post_process_start
        return post_process_time

    def save_simulation_parameters(self, filename_prefix="aneurysm_params"):
        """Save simulation parameters to JSON file"""
        parameters = {
            "input_parameters": self.input_params,
            "physical": {
                "vessel_length": self.grid_shape[0] * self.resolution,
                "vessel_width": self.grid_shape[1] * self.resolution,
                "resolution": self.resolution,
                "kinematic_viscosity": self.input_params["kinematic_viscosity"],
                "max_velocity": self.u_max,
                "dt": self.dt,
                "dx": self.dx
            },
            "numerical": {
                "grid_shape": self.grid_shape,
                "omega": self.omega,
                "tau": 1/self.omega,
                "backend": str(self.backend),
                "precision_policy": str(self.precision_policy),
                "velocity_set": "D2Q9"
            },
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_nodes": self.grid_shape[0] * self.grid_shape[1]
            }
        }
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(parameters, f, indent=4)
        
        print(f"Parameters saved to {filename}")
        


def aneurysm_simulation_setup(
    vessel_length_mm=10,
    vessel_width_mm=10,
    resolution_mm=0.01,
    kinematic_viscosity=3.3e-6,  # Blood viscosity in m^2/s
    dt=5e-7,  # Time step in seconds
    u_max=0.04  # Maximum inlet velocity in lattice units
) -> AneurysmSimulation2D:
    """Setup aneurysm simulation with configurable parameters"""
    
    # Convert mm to meters
    mm_to_m = 0.001
    vessel_length_m = vessel_length_mm * mm_to_m
    vessel_width_m = vessel_width_mm * mm_to_m
    resolution_m = resolution_mm * mm_to_m
    
    # Calculate grid dimensions
    grid_x = int(round(vessel_length_m / resolution_m))
    grid_y = int(round(vessel_width_m / resolution_m))
    grid_shape = (grid_x, grid_y)

    # Simulation parameters
    backend = ComputeBackend.WARP
    precision_policy = PrecisionPolicy.FP32FP32
    
    velocity_set = xlb.velocity_set.D2Q9(
        precision_policy=precision_policy,
        backend=backend
    )
    
    # Calculate relaxation parameter (tau)
    dx = resolution_m
    nu_lbm = kinematic_viscosity * dt / (dx**2)
    omega = 1.0 / (3 * nu_lbm + 0.5)
    tau = 1/omega

    # Validate tau for stability
    assert 0.5 <= tau <= 2.0, f"Tau value {tau:.3f} out of stable range [0.5, 2.0]"
    
    # Create simulation with all parameters
    simulation = AneurysmSimulation2D(
        omega=omega,
        grid_shape=grid_shape,
        velocity_set=velocity_set,
        backend=backend,
        precision_policy=precision_policy,
        resolution=resolution_m,
        input_params={
            "vessel_length_mm": vessel_length_mm,
            "vessel_width_mm": vessel_width_mm,
            "resolution_mm": resolution_mm,
            "kinematic_viscosity": kinematic_viscosity,
            "dt": dt,
            "u_max": u_max,
            "dx": dx
        }
    )
    
    return simulation


if __name__ == "__main__":
    # Create simulation with custom parameters
    simulation = aneurysm_simulation_setup(
        vessel_length_mm=10,
        vessel_width_mm=10,
        resolution_mm=0.01,
        kinematic_viscosity=3.3e-6,
        dt=5e-7,
        u_max=0.04
    )
    
    # Run simulation
    simulation.run(10000, post_process_interval=100)

    # TODO:
    # look into the post_process method to see if it can be modified to save the results in a more useful format
    # this includes extraction of blood vessel features such as wall shear stress, velocity profiles, etc.
    # try to modify the speed to use a newtonian model, or better yet, a non-newtonian model
    # adjust the velocity to better reflect a fluid flow in a blood vessel (set 0.4)
    # look into the boundary conditions to see if they can be modified to better reflect the conditions in a blood vessel
    # expand to 3D
    # look into how much time/ the necessary conditions required to declare a flow model "stable". this may be velocity related
    # refine this and add better comments 
    # develop algorithms to remove redundant pixels from simulation results (trimming dims)