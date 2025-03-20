import xlb
from xlb.compute_backend import ComputeBackend
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
import os
from pathlib import Path
import jax

from direct_bc import DirectTimeDependentBC
from timestep_stepper import INSETimestepStepper
# from velocity_profiles import VelocityProfileRegistry


MM_TO_M = 0.001

class AneurysmSimulation2D:
    def __init__(self, omega, grid_shape, velocity_set, backend, precision_policy, resolution, input_params):
        # initialize backend
        xlb.init(
            velocity_set=velocity_set,
            default_backend=backend,
            default_precision_policy=precision_policy,
        )

        # Start velocity profile manager
        # self.velocity_profiles = VelocityProfileRegistry(default_backend=backend, default_precision_policy=precision_policy)

        # Store input parameters
        self.input_params = input_params
        self.grid_shape = grid_shape
        self.velocity_set = velocity_set
        self.backend = backend
        self.precision_policy = precision_policy
        self.omega = omega
        self.resolution = resolution
        
        self.dt = input_params["dt"]
        self.dx = input_params["dx"]
        self.boundary_conditions = []

        # Pop flow profile from the dict after saving to class
        self.flow_profile = self.input_params.pop("flow_profile")
        self.flow_profile_name = self.flow_profile.get("name", "Sinusoidal (default)")

        self.u_max = self.flow_profile.get("max_velocity", 0.4)   # Default max velocity is 0.4 m/s


        # Setup output directories
        self.output_dir = Path("aneurysm_simulation")
        self.vtk_dir = self.output_dir / "vtk"
        self.img_dir = self.output_dir / "images"
        self.params_dir = self.output_dir / "parameters"
        
       # Check if directories exist and ask for cleanup
        if any(d.exists() for d in [self.vtk_dir, self.img_dir, self.params_dir]):
            response = input("Output directories exist. Would you like to clear them? (y/n): ").lower()
            if response == 'y':
                print("Cleaning up previous simulation outputs...")
                import shutil
                if self.output_dir.exists():
                    shutil.rmtree(self.output_dir)
                print("Cleanup complete.")
        
        # Create fresh directories
        for directory in [self.vtk_dir, self.img_dir, self.params_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        # Create grid using factory
        self.grid = grid_factory(grid_shape, compute_backend=backend)
        self._setup()

    def _setup(self):
        self.setup_boundary_conditions()
        # wait for boundary conditions to be set up
        self.setup_stepper()
        # Initialize fields using the stepper
        self.f_0, self.f_1, self.bc_mask, self.missing_mask = self.stepper.prepare_fields()

    def define_boundary_indices(self):
        x, y = self.grid_shape

        # Retrieve vessel parameters
        vessel_length = self.input_params["vessel_length_lu"]  # Length of vessel in lattice units
        vessel_diameter = self.input_params["vessel_diameter_lu"]    # Diameter of vessel in lattice units
        bulge_horizontal_diameter = self.input_params["bulge_horizontal_lu"]  # Horizontal diameter of aneurysm bulge
        bulge_vertical_diameter = self.input_params["bulge_vertical_lu"]      # Vertical diameter of aneurysm bulge
        bulge_centre_x = self.input_params["bulge_centre_x_lu"]  # Horizontal centre of bulge
        bulge_centre_y = self.input_params["bulge_centre_y_lu"]    # Vertical centre of bulge
        vessel_centre = self.input_params["vessel_centre_lu"]  # Centre of vessel

        # print("Vessel length:", vessel_length)
        # print("Vessel diameter:", vessel_diameter)
        # print("Bulge horizontal diameter:", bulge_horizontal_diameter)
        # print("Bulge vertical diameter:", bulge_vertical_diameter)
        # print("Bulge centre x:", bulge_centre_x)
        # print("Bulge centre y:", bulge_centre_y)

        # print(self.grid_shape)


        # # Ovoid parameters
        x0 = bulge_centre_x   # centre x position
        y0 = bulge_centre_y  # centre y position
        a = bulge_horizontal_diameter // 2                   # semi-major axis
        b = bulge_vertical_diameter                 # semi-minor axis
        
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
        # print("Number of curve points:", len(curve_x))
        # print("X range:", min(curve_x), "to", max(curve_x))
        # print("Y range:", min(curve_y), "to", max(curve_y))

        # Format for XLB
        curve_wall = [list(curve_x), list(curve_y)]
        # print("Curve wall:", curve_wall)

        loc_upper = round(vessel_centre + vessel_diameter // 2) 
        loc_lower = round(vessel_centre - vessel_diameter // 2)

        # print("loc_upper:", loc_upper)
        # print("loc_lower:", loc_lower)

        
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



        ####### OVERRIDE #######
        # No buldge, just a pipe
        array_upper = [loc_upper for i in range(x)]

        wall_width = (wall_width_lower +
                    wall_width_lower)
        
        wall_height = (array_upper + 
                    array_lower)






        # Format for XLB
        walls = [wall_width, wall_height]
        

        # Get inlet/outlet from box_no_edge (similar to how lid-driven cavity gets its lid)
        x_array_left = [0 for i in range(loc_lower + 1, loc_upper)]
        x_array_right = [x - 1 for i in range(loc_lower + 1, loc_upper)]
        y_array = [i for i in range(loc_lower + 1, loc_upper)]
        inlet = [x_array_left, y_array]
        outlet = [x_array_right, y_array]

        # Convert to numpy arrays and ensure proper formatting
        walls = np.array([wall_width, wall_height], dtype=np.int32)
        walls = np.unique(walls, axis=-1).tolist()
        
        inlet = np.array([x_array_left, y_array], dtype=np.int32).tolist()
        outlet = np.array([x_array_right, y_array], dtype=np.int32).tolist()
        
        # Debug prints
        # print("Inlet:", inlet)
        # print("Outlet:", outlet)
        # print("Walls:", walls)
        
        return inlet, outlet, walls

    def setup_boundary_conditions(self):
        inlet, outlet, walls = self.define_boundary_indices()
        
        # Inlet: use new direct BC
        bc_inlet = DirectTimeDependentBC(
            bc_type="velocity",
            indices=inlet,
            dt=self.dt,
            dx=self.dx,
            u_max=self.u_max,
            flow_profile=self.flow_profile,
            frequency=1.0
        )


        # Walls: no-slip boundary condition
        bc_walls = FullwayBounceBackBC(indices=walls)
        
        # Inlet: constant velocity profile
        # bc_inlet = TimeDependentZouHeBC("velocity", profile=self.bc_profile(), indices=inlet)
        
        


        # bc_inlet = ZouHeBC("velocity", profile=self.bc_profile(), indices=inlet)
        # bc_inlet = TimeDependentZouHeBC("velocity", profile=self.velocity_profiles.get("ICA"), indices=inlet)
        # bc_inlet = ZouHeBC("velocity", prescribed_value=(0.0, self.u_max), indices=inlet)
        
        # self.u_max = bc_inlet.u_max
        # print("u_max:", self.u_max)
        
        # Outlet: zero-gradient outflow
        bc_outlet = ExtrapolationOutflowBC(indices=outlet)
        
        # self.boundary_conditions = [bc_inlet]
        self.boundary_conditions = [bc_walls, bc_inlet, bc_outlet]

    def setup_stepper(self):
        # self.stepper = IncompressibleNavierStokesStepper(
        #     omega=self.omega,
        #     grid=self.grid,
        #     boundary_conditions=self.boundary_conditions,
        #     collision_type="BGK"
        # )     

        # New custom stepper with timestep support
        self.stepper = INSETimestepStepper(
            omega=self.omega,
            grid=self.grid,
            boundary_conditions=self.boundary_conditions,
            collision_type="BGK"
        )
    

    # def bc_profile(self, time):
    #     time = wp.static(self.precision_policy.store_precision.wp_dtype(time))
    #     u_max = wp.static(self.precision_policy.store_precision.wp_dtype(self.u_max))
    #     dt = wp.static(self.precision_policy.store_precision.wp_dtype(self.dt))
    #     omega = wp.static(self.precision_policy.store_precision.wp_dtype(2.0 * np.pi))

    #     @wp.func
    #     def bc_profile_warp(index: wp.vec3i):  # Optional timestep parameter
    #         # Calculate time from actual timestep
    #         t = dt * wp.float32(time)
    #         # Sinusoidal velocity with positive offset
    #         u_x = u_max * (0.5 + 0.5 * wp.sin(omega * t))
    #         return wp.vec(u_x, length=1)

    #     def bc_profile_jax():
    #         def velocity(time):
    #             t = self.dt * time
    #             u_x = self.u_max * (0.5 + 0.5 * jnp.sin(omega * t))
    #             u_y = jnp.zeros_like(u_x)
    #             return jnp.array([u_x, u_y])
    #         return velocity

    #     if self.backend == ComputeBackend.JAX:
    #         return bc_profile_jax
    #     elif self.backend == ComputeBackend.WARP:
    #         return bc_profile_warp


    def bc_profile(self):
        u_max = wp.static(self.precision_policy.store_precision.wp_dtype(self.u_max))
        dt = wp.static(self.precision_policy.store_precision.wp_dtype(self.dt))
        omega = wp.static(self.precision_policy.store_precision.wp_dtype(2.0 * np.pi * 20))

        @wp.func
        def bc_profile_warp(index: wp.vec3i, timestep: wp.int32=10):  # Optional timestep parameter
            # Calculate time from actual timestep
            t = dt * wp.float32(timestep)
            # Sinusoidal velocity with positive offset
            u_x = u_max * (0.5 + 0.5 * wp.sin(omega * t))
            # u_x = u_max * 0.5 * t
            return wp.vec(u_x, length=1)

        def bc_profile_jax():
            def velocity(timestep):
                t = self.dt * timestep
                u_x = self.u_max * (0.5 + 0.5 * jnp.sin(omega * t))
                u_y = jnp.zeros_like(u_x)
                return jnp.array([u_x, u_y])
            return velocity

        if self.backend == ComputeBackend.JAX:
            return bc_profile_jax
        elif self.backend == ComputeBackend.WARP:
            return bc_profile_warp

    # def bc_profile(self):
    #     # Convert values to static WARP types for kernel use
    #     u_max = wp.static(self.precision_policy.store_precision.wp_dtype(self.u_max))
    #     dt = wp.static(self.precision_policy.store_precision.wp_dtype(self.dt))
    #     omega = wp.static(self.precision_policy.store_precision.wp_dtype(2.0 * np.pi * 2345))

    #     @wp.func
    #     def bc_profile_warp(index: wp.vec3i):  # Note: only takes index parameter
    #         # Calculate time from x-component of index
    #         t = dt * wp.float32(index[0])
    #         # Sinusoidal velocity with positive offset
    #         u_x = u_max * (0.5 + 10.5 * wp.sin(omega * t))
    #         # Return single value as vector
    #         return wp.vec(u_x, length=1)

    #     def bc_profile_jax():
    #         def velocity(timestep):
    #             t = self.dt * timestep
    #             u_x = self.u_max * (0.5 + 0.5 * jnp.sin(omega * t))
    #             u_y = jnp.zeros_like(u_x)
    #             return jnp.array([u_x, u_y])
    #         return velocity

    #     if self.backend == ComputeBackend.JAX:
    #         return bc_profile_jax
    #     elif self.backend == ComputeBackend.WARP:
    #         return bc_profile_warp

    
    # PARTIALLY COMPLETED POUSIELLE FLOW PROFILE
    # def bc_profile(self):
    #     u_max = self.u_max  # u_max = 0.04
    #     # Get the grid dimensions for the y and z directions
    #     H_y = float(self.grid_shape[1] - 1)  # Height in y direction
    #     # H_z = float(self.grid_shape[2] - 1)  # Height in z direction

    #     @wp.func
    #     def bc_profile_warp(index: wp.vec3i):
    #         # Poiseuille flow profile: parabolic velocity distribution
    #         y = self.precision_policy.store_precision.wp_dtype(index[1])
    #         # z = self.precision_policy.store_precision.wp_dtype(index[2])

    #         # Calculate normalized distance from center
    #         y_center = y - (H_y / 2.0)
    #         r_squared = (2.0 * y_center / H_y) ** 2.0

    #         # Parabolic profile: u = u_max * (1 - r²)
    #         return wp.vec(u_max * wp.max(0.0, 1.0 - r_squared), length=1)

    #     def bc_profile_jax():
    #         y = jnp.arange(self.grid_shape[1])

    #         # Calculate normalized distance from center
    #         y_center = y - (H_y / 2.0)
    #         r_squared = (2.0 * y_center / H_y) ** 2.0

    #         # Parabolic profile for x velocity, zero for y and z
    #         u_x = u_max * jnp.maximum(0.0, 1.0 - r_squared)
    #         u_y = jnp.zeros_like(u_x)

    #         return jnp.stack([u_x, u_y])

    #     if self.backend == ComputeBackend.JAX:
    #         return bc_profile_jax
    #     elif self.backend == ComputeBackend.WARP:
    #         return bc_profile_warp

    def run(self, num_steps, post_process_interval=100):
        """Run simulation with detailed performance tracking"""

        # At the start of your run method
        print(f"Starting simulation with backend: {self.backend}")
        print(f"Boundary conditions: {[type(bc).__name__ for bc in self.boundary_conditions]}")
        # Initialize timing variables
        start_time = time.time()
        last_sim_step_time = start_time
        total_sim_time = 0
        total_post_process_time = 0
        post_process_calls = 0
        total_nodes = self.grid_shape[0] * self.grid_shape[1]
        
        # Performance tracking lists
        step_times = []
        mlups_history = []
        
        for i in range(num_steps + 1):

            # # update bc profile
            # self.bc_inlet = self.bc_inlet = ZouHeBC("velocity", profile=self.bc_profile(time=i), indices=self.inlet)
            # self.bc_mask, self.missing_mask = self.stepper._process_boundary_conditions([self.bc_inlet], self.bc_mask, self.missing_mask)
            # self.f_0, self.f_1 = self.stepper._initialize_auxiliary_data([self.bc_inlet], self.f_0, self.f_1, self.bc_mask, self.missing_mask)


            # Update timestep for time-dependent boundary conditions
            for bc in self.boundary_conditions:
                if hasattr(bc, 'update_timestep'):
                    bc.update_timestep(i)

            # Measure pure simulation step time
            step_start = time.time()
            self.f_0, self.f_1 = self.stepper(self.f_0, self.f_1, self.bc_mask, self.missing_mask, i)
            self.f_0, self.f_1 = self.f_1, self.f_0
            step_end = time.time()
            
            # Track step performance
            step_time = step_end - step_start
            step_times.append(step_time)
            total_sim_time += step_time
            
            # Calculate running MLUPS
            current_mlups = (total_nodes / step_time) / 1e6
            mlups_history.append(current_mlups)

            if i % post_process_interval == 0:
                # Calculate statistics
                avg_step_time = sum(step_times[-post_process_interval:]) / len(step_times[-post_process_interval:])
                avg_mlups = (total_nodes / avg_step_time) / 1e6
                remaining_steps = num_steps - i
                estimated_seconds = remaining_steps * avg_step_time
                
                # Status update
                print(f"\nStep {i}/{num_steps} ({(i/num_steps)*100:.1f}%)")
                print(f"Simulation Statistics:")
                print(f"├── Current MLUPS: {current_mlups:.2f}")
                print(f"├── Average MLUPS: {avg_mlups:.2f}")
                print(f"├── Step time: {step_time*1000:.2f}ms")
                print(f"└── ETA: {str(timedelta(seconds=int(estimated_seconds)))}")
                
                # Post-processing
                post_start = time.time()
                self.post_process(i)
                post_time = time.time() - post_start
                total_post_process_time += post_time
                post_process_calls += 1
                print(f"\nPost-processing time: {post_time:.3f}s")
        
        # Final statistics
        total_time = time.time() - start_time
        avg_mlups = (total_nodes * num_steps / total_sim_time) / 1e6
        avg_post_time = total_post_process_time / post_process_calls if post_process_calls > 0 else 0
        
        print("\n=== Simulation Summary ===")
        print(f"Grid size: {self.grid_shape[0]}x{self.grid_shape[1]} = {total_nodes} nodes")
        print(f"\nTiming Breakdown:")
        print(f"├── Total time: {str(timedelta(seconds=int(total_time)))}")
        print(f"├── Pure simulation: {str(timedelta(seconds=int(total_sim_time)))}")
        print(f"└── Post-processing: {str(timedelta(seconds=int(total_post_process_time)))}")
        
        print(f"\nPerformance Metrics:")
        print(f"├── Average MLUPS: {avg_mlups:.2f}")
        print(f"├── Best MLUPS: {max(mlups_history):.2f}")
        print(f"├── Worst MLUPS: {min(mlups_history):.2f}")
        print(f"├── Avg step time: {(total_sim_time/num_steps)*1000:.2f}ms")
        print(f"└── Avg post-process time: {avg_post_time:.3f}s")

        simulated_total_time = self.dt * num_steps

        # Save final performance metrics
        self.save_simulation_parameters(
            final_metrics={
                "total_time": total_time,
                "total_sim_time": total_sim_time,
                "total_post_process_time": total_post_process_time,
                "avg_mlups": avg_mlups,
                "best_mlups": max(mlups_history),
                "worst_mlups": min(mlups_history),
                "avg_step_time_ms": (total_sim_time/num_steps)*1000,
                "avg_post_process_time": avg_post_time,
                "total_steps": num_steps,
                "post_process_calls": post_process_calls,
                "total_simulation_time": simulated_total_time
            }
        )

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
        u_magnitude_physical = u_magnitude * (self.dx/self.dt)

        fields = {
            "rho": rho[0], 
            "u_x": u[0] * (self.dx/self.dt), 
            "u_y": u[1] * (self.dx/self.dt), 
            "u_magnitude": u_magnitude_physical
        }

        # Save VTK file in vtk subdirectory
        vtk_path = self.vtk_dir
        save_fields_vtk(fields, output_dir=vtk_path, timestep=i, prefix="aneurysm")
        
        # Save image with fixed colorbar range in images subdirectory
        # Use theoretical maximum velocity as upper bound
        vmin = 0.0
        vmax = self.u_max * 1.5     # 50% margin max velocity
        print(self.u_max)
        print(self.dx, self.dt, self.dx/self.dt)
        print(f"Saving image with vmin={vmin}, vmax={vmax}")
        print("rho values:", np.min(fields["rho"]), np.max(fields["rho"]))
        print("u_x values:", np.min(fields["u_x"]), np.max(fields["u_x"]))
        print("u_y values:", np.min(fields["u_y"]), np.max(fields["u_y"]))
        print("u_magnitude values:", np.min(fields["u_magnitude"]), np.max(fields["u_magnitude"]))

        # print(self.img_dir)

        save_image(
            fields["u_magnitude"],
            prefix=str(self.img_dir / f"aneurysm_"),
            timestep=i,
            vmin=vmin,
            vmax=vmax
        )
                
        post_process_time = time.time() - post_process_start
        return post_process_time

    def save_simulation_parameters(self, filename_prefix="aneurysm_params", final_metrics=None):
        """Save simulation parameters and final metrics to JSON file
        
        Args:
            filename_prefix (str): Prefix for the output JSON file
            final_metrics (dict): Dictionary containing final simulation metrics
        """
        parameters = {
            "input_parameters": self.input_params,
            "physical": {
                "vessel_length": self.grid_shape[0] * self.resolution,
                "vessel_width": self.grid_shape[1] * self.resolution,
                "resolution": self.resolution,
                "kinematic_viscosity": self.input_params["kinematic_viscosity"],
                "max_velocity": float(self.u_max),
                "dt": self.dt,
                "dx": self.dx,
                "fps": self.input_params["fps"]
            },
            "numerical": {
                "grid_shape": self.grid_shape,
                "omega": self.omega,
                "tau": 1/self.omega,
                "backend": str(self.backend),
                "precision_policy": str(self.precision_policy),
                "velocity_set": "D2Q9",
                "total_nodes": self.grid_shape[0] * self.grid_shape[1],
                "final_generation_time_s": final_metrics["total_simulation_time"] if final_metrics else None      # Add simulated time if final data has been supplied
            },
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "hostname": os.uname().nodename,
                "output_directories": {
                    "vtk": str(self.vtk_dir),
                    "images": str(self.img_dir),
                    "parameters": str(self.params_dir)
                }
            }
        }
        
        # Add final metrics if available
        if final_metrics:
            parameters["performance"] = {
                "timing": {
                    "total_runtime": str(timedelta(seconds=int(final_metrics["total_time"]))),
                    "simulation_time": str(timedelta(seconds=int(final_metrics["total_sim_time"]))),
                    "post_processing_time": str(timedelta(seconds=int(final_metrics["total_post_process_time"])))
                },
                "performance_metrics": {
                    "average_mlups": round(final_metrics["avg_mlups"], 2),
                    "peak_mlups": round(final_metrics["best_mlups"], 2),
                    "minimum_mlups": round(final_metrics["worst_mlups"], 2),
                    "average_step_time_ms": round(final_metrics["avg_step_time_ms"], 3),
                    "average_post_process_time_s": round(final_metrics["avg_post_process_time"], 3)
                },
                "execution_stats": {
                    "total_steps": final_metrics["total_steps"],
                    "post_process_calls": final_metrics["post_process_calls"],
                    "steps_per_post_process": final_metrics["total_steps"] // final_metrics["post_process_calls"]
                }
            }
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%H:%M:%S %d-%m-%Y")
        filename = self.params_dir / f"{filename_prefix} {timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(parameters, f, indent=4)
        
        print(f"Parameters and performance metrics saved to {filename}")
        


