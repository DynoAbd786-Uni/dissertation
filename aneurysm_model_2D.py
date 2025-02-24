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
import os
from pathlib import Path


MM_TO_M = 0.001

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
            boundary_conditions=self.boundary_conditions
        )

    def run(self, num_steps, post_process_interval=100):
        """Run simulation with detailed performance tracking"""
        
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
        
        for i in range(num_steps):
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

        fields = {"rho": rho[0], "u_x": u[0], "u_y": u[1], "u_magnitude": u_magnitude}

        # Save VTK file in vtk subdirectory
        vtk_path = self.vtk_dir
        save_fields_vtk(fields, output_dir=vtk_path, timestep=i, prefix="aneurysm")
        
        # Save image with fixed colorbar range in images subdirectory
        # Use theoretical maximum velocity as upper bound
        vmin = 0.0
        vmax = self.input_params["u_max"] * 1.5  # 50% margin above inlet velocity
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
                "max_velocity": self.u_max,
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
                },
                "final generation time": final_metrics["total_simulation_time"]
            }
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%H:%M:%S %d-%m-%Y")
        filename = self.params_dir / f"{filename_prefix} {timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(parameters, f, indent=4)
        
        print(f"Parameters and performance metrics saved to {filename}")
        


def aneurysm_simulation_setup(
    vessel_length_mm=10,
    vessel_diameter_mm=4,
    bulge_horizontal_mm=6,
    bulge_vertical_mm=4,
    resolution_mm=0.01,
    kinematic_viscosity=3.3e-6,
    dt=5e-7,
    u_max=0.04,
    fps=100
) -> AneurysmSimulation2D:
    """Setup aneurysm simulation with configurable parameters"""
    
    # Convert mm to meters
    mm_to_m = 0.001
    vessel_length_m = vessel_length_mm * mm_to_m
    vessel_diameter_m = vessel_diameter_mm * mm_to_m
    resolution_m = resolution_mm * mm_to_m
    
    # Calculate base grid dimensions (without bulge)
    grid_x = int(round(vessel_length_m / resolution_m))
    grid_y = int(round(vessel_diameter_m / resolution_m))
    
    # Calculate bulge dimensions in lattice units
    bulge_horizontal_lu = int(round(bulge_horizontal_mm * mm_to_m / resolution_m))
    bulge_vertical_lu = int(round((bulge_vertical_mm / 2) * mm_to_m / resolution_m))
    
    # Final grid shape including bulge
    grid_shape = (grid_x + 1, grid_y + bulge_vertical_lu + 1) # Add 1 for boundary cells
    
    # Calculate vessel centerline
    vessel_centre_lu = grid_y // 2
    
    # Simulation parameters
    backend = ComputeBackend.WARP
    precision_policy = PrecisionPolicy.FP32FP32
    
    velocity_set = xlb.velocity_set.D2Q9(
        precision_policy=precision_policy,
        backend=backend
    )
    
    # Calculate relaxation parameter
    dx = resolution_m
    nu_lbm = kinematic_viscosity * dt / (dx**2)
    omega = 1.0 / (3 * nu_lbm + 0.5)
    
    # Validate tau for stability
    tau = 1/omega
    assert 0.5 <= tau <= 2.0, f"Tau value {tau:.3f} out of stable range [0.5, 2.0]"
    
    # Post-processing interval
    post_process_interval = max(1, int(1 / (fps * dt)))
    
    # Create input parameters dictionary
    input_params = {
        "vessel_length_mm": vessel_length_mm,
        "vessel_diameter_mm": vessel_diameter_mm,
        "bulge_horizontal_mm": bulge_horizontal_mm,
        "bulge_vertical_mm": bulge_vertical_mm,
        "resolution_mm": resolution_mm,
        "kinematic_viscosity": kinematic_viscosity,
        "dt": dt,
        "u_max": u_max,
        "dx": dx,
        "fps": fps,
        "vessel_length_lu": grid_x,
        "vessel_diameter_lu": grid_y,
        "bulge_horizontal_lu": bulge_horizontal_lu,
        "bulge_vertical_lu": bulge_vertical_lu,
        "vessel_centre_lu": vessel_centre_lu,
        "bulge_centre_x_lu": grid_x // 2,
        "bulge_centre_y_lu": vessel_centre_lu + (grid_y // 2)
    }
    
    # Create simulation
    simulation = AneurysmSimulation2D(
        omega=omega,
        grid_shape=grid_shape,
        velocity_set=velocity_set,
        backend=backend,
        precision_policy=precision_policy,
        resolution=resolution_mm,
        input_params=input_params
    )
    
    return simulation, post_process_interval

if __name__ == "__main__":
    # Create simulation with realistic vessel parameters
    simulation, post_process_interval = aneurysm_simulation_setup(
        vessel_length_mm=10,         # 10mm vessel length
        vessel_diameter_mm=2,        # 2mm vessel diameter (typical cerebral artery)
        bulge_horizontal_mm=6,       # 6mm horizontal bulge
        bulge_vertical_mm=4,         # 4mm vertical bulge
        resolution_mm=0.01,          # 0.01mm resolution
        kinematic_viscosity=3.3e-6,  # Blood viscosity
        dt=5e-7,                     # Time step
        u_max=0.04,                   # More realistic blood velocity TODO: remove in place for a better flow model
        fps=1000                      # Output frames per second
    )
    
    # Run simulation
    simulation.run(10000, post_process_interval=post_process_interval)

    # TODO:
    # look into the post_process method to see if it can be modified to save the results in a more useful format
    # this includes extraction of blood vessel features such as wall shear stress, velocity profiles, etc.
    # try to modify the speed to use a newtonian model, or better yet, a non-newtonian model
    # adjust the velocity to better reflect a fluid flow in a blood vessel (set 0.4)
    # look into the boundary conditions to see if they can be modified to better reflect the conditions in a blood vessel
    # expand to 3D timestep