import xlb
from xlb.compute_backend import ComputeBackend
from xlb.grid import grid_factory
from xlb.operator.stepper import IncompressibleNavierStokesStepper
from xlb.operator.boundary_condition import HalfwayBounceBackBC, FullwayBounceBackBC, ZouHeBC, ExtrapolationOutflowBC
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

from boundary_conditions.direct_bc import TimeDependentZouHeBC
from stepper.custom_nse_stepper import CustomNSEStepper
from utils.wss_calculation import calculate_wss as wss_calculator


MM_TO_M = 0.001

class PipeSimulation2D:
    def __init__(self, omega, grid_shape, velocity_set, backend, precision_policy, resolution, 
                 input_params, use_time_dependent_zou_he=False, use_non_newtonian_bgk=False, 
                 output_path=None, save_png=False):
        # initialize backend
        xlb.init(
            velocity_set=velocity_set,
            default_backend=backend,
            default_precision_policy=precision_policy,
        )

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
        self.u_max = input_params["max_velocity_lu"]
        self.use_time_dependent_zou_he = use_time_dependent_zou_he
        self.use_non_newtonian_bgk = use_non_newtonian_bgk
        
        # Get the flow profile from input parameters
        self.flow_profile = input_params.get("flow_profile", {"name": "sinusoidal"})
        self.flow_profile_name = self.flow_profile.get("name", "sinusoidal")
        
        # Flag to control PNG generation
        self.save_png = save_png

        # Calculate default post-processing interval based on fps
        self.post_process_interval = max(1, int(1 / (self.input_params.get("fps", 100) * self.dt)))

        # Setup output directories with naming based on BC and collision operator flags
        if output_path is not None:
            # Use the provided output path
            self.output_dir = Path(output_path)
        else:
            # Create default output path based on BC and collision operator types
            bc_type = "tdzh" if use_time_dependent_zou_he else "zh"
            col_type = "nnbgk" if use_non_newtonian_bgk else "bgk"
            output_dirname = f"pipe_{bc_type}_{col_type}_results"
            self.output_dir = Path(f"../{output_dirname}")
        
        self.vtk_dir = self.output_dir / "vtk"
        self.img_dir = self.output_dir / "images"
        self.params_dir = self.output_dir / "parameters"
        
        # Check if directories exist and delete them if needed
        if any(d.exists() for d in [self.vtk_dir, self.img_dir, self.params_dir]):
            print(f"Output directories already exist at {self.output_dir}")
            print(f"Automatically deleting old files in {self.output_dir}")
            # Delete existing directory contents but keep the directory structure
            for directory in [self.vtk_dir, self.img_dir, self.params_dir]:
                if directory.exists():
                    for file in directory.iterdir():
                        if file.is_file():
                            file.unlink()
                        elif file.is_dir():
                            import shutil
                            shutil.rmtree(file)
        
        # Create fresh directories
        for directory in [self.vtk_dir, self.img_dir, self.params_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        # Create grid using factory
        self.grid = grid_factory(grid_shape, compute_backend=backend)
        self._setup()

        # Print configuration information
        print(f"Pipe simulation initialized with grid shape: {grid_shape}")
        print(f"Boundary condition: {'Time-dependent Zou-He' if self.use_time_dependent_zou_he else 'Standard Zou-He'}")
        print(f"Collision operator: {'Non-Newtonian BGK' if self.use_non_newtonian_bgk else 'Standard BGK'}")
        print(f"Output will be saved to: {self.output_dir}")

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
        vessel_diameter = self.input_params["vessel_diameter_lu"]  # Diameter of vessel in lattice units
        vessel_centre = self.input_params.get("vessel_centre_lu", vessel_diameter)  # Centre of vessel
        
        # Calculate pipe wall locations
        loc_upper = round(vessel_centre + vessel_diameter // 2)
        loc_lower = round(vessel_centre - vessel_diameter // 2)

        # Create straight sections for upper and lower walls
        wall_width_upper = list(range(x))
        wall_width_lower = list(range(x))
        array_upper = [loc_upper for i in range(x)]
        array_lower = [loc_lower for i in range(x)]

        # Combine all wall sections
        wall_width = wall_width_upper + wall_width_lower
        wall_height = array_upper + array_lower

        # Format for XLB
        walls = [wall_width, wall_height]

        # Get inlet/outlet
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
        
        # SHIFT ALL BOUNDARY INDICES UP BY 1 IN Y-DIRECTION
        # Save original indices for diagnostics
        original_wall_y_min = min(walls[1]) if walls[1] else None
        original_wall_y_max = max(walls[1]) if walls[1] else None
        
        # Shift wall indices up by 1
        if len(walls[1]) > 0:
            walls[1] = [y + 1 for y in walls[1]]
            
        # Shift inlet indices up by 1
        if len(inlet[1]) > 0:
            inlet[1] = [y + 1 for y in inlet[1]]
            
        # Shift outlet indices up by 1
        if len(outlet[1]) > 0:
            outlet[1] = [y + 1 for y in outlet[1]]
            
        # Print diagnostic info about the shift
        print("Boundary indices shifted up by 1 pixel in y-direction")
        print(f"Wall height range: before=[{original_wall_y_min}, {original_wall_y_max}], after=[{min(walls[1]) if walls[1] else None}, {max(walls[1]) if walls[1] else None}]")
        print(f"Inlet height range: before=[{min(y_array) if y_array else None}, {max(y_array) if y_array else None}], after=[{min(inlet[1]) if inlet[1] else None}, {max(inlet[1]) if inlet[1] else None}]")
        print(f"Outlet height range: before=[{min(y_array) if y_array else None}, {max(y_array) if y_array else None}], after=[{min(outlet[1]) if outlet[1] else None}, {max(outlet[1]) if outlet[1] else None}]")
        
        return inlet, outlet, walls

    def setup_boundary_conditions(self):
        inlet, outlet, walls = self.define_boundary_indices()
        
        # Save the boundary indices for later use in post-processing
        self.inlet_indices = inlet
        self.outlet_indices = outlet
        self.wall_indices = walls
        
        # Get the velocity profile name
        profile_name = self.input_params.get("flow_profile", {}).get("name", "sinusoidal")
        
        # Use TimeDependentZouHeBC for time-dependent boundary conditions
        if self.use_time_dependent_zou_he:
            from boundary_conditions.direct_bc import TimeDependentZouHeBC

            # Check if the profile is valid
            if not self.flow_profile:
                raise ValueError("Flow profile is not defined. Please provide a valid flow profile.")
            if not isinstance(self.flow_profile, dict):
                raise ValueError("Flow profile should be a dictionary with 'name' and 'data' keys.")
            if "x" not in self.flow_profile["data"] or "y" not in self.flow_profile["data"]:
                raise ValueError("Flow profile data must contain 'x' and 'y' keys.")
            
            print(f"Using time-dependent Zou-He BC with profile: {self.flow_profile_name}")

            # Inlet: use new direct BC
            bc_inlet = TimeDependentZouHeBC(
                bc_type="velocity",
                indices=inlet,
                dt=self.dt,
                dx=self.dx,
                u_max=self.u_max,
                flow_profile=self.flow_profile,
                frequency=1.0
            )
        else:
            print(f"USING STANDARD ZOU-HE BC")
            print(f"Max velocity (LU): {self.u_max:.3f}")
            print(self.input_params["max_velocity_lu"])

            # Standard ZouHeBC for steady-state boundary conditions
            bc_inlet = ZouHeBC(
                bc_type="velocity",
                indices=inlet,
                prescribed_value=[self.input_params.get("max_velocity_lu", 0.04), 0.0] # Prescribed velocity at inlet
            )

        # Walls: bounce-back boundary condition
        bc_walls = FullwayBounceBackBC(indices=walls)
        
        # Outlet: zero-gradient outflow
        bc_outlet = ExtrapolationOutflowBC(indices=outlet)
        
        self.boundary_conditions = [bc_walls, bc_inlet, bc_outlet]

    def setup_stepper(self):
        # Choose stepper based on flag
        if self.use_non_newtonian_bgk:
            # Use custom stepper with non-Newtonian BGK
            self.stepper = CustomNSEStepper(
                omega=self.omega,
                grid=self.grid,
                boundary_conditions=self.boundary_conditions,
                collision_type="BGKNonNewtonian"
            )
        else:
            # Use standard stepper with regular BGK
            self.stepper = CustomNSEStepper(
                omega=self.omega,
                grid=self.grid,
                boundary_conditions=self.boundary_conditions,
                collision_type="BGK"
            )
    
    def run(self, num_steps, post_process_interval=None):
        """Run simulation with detailed performance tracking"""

        # At the start of your run method
        print(f"Starting simulation with backend: {self.backend}")
        print(f"Boundary conditions: {[type(bc).__name__ for bc in self.boundary_conditions]}")
        
        # Use class attribute for post_process_interval if not specified
        if post_process_interval is None:
            post_process_interval = self.post_process_interval
            print(f"Using class default post-processing interval: {post_process_interval} steps")
            
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
                progress = i / num_steps * 100
                
                # Use recent steps for performance estimate
                recent_steps = min(post_process_interval, len(step_times))
                avg_step_time = sum(step_times[-recent_steps:]) / recent_steps
                avg_mlups = (total_nodes / avg_step_time) / 1e6
                remaining_steps = num_steps - i
                estimated_seconds = remaining_steps * avg_step_time
                
                # Status update
                print(f"\nStep {i}/{num_steps} ({progress:.1f}%)")
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

    def run_for_duration(self, duration_seconds, warmup_seconds=0.0, post_process_interval=None):
        """Run simulation for a specific duration in seconds with an optional additional warmup period.
        
        Args:
            duration_seconds (float): Simulation duration in physical time (seconds) for analysis/visualization
                                     after the warmup period
            warmup_seconds (float): Additional initial period in seconds to run before starting post-processing
                                   and visualization. This helps achieve a fully developed flow state.
            post_process_interval (int): Steps between post-processing calls, defaults to 
                                        class attribute if not specified
        
        Returns:
            The total number of steps executed (warmup + main simulation)
        """
        # Calculate number of steps needed for the full duration and warmup
        main_steps = int(round(duration_seconds / self.dt))
        warmup_steps = int(round(warmup_seconds / self.dt))
        total_steps = main_steps + warmup_steps
        
        print(f"\n=== Simulation Configuration ===")
        print(f"Running simulation for {duration_seconds:.4f} seconds of analysis time")
        if warmup_seconds > 0:
            print(f"With additional {warmup_seconds:.4f} seconds initial warmup period")
            print(f"Total physical time: {duration_seconds + warmup_seconds:.4f} seconds")
        print(f"Time step: {self.dt:.2e} seconds")
        print(f"Total steps: {total_steps:,} ({warmup_steps:,} warmup + {main_steps:,} main)")
        
        # Use class attribute for post_process_interval if not specified
        if post_process_interval is None:
            post_process_interval = self.post_process_interval
            print(f"Using class default post-processing interval: {post_process_interval} steps")
        
        # Initialize timing variables
        start_time = time.time()
        total_sim_time = 0
        total_post_process_time = 0
        post_process_calls = 0
        total_nodes = self.grid_shape[0] * self.grid_shape[1]
        
        # Performance tracking lists
        step_times = []
        mlups_history = []
        
        # Begin simulation
        print(f"Starting simulation with backend: {self.backend}")
        print(f"Boundary conditions: {[type(bc).__name__ for bc in self.boundary_conditions]}")
        
        # Run the warmup phase
        for i in range(warmup_steps):
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
                # During warmup phase, we only print status, no post-processing
                warmup_progress = i / warmup_steps * 100 if warmup_steps > 0 else 100
                overall_progress = i / total_steps * 100
                print(f"\rWarmup: Step {i}/{warmup_steps} ({warmup_progress:.1f}%) - Overall: {overall_progress:.1f}%", end="")
    
        # Print completion of warmup phase
        if warmup_steps > 0:
            print(f"\nWarmup phase completed after {warmup_steps} steps")
        
        # Run the main simulation with reset timestep counter
        for i in range(main_steps + 1):
            # Measure pure simulation step time
            step_start = time.time()
            # Use the actual simulation step for the stepper, but reset counter for post-processing
            self.f_0, self.f_1 = self.stepper(self.f_0, self.f_1, self.bc_mask, self.missing_mask, i + warmup_steps)
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
                main_progress = i / main_steps * 100
                
                # Use recent steps for performance estimate
                recent_steps = min(post_process_interval, len(step_times))
                avg_step_time = sum(step_times[-recent_steps:]) / recent_steps
                avg_mlups = (total_nodes / avg_step_time) / 1e6
                remaining_steps = main_steps - i
                estimated_seconds = remaining_steps * avg_step_time
                
                # Status update
                print(f"\nStep {i}/{main_steps} ({main_progress:.1f}%) - Analysis phase")
                print(f"Simulation Statistics:")
                print(f"├── Current MLUPS: {current_mlups:.2f}")
                print(f"├── Average MLUPS: {avg_mlups:.2f}")
                print(f"├── Step time: {step_time*1000:.2f}ms")
                print(f"└── ETA: {str(timedelta(seconds=int(estimated_seconds)))}")
                
                # Post-processing with reset timestep counter
                post_start = time.time()
                self.post_process(i)  # Use reset counter i instead of i+warmup_steps
                post_time = time.time() - post_start
                total_post_process_time += post_time
                post_process_calls += 1
                print(f"\nPost-processing time: {post_time:.3f}s")
    
        # Final statistics
        total_time = time.time() - start_time
        avg_mlups = (total_nodes * (warmup_steps + main_steps) / total_sim_time) / 1e6
        avg_post_time = total_post_process_time / post_process_calls if post_process_calls > 0 else 0
        
        print("\n=== Simulation Summary ===")
        print(f"Grid size: {self.grid_shape[0]}x{self.grid_shape[1]} = {total_nodes} nodes")
        if warmup_seconds > 0:
            print(f"Simulation phases: {warmup_seconds:.4f}s warmup + {duration_seconds:.4f}s analysis")
        print(f"\nTiming Breakdown:")
        print(f"├── Total time: {str(timedelta(seconds=int(total_time)))}")
        print(f"├── Pure simulation: {str(timedelta(seconds=int(total_sim_time)))}")
        print(f"└── Post-processing: {str(timedelta(seconds=int(total_post_process_time)))}")
        
        print(f"\nPerformance Metrics:")
        print(f"├── Average MLUPS: {avg_mlups:.2f}")
        print(f"├── Best MLUPS: {max(mlups_history):.2f}")
        print(f"├── Worst MLUPS: {min(mlups_history):.2f}")
        print(f"├── Avg step time: {(total_sim_time/(warmup_steps + main_steps))*1000:.2f}ms")
        print(f"└── Avg post-process time: {avg_post_time:.3f}s")

        # Calculate physical time simulated
        simulated_total_time = self.dt * (warmup_steps + main_steps)
        
        # Save final performance metrics
        self.save_simulation_parameters(
            final_metrics={
                "total_time": total_time,
                "total_sim_time": total_sim_time,
                "total_post_process_time": total_post_process_time,
                "avg_mlups": avg_mlups,
                "best_mlups": max(mlups_history),
                "worst_mlups": min(mlups_history),
                "avg_step_time_ms": (total_sim_time/(warmup_steps + main_steps))*1000,
                "avg_post_process_time": avg_post_time,
                "total_steps": warmup_steps + main_steps,
                "warmup_steps": warmup_steps,
                "main_steps": main_steps,
                "post_process_calls": post_process_calls,
                "total_simulation_time": simulated_total_time,
                "warmup_time": warmup_seconds,
                "analysis_time": duration_seconds,
                "boundary_condition": "time_dependent_zou_he" if self.use_time_dependent_zou_he else "standard_zou_he",
                "collision_operator": "non_newtonian_bgk" if self.use_non_newtonian_bgk else "standard_bgk"
            }
        )
        
        return warmup_steps + main_steps

    def post_process(self, i):
        """
        Post-process the simulation results and save visualization files.
        
        Args:
            i: Current timestep
            
        Returns:
            float: Time taken for post-processing
        """
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
        
        # Check for NaN or infinity values in macroscopic quantities
        if np.any(np.isnan(rho)) or np.any(np.isinf(rho)):
            raise ValueError(
                "ERROR: NaN or infinity detected in density (rho) field. This indicates numerical instability.\n"
                "Possible causes:\n"
                "1. Relaxation parameter (tau or omega) is too small or too close to 0.5\n"
                "2. Inlet velocity is too high, causing Mach number > 0.1\n"
                "3. Complex geometry with sharp corners\n"
                f"Current parameters: tau={1.0/self.omega:.3f}, omega={self.omega:.3f}, max_velocity={self.u_max:.3f}"
            )
            
        if np.any(np.isnan(u)) or np.any(np.isinf(u)):
            raise ValueError(
                "ERROR: NaN or infinity detected in velocity field. This indicates numerical instability.\n"
                "Possible causes:\n"
                "1. Relaxation parameter (tau or omega) is too small or too close to 0.5\n"
                "2. Inlet velocity is too high, causing Mach number > 0.1\n"
                "3. Complex geometry with sharp corners\n"
                f"Current parameters: tau={1.0/self.omega:.3f}, omega={self.omega:.3f}, max_velocity={self.u_max:.3f}"
            )

        # Calculate Wall Shear Stress (WSS) using the external function
        carreau_yasuda_params = {}
        
        # Extract Carreau-Yasuda parameters from collision operator if available
        if hasattr(self.stepper, 'collision') and hasattr(self.stepper.collision, 'mu_0_lu'):
            carreau_yasuda_params = {
                'mu_0': self.stepper.collision.mu_0_lu,
                'mu_inf': self.stepper.collision.mu_inf_lu,
                'lambda_cy': self.stepper.collision.lambda_cy_lu * self.stepper.collision.dt_physical,
                'n': self.stepper.collision.n,
                'a': self.stepper.collision.a
            }
            
        # Call the external WSS calculation function with updated return values
        wss_magnitude, wss_x, wss_y, wall_mask = wss_calculator(
            velocity_field=u,
            wall_indices=self.wall_indices,
            dx_physical=self.dx,
            dt_physical=self.dt,
            **carreau_yasuda_params
        )

        # Save full grid dimensions for reference
        full_shape = wss_magnitude.shape if wss_magnitude is not None else None

        # Keep all cells including boundary cells
        u_magnitude = (u[0] ** 2 + u[1] ** 2) ** 0.5
        
        # Protect against division by zero when converting to physical units
        dt_safe = max(self.dt, 1.0e-10)  # Ensure dt is not zero
        u_magnitude_physical = u_magnitude * (self.dx/dt_safe)

        fields = {
            "rho": rho[0], 
            "u_x": u[0] * (self.dx/dt_safe), 
            "u_y": u[1] * (self.dx/dt_safe), 
            "u_magnitude": u_magnitude_physical
        }

        # Add WSS fields and wall mask if available
        if wss_magnitude is not None:
            fields["wss_magnitude"] = wss_magnitude
            fields["wss_x"] = wss_x
            fields["wss_y"] = wss_y
            # Convert boolean wall mask to integer (0/1) for VTK output
            fields["wall_mask"] = wall_mask.astype(np.int32)
            
            print(f"WSS magnitude range: {np.min(wss_magnitude):.3e} to {np.max(wss_magnitude):.3e} Pa")
            print(f"WSS x-component range: {np.min(wss_x):.3e} to {np.max(wss_x):.3e} Pa")
            print(f"WSS y-component range: {np.min(wss_y):.3e} to {np.max(wss_y):.3e} Pa")
            print(f"Wall cells count: {np.sum(wall_mask)}")
            print(f"WSS field shape: {wss_magnitude.shape}")
            print(f"Other fields shape: {u_magnitude_physical.shape}")

            # Zero out velocity components inside wall mask
            original_u_x = fields["u_x"].copy()
            original_u_y = fields["u_y"].copy()
            original_u_mag = fields["u_magnitude"].copy()
            
            # Create masks for inlet and outlet from boundary indices
            inlet_mask = np.zeros_like(wall_mask, dtype=bool)
            outlet_mask = np.zeros_like(wall_mask, dtype=bool)
            
            # Convert inlet and outlet indices to masks
            if hasattr(self, 'inlet_indices') and self.inlet_indices:
                inlet_x, inlet_y = self.inlet_indices
                for x, y in zip(inlet_x, inlet_y):
                    if 0 <= x < inlet_mask.shape[0] and 0 <= y < inlet_mask.shape[1]:
                        inlet_mask[x, y] = True
                        
            if hasattr(self, 'outlet_indices') and self.outlet_indices:
                outlet_x, outlet_y = self.outlet_indices
                for x, y in zip(outlet_x, outlet_y):
                    if 0 <= x < outlet_mask.shape[0] and 0 <= y < outlet_mask.shape[1]:
                        outlet_mask[x, y] = True
            
            # Apply wall mask to velocities using JAX's at[] syntax for immutable arrays
            if isinstance(fields["u_x"], jnp.ndarray):
                # JAX arrays need special handling
                fields["u_x"] = fields["u_x"].at[wall_mask].set(0.0)
                fields["u_y"] = fields["u_y"].at[wall_mask].set(0.0)
                fields["u_magnitude"] = fields["u_magnitude"].at[wall_mask].set(0.0)
                
                # Also zero out velocities at inlet and outlet
                fields["u_x"] = fields["u_x"].at[inlet_mask].set(0.0)
                fields["u_y"] = fields["u_y"].at[inlet_mask].set(0.0)
                fields["u_magnitude"] = fields["u_magnitude"].at[inlet_mask].set(0.0)
                
                fields["u_x"] = fields["u_x"].at[outlet_mask].set(0.0)
                fields["u_y"] = fields["u_y"].at[outlet_mask].set(0.0)
                fields["u_magnitude"] = fields["u_magnitude"].at[outlet_mask].set(0.0)
            else:
                # NumPy arrays can use direct assignment
                fields["u_x"][wall_mask] = 0.0
                fields["u_y"][wall_mask] = 0.0
                fields["u_magnitude"][wall_mask] = 0.0
                
                # Also zero out velocities at inlet and outlet
                fields["u_x"][inlet_mask] = 0.0
                fields["u_y"][inlet_mask] = 0.0
                fields["u_magnitude"][inlet_mask] = 0.0
                
                fields["u_x"][outlet_mask] = 0.0
                fields["u_y"][outlet_mask] = 0.0
                fields["u_magnitude"][outlet_mask] = 0.0
            
            # Calculate how many cells were modified
            modified_wall_cells = np.sum(wall_mask)
            modified_inlet_cells = np.sum(inlet_mask)
            modified_outlet_cells = np.sum(outlet_mask)
            total_modified_cells = modified_wall_cells + modified_inlet_cells + modified_outlet_cells
            
            print(f"Zeroed velocity components in {modified_wall_cells} wall cells, {modified_inlet_cells} inlet cells, {modified_outlet_cells} outlet cells (total: {total_modified_cells})")
            
            # Report velocity ranges before and after masking
            print(f"u_x range: before=[{np.min(original_u_x):.3e}, {np.max(original_u_x):.3e}], after=[{np.min(fields['u_x']):.3e}, {np.max(fields['u_x']):.3e}]")
            print(f"u_y range: before=[{np.min(original_u_y):.3e}, {np.max(original_u_y):.3e}], after=[{np.min(fields['u_y']):.3e}, {np.max(fields['u_y']):.3e}]")
            print(f"u_magnitude range: before=[{np.min(original_u_mag):.3e}, {np.max(original_u_mag):.3e}], after=[{np.min(fields['u_magnitude']):.3e}, {np.max(fields['u_magnitude']):.3e}]")
            
            # Add inlet and outlet masks to fields for visualization
            fields["inlet_mask"] = inlet_mask.astype(np.int32)
            fields["outlet_mask"] = outlet_mask.astype(np.int32)
                
        # Save VTK file in vtk subdirectory
        vtk_path = self.vtk_dir
        bc_type = "tdzh" if self.use_time_dependent_zou_he else "zh"
        col_type = "nnbgk" if self.use_non_newtonian_bgk else "bgk"
        prefix = f"pipe_{bc_type}_{col_type}"
        save_fields_vtk(fields, output_dir=vtk_path, timestep=i, prefix=prefix)
        
        # Only generate PNG images if save_png flag is True
        if self.save_png:
            # Save image with fixed colorbar range in images subdirectory
            # Use theoretical maximum velocity as upper bound
            vmin = 0.0
            # Convert to physical units for the comment (LU * dx/dt = m/s)
            vmax = self.u_max * 1.5     # 50% margin over max velocity
            print(f"Saving PNGs with vmin={vmin}, vmax={vmax}")
            print("rho values:", np.min(fields["rho"]), np.max(fields["rho"]))
            print("u_x values:", np.min(fields["u_x"]), np.max(fields["u_x"]))
            print("u_y values:", np.min(fields["u_y"]), np.max(fields["u_y"]))
            print("u_magnitude values:", np.min(fields["u_magnitude"]), np.max(fields["u_magnitude"]))

            # Generate velocity field image
            save_image(
                fields["u_magnitude"],
                prefix=str(self.img_dir / f"{prefix}_"),
                timestep=i,
                vmin=vmin,
                vmax=vmax
            )
            
            # Generate WSS magnitude image if available
            if wss_magnitude is not None:
                # Use a reasonable scale for WSS
                wss_vmax = np.max(wss_magnitude) * 1.1  # 10% margin
                save_image(
                    fields["wss_magnitude"],
                    prefix=str(self.img_dir / f"{prefix}_wss_"),
                    timestep=i,
                    vmin=0.0,
                    vmax=wss_vmax,
                    cmap='hot'  # Use a different colormap for WSS
                )
                
                # Generate wall mask image
                save_image(
                    fields["wall_mask"],
                    prefix=str(self.img_dir / f"{prefix}_wall_"),
                    timestep=i,
                    vmin=0.0,
                    vmax=1.0,
                    cmap='binary'  # Use binary colormap for wall mask
                )
        else:
            print("PNG generation skipped (--generate-pngs flag not set)")
                
        post_process_time = time.time() - post_process_start
        return post_process_time
    
    def save_simulation_parameters(self, filename_prefix=None, final_metrics=None):
        """Save simulation parameters and final metrics to JSON file
        
        Args:
            filename_prefix (str): Prefix for the output JSON file
            final_metrics (dict): Dictionary containing final simulation metrics
        """
        # Helper function to convert NumPy arrays to lists for JSON serialization
        def numpy_to_json_serializable(obj):
            if isinstance(obj, (np.ndarray, np.number)):
                return obj.tolist()
            elif isinstance(obj, (list, tuple)):
                return [numpy_to_json_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: numpy_to_json_serializable(val) for key, val in obj.items()}
            elif isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            else:
                return str(obj)  # Convert other types to strings
    
        # Get collision operator details
        collision_type = "BGK" if not self.use_non_newtonian_bgk else "BGKNonNewtonian"
        collision_details = {}
        
        if hasattr(self.stepper, 'collision'):
            if hasattr(self.stepper.collision, '__class__'):
                collision_type = self.stepper.collision.__class__.__name__
                
                # Extract Carreau-Yasuda parameters if using BGKNonNewtonian
                if collision_type == "BGKNonNewtonian" and hasattr(self.stepper.collision, 'mu_0_lu'):
                    collision_details = {
                        "mu_0": self.stepper.collision.mu_0_lu,  # Zero-shear viscosity
                        "mu_inf": self.stepper.collision.mu_inf_lu,  # Infinite-shear viscosity
                        "lambda": self.stepper.collision.lambda_cy_lu * self.stepper.collision.dt_physical,  # Relaxation time
                        "n": self.stepper.collision.n,  # Power law index
                        "a": self.stepper.collision.a,  # Transition parameter
                        "dx_physical": self.stepper.collision.dx_physical,
                        "dt_physical": self.stepper.collision.dt_physical
                    }
    
        # Create custom filename prefix if not provided
        if filename_prefix is None:
            bc_type = "tdzh" if self.use_time_dependent_zou_he else "zh"
            col_type = "nnbgk" if self.use_non_newtonian_bgk else "bgk"
            filename_prefix = f"pipe_{bc_type}_{col_type}_params"
        
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
                "final_generation_time_s": final_metrics["total_simulation_time"] if final_metrics else None,     # Add simulated time if final data has been supplied
                "collision_operator": {
                    "type": collision_type,
                    "details": collision_details
                },
                "boundary_conditions": [
                    {
                        "type": bc.__class__.__name__,
                        "position": idx,
                        "location": "inlet" if isinstance(bc, (TimeDependentZouHeBC, ZouHeBC)) and idx == 1 else 
                                   "outlet" if isinstance(bc, ExtrapolationOutflowBC) else 
                                   "walls" if isinstance(bc, FullwayBounceBackBC) else "other"
                    } for idx, bc in enumerate(self.boundary_conditions)
                ],
                "flow_profile": {
                    "name": self.flow_profile_name,
                    "max_velocity": float(self.u_max),
                    "is_time_dependent": self.use_time_dependent_zou_he
                },
                "model_configuration": {
                    "time_dependent_zou_he": self.use_time_dependent_zou_he,
                    "non_newtonian_bgk": self.use_non_newtonian_bgk
                }
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
                    "steps_per_post_process": final_metrics["total_steps"] // final_metrics["post_process_calls"] if final_metrics["post_process_calls"] > 0 else 0
                }
            }
        
        # Generate filename with timestamp
        filename = self.params_dir / f"{filename_prefix}.json"
        
        # Convert all NumPy arrays and other non-serializable objects to JSON serializable types
        serializable_parameters = numpy_to_json_serializable(parameters)
        
        with open(filename, 'w') as f:
            json.dump(serializable_parameters, f, indent=4)
        
        print(f"Parameters and performance metrics saved to {filename}")
        print(f"Boundary condition: {'Time-dependent Zou-He' if self.use_time_dependent_zou_he else 'Standard Zou-He'}")
        print(f"Collision operator: {'Non-Newtonian BGK' if self.use_non_newtonian_bgk else 'Standard BGK'}")
