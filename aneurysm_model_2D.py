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


class AneurysmSimulation2D:
    def __init__(self, omega, grid_shape, velocity_set, backend, precision_policy, resolution):
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
        self.u_max = 0.04  # Maximum inlet velocity
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
        for i in range(num_steps):
            self.f_0, self.f_1 = self.stepper(self.f_0, self.f_1, self.bc_mask, self.missing_mask, i)
            self.f_0, self.f_1 = self.f_1, self.f_0

            if i % post_process_interval == 0 or i == num_steps - 1:
                self.post_process(i)

    def post_process(self, i):
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
        save_image(fields["u_magnitude"], timestep=i, prefix="lid_driven_cavity")


def aneurysm_simulation_setup() -> AneurysmSimulation2D:
    # Constants 
    # TODO: move to more appropriate place
    mm_to_m = 0.001

    # Physical parameters
    # Ideally, for the final protoype, we find the smallest possible values for these parameters
    # based on the actual physical dimensions of the aneurysm model
    # to reduce the computational cost
    vessel_length_m = 10 * mm_to_m    # 10 mm long
    vessel_width_m = 10 * mm_to_m      # 10 mm wide
    resolution_m = 0.01 * mm_to_m     # Each grid cell = 0.01 mm
    
    # Calculate grid dimensions
    grid_x = int(round(vessel_length_m / resolution_m))
    grid_y = int(round(vessel_width_m / resolution_m))
    grid_shape = (grid_x, grid_y)

    print(grid_shape)
    
    # Simulation parameters
    backend = ComputeBackend.WARP
    precision_policy = PrecisionPolicy.FP32FP32
    # kinematic_viscosity = 3.3e-6 * units.m**2/units.s  # Blood viscosity
    
    # Velocity set for 2D
    velocity_set = xlb.velocity_set.D2Q9(
        precision_policy=precision_policy,
        backend=backend
    )
    
    # Calculate relaxation parameter (tau) from physical properties
    kinematic_viscosity = 3.3e-6  # Blood viscosity in m^2/s
    dx = resolution_m  # Grid spacing in meters
    dt = 1e-7  # Time step in seconds
    nu_lbm = kinematic_viscosity * dt / (dx**2)
    omega = 1.0 / (3 * nu_lbm + 0.5)

    print((3 * nu_lbm + 0.5))   # Tau value. should be in range 0.5 to 2.0 for stability (source: https://www.researchgate.net/publication/320000000_Lattice_Boltzmann_Methods_for_Complex_Flows_in_Microgeometries)
    assert 0.5 <= (3 * nu_lbm + 0.5) <= 2.0, "Tau value out of range!"
    print(f"omega = {omega}")

    velocity_set = xlb.velocity_set.D2Q9(precision_policy=precision_policy, backend=backend)

    simulation = AneurysmSimulation2D(omega, grid_shape, velocity_set, backend, precision_policy, resolution_m)
    return simulation


if __name__ == "__main__":
    aneurysm_simulation = aneurysm_simulation_setup()
    aneurysm_simulation.run(10000, post_process_interval=100)

    # TODO:
    # look into the post_process method to see if it can be modified to save the results in a more useful format
    # this includes extraction of blood vessel features such as wall shear stress, velocity profiles, etc.
    # try to modify the speed to use a newtonian model, or better yet, a non-newtonian model
    # adjust the velocity to better reflect a fluid flow in a blood vessel (set 0.4)
    # look into the boundary conditions to see if they can be modified to better reflect the conditions in a blood vessel
    # expand to 3D
    # look into how much time/ the necessary conditions required to declare a flow model "stable". this may be velocity related
    # refine this and add better comments 