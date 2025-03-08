from aneurysm_model_2D import AneurysmSimulation2D
from xlb import ComputeBackend, PrecisionPolicy
import xlb

def aneurysm_simulation_setup(
    vessel_length_mm=10,
    vessel_diameter_mm=4,
    bulge_horizontal_mm=6,
    bulge_vertical_mm=4,
    resolution_mm=0.01,
    dynamic_viscosity=0.0035,
    blood_density=1056,
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
    
    # Calculate kinematic viscosity from dynamic viscosity and density
    kinematic_viscosity = dynamic_viscosity / blood_density  # m²/s

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
        dynamic_viscosity=0.0035,    # Blood dynamic viscosity (Pa·s)
        blood_density=1056,          # Blood density (kg/m³)
        dt=5e-7,                     # Time step
        u_max=0.04,                  # More realistic blood velocity TODO: remove in place for a better flow model
        fps=1000                     # Output frames per second
    )
    
    # Run simulation
    # TODO: add or change to run for a set duration rather than a set number of steps
    simulation.run(150000, post_process_interval=post_process_interval)

    # TODO:
    # look into the post_process method to see if it can be modified to save the results in a more useful format
    # this includes extraction of blood vessel features such as wall shear stress, velocity profiles, etc.
    # try to modify the speed to use a newtonian model, or better yet, a non-newtonian model
    # adjust the velocity to better reflect a fluid flow in a blood vessel (set 0.4)
    # look into the boundary conditions to see if they can be modified to better reflect the conditions in a blood vessel
    # expand to 3D timestep