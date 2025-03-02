def convert_velocity(physical_velocity_ms, dx_m, dt_s):
    """Convert physical velocity to lattice velocity units
    
    Args:
        physical_velocity_ms (float): Velocity in m/s
        dx_m (float): Lattice spacing in meters
        dt_s (float): Time step in seconds
        
    Returns:
        float: Velocity in lattice units
    """
    return physical_velocity_ms * (dt_s / dx_m)

def convert_lattice_to_physical(lattice_velocity, dx_m, dt_s):
    """Convert lattice velocity to physical units
    
    Args:
        lattice_velocity (float): Velocity in lattice units
        dx_m (float): Lattice spacing in meters
        dt_s (float): Time step in seconds
    
    Returns:
        float: Velocity in m/s
    """
    return lattice_velocity * (dx_m / dt_s)