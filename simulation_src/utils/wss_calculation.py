import numpy as np
import jax.numpy as jnp

def calculate_wss(velocity_field, wall_indices, dx_physical, dt_physical, 
                  mu_0=0.056, mu_inf=0.00345, lambda_cy=3.313, n=0.3568, a=2.0):
    """
    Calculate Wall Shear Stress (WSS) at the boundaries using the Carreau-Yasuda viscosity model.
    
    This function:
    1. Uses provided wall indices to identify boundary locations
    2. Computes velocity gradients near the walls using finite differences
    3. Calculates shear rates at wall boundaries
    4. Applies the Carreau-Yasuda model to obtain variable viscosity values
    5. Computes WSS as the product of viscosity and shear rate
    
    Args:
        velocity_field: Velocity field (array of shape [components, x, y]) from macroscopic calculations
        wall_indices: List of wall indices [x_indices, y_indices]
        dx_physical: Physical size of lattice cell in physical units (e.g., meters)
        dt_physical: Physical time step in physical units (e.g., seconds)
        mu_0: Zero-shear viscosity [Pa·s], default 0.056
        mu_inf: Infinite-shear viscosity [Pa·s], default 0.00345
        lambda_cy: Relaxation time [s], default 3.313
        n: Power law index, default 0.3568
        a: Transition parameter, default 2.0
            
    Returns:
        tuple: (wss_magnitude, wss_x, wss_y, wall_mask) where:
            - wss_magnitude is a 2D array of WSS magnitude values
            - wss_x is a 2D array of x-component of WSS vector
            - wss_y is a 2D array of y-component of WSS vector
            - wall_mask is a boolean array indicating wall locations
    """
    # Extract velocity components and convert to numpy if needed
    if isinstance(velocity_field, jnp.ndarray):
        u_x = np.array(velocity_field[0])
        u_y = np.array(velocity_field[1])
    else:
        u_x = velocity_field[0]
        u_y = velocity_field[1]
    
    # Get grid dimensions
    shape = u_x.shape
    width, height = shape
    
    # Create arrays for results
    wss_magnitude = np.zeros(shape)
    wss_x = np.zeros(shape)
    wss_y = np.zeros(shape)
    wall_mask = np.zeros(shape, dtype=bool)
    
    # Mark wall locations in the wall mask using wall indices
    wall_x = wall_indices[0]
    wall_y = wall_indices[1]
    
    for i in range(len(wall_x)):
        x, y = wall_x[i], wall_y[i]
        if 0 <= x < width and 0 <= y < height:
            wall_mask[x, y] = True
    
    # Define directions for neighbor checking (four adjacent directions)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Process each potential wall node
    for x in range(1, width-1):
        for y in range(1, height-1):
            if wall_mask[x, y]:
                # For each wall node, find the velocity gradient from adjacent fluid nodes
                du_dx = 0.0  # x-derivative of u_x
                du_dy = 0.0  # y-derivative of u_x
                dv_dx = 0.0  # x-derivative of u_y
                dv_dy = 0.0  # y-derivative of u_y
                valid_neighbors = 0
                
                # Calculate normal vector to wall (approximate)
                wall_normal_x = 0.0
                wall_normal_y = 0.0
                
                # Check each neighboring direction
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    # Check if neighbor is valid fluid node
                    if 0 <= nx < width and 0 <= ny < height and not wall_mask[nx, ny]:
                        # Use central difference gradient approximation
                        distance = np.sqrt(dx**2 + dy**2) * dx_physical
                        
                        # Accumulate normal vector components (pointing from wall to fluid)
                        wall_normal_x += dx
                        wall_normal_y += dy
                        
                        # Accumulate gradients (du_i/dx_j)
                        du_dx += u_x[nx, ny] * dx / distance
                        du_dy += u_x[nx, ny] * dy / distance
                        dv_dx += u_y[nx, ny] * dx / distance
                        dv_dy += u_y[nx, ny] * dy / distance
                        
                        valid_neighbors += 1
                
                if valid_neighbors > 0:
                    # Normalize gradients by number of valid neighbors
                    du_dx /= valid_neighbors
                    du_dy /= valid_neighbors
                    dv_dx /= valid_neighbors
                    dv_dy /= valid_neighbors
                    
                    # Normalize wall normal vector
                    norm = np.sqrt(wall_normal_x**2 + wall_normal_y**2)
                    if norm > 1e-10:
                        wall_normal_x /= norm
                        wall_normal_y /= norm
                    
                    # Convert velocity gradients to physical units
                    conversion = dx_physical / dt_physical
                    du_dx *= conversion
                    du_dy *= conversion
                    dv_dx *= conversion
                    dv_dy *= conversion
                    
                    # Calculate strain rate tensor components
                    e_xx = du_dx
                    e_xy = 0.5 * (du_dy + dv_dx)
                    e_yy = dv_dy
                    
                    # Calculate shear rate (magnitude of strain rate tensor)
                    # γ̇ = sqrt(2 * e_ij * e_ij)
                    shear_rate = np.sqrt(2.0 * (e_xx**2 + 2.0*e_xy**2 + e_yy**2))
                    
                    # Apply Carreau-Yasuda model to get variable viscosity
                    # μ = μ∞ + (μ0 - μ∞) * [1 + (λγ̇)^a]^((n-1)/a)
                    factor = (1.0 + (lambda_cy * shear_rate)**a)**((n - 1.0) / a)
                    viscosity = mu_inf + (mu_0 - mu_inf) * factor
                    
                    # Calculate WSS magnitude (τ = μ * γ̇)
                    wss_magnitude[x, y] = viscosity * shear_rate
                    
                    # Calculate WSS vector components
                    # Wall shear stress acts in the direction tangential to the wall
                    # So we need to calculate tangential components
                    tangential_x = -wall_normal_y  # perpendicular to normal
                    tangential_y = wall_normal_x   # perpendicular to normal
                    
                    # Compute strain rate in tangential direction
                    strain_tangential = (
                        e_xx * tangential_x * tangential_x +
                        e_yy * tangential_y * tangential_y +
                        2 * e_xy * tangential_x * tangential_y
                    )
                    
                    # Calculate directional WSS (sign indicates direction along tangent)
                    # This ensures the WSS vector points in the correct direction along the wall
                    wss_dir = np.sign(strain_tangential)
                    
                    # Assign WSS vector components (magnitude * direction)
                    wss_x[x, y] = wss_magnitude[x, y] * wss_dir * tangential_x
                    wss_y[x, y] = wss_magnitude[x, y] * wss_dir * tangential_y
    
    return wss_magnitude, wss_x, wss_y, wall_mask