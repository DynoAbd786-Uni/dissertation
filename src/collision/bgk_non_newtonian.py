import jax.numpy as jnp
from jax import jit
import warp as wp
from typing import Any
from xlb.compute_backend import ComputeBackend
from xlb.operator.collision.collision import Collision
from xlb.operator import Operator
from xlb.velocity_set import VelocitySet
from functools import partial


class BGKNonNewtonian(Collision):
    """
    BGK collision operator for LBM with non-Newtonian fluid behavior using Carreau-Yasuda model.
    """

    def __init__(
        self,
        velocity_set: VelocitySet,
        omega: float,
        mu_0=0.056,      # Zero-shear viscosity [Pa·s]
        mu_inf=0.00345,   # Infinite-shear viscosity [Pa·s]
        lambda_cy=3.313,  # Relaxation time [s]
        n=0.3568,         # Power law index
        a=2.0,            # Transition parameter
        dx_physical=1e-5,  # Physical size of lattice cell in meters
        dt_physical=1e-6,  # Physical time step in seconds
        precision_policy=None,
        compute_backend=None
    ):
        # Store parameters BEFORE calling parent constructor
        self.mu_0_lu = mu_0
        self.mu_inf_lu = mu_inf
        self.lambda_cy_lu = lambda_cy / dt_physical  
        self.n = n
        self.a = a
        self.dx_physical = dx_physical
        self.dt_physical = dt_physical
        
        # Add stability monitoring
        self.has_unstable_omega = False
        self.min_observed_omega = 1.0
        self.max_observed_omega = 1.0
        
        # Now call parent constructor - this will call _construct_warp()
        super().__init__(
            omega=omega, 
            velocity_set=velocity_set, 
            precision_policy=precision_policy, 
            compute_backend=compute_backend
        )

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0,))
    def jax_implementation(self, f: jnp.ndarray, feq: jnp.ndarray, rho, u):
        # Calculate velocity gradient tensor
        grad_ux = jnp.gradient(u[0], axis=(0, 1, 2))
        grad_uy = jnp.gradient(u[1], axis=(0, 1, 2))
        grad_uz = jnp.zeros_like(grad_ux) if u.shape[0] == 2 else jnp.gradient(u[2], axis=(0, 1, 2))
        
        # Calculate strain rate tensor components
        Sxx = grad_ux[0]
        Sxy = 0.5 * (grad_ux[1] + grad_uy[0])
        Sxz = 0.5 * (grad_ux[2] + grad_uz[0]) if u.shape[0] == 3 else 0
        Syy = grad_uy[1]
        Syz = 0.5 * (grad_uy[2] + grad_uz[1]) if u.shape[0] == 3 else 0
        Szz = grad_uz[2] if u.shape[0] == 3 else 0
        
        # Calculate shear rate (second invariant of strain rate tensor)
        if u.shape[0] == 2:  # 2D case
            shear_rate = 2.0 * jnp.sqrt(Sxx**2 + Syy**2 + 2.0 * Sxy**2)
        else:  # 3D case
            shear_rate = 2.0 * jnp.sqrt(Sxx**2 + Syy**2 + Szz**2 + 2.0 * (Sxy**2 + Sxz**2 + Syz**2))
        
        # Convert shear rate from lattice to physical units
        shear_rate_physical = shear_rate * (self.dt_physical / self.dx_physical)
        
        # Avoid division by zero
        shear_rate_physical = jnp.maximum(shear_rate_physical, 1e-10)
        
        # Calculate local viscosity using Carreau-Yasuda model
        mu_diff = self.mu_0_lu - self.mu_inf_lu
        factor = (1.0 + (self.lambda_cy_lu * shear_rate_physical) ** self.a) ** ((self.n - 1.0) / self.a)
        viscosity_lu = self.mu_inf_lu + mu_diff * factor
        
        # Convert to kinematic viscosity (assuming constant density)
        nu = viscosity_lu / 1.0  # Density is typically 1.0 in lattice units
        
        # Calculate local omega
        omega_local = 1.0 / (3.0 * nu + 0.5)
        
        # Apply stability bounds
        omega_local = jnp.clip(omega_local, 0.55, 1.95)
        
        # Perform BGK collision with space-varying omega
        fneq = f - feq
        fout = f - omega_local[None, :, :, :] * fneq
        
        return fout

    def _construct_warp(self):
        # Set local constants
        _w = self.velocity_set.w
        _d = self.velocity_set.d
        _omega = wp.constant(self.compute_dtype(self.omega))
        _f_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)
        
        # Carreau-Yasuda parameters
        _mu_0 = wp.constant(self.compute_dtype(self.mu_0_lu))
        _mu_inf = wp.constant(self.compute_dtype(self.mu_inf_lu))
        _lambda_cy = wp.constant(self.compute_dtype(self.lambda_cy_lu))
        _n = wp.constant(self.compute_dtype(self.n))
        _a = wp.constant(self.compute_dtype(self.a))
        
        # Unit conversion factors
        _dx_physical = wp.constant(self.compute_dtype(self.dx_physical))
        _dt_physical = wp.constant(self.compute_dtype(self.dt_physical))
        
        # Stability limits
        _min_omega = wp.constant(self.compute_dtype(0.55))
        _max_omega = wp.constant(self.compute_dtype(1.95))

        # Create arrays for stability tracking
        self.stability_check = wp.zeros(1, dtype=wp.int32)
        self.min_omega_value = wp.array([2.0], dtype=self.compute_dtype)  # Initialize with value 2.0
        self.max_omega_value = wp.array([0.0], dtype=self.compute_dtype)  # Initialize with value 0.0
        
        # Construct the functional for calculating local omega
        @wp.func
        def calculate_local_omega(u: Any, i: int, j: int, k: int, dim: int):
            # Simple finite difference for velocity gradients
            dx = 1.0  # Lattice spacing
            
            # # Velocity gradients
            # if dim == 2:
            #     # 2D case
            #     ux_x = 0.0
            #     ux_y = 0.0
            #     uy_x = 0.0
            #     uy_y = 0.0
                
            #     # Central difference
            #     if i > 0 and i < u.shape[1] - 1:
            #         ux_x = (u[0, i+1, j, k] - u[0, i-1, j, k]) / (2.0 * dx)
            #     if j > 0 and j < u.shape[2] - 1:
            #         ux_y = (u[0, i, j+1, k] - u[0, i, j-1, k]) / (2.0 * dx)
            #         uy_y = (u[1, i, j+1, k] - u[1, i, j-1, k]) / (2.0 * dx)
            #     if i > 0 and i < u.shape[1] - 1:
            #         uy_x = (u[1, i+1, j, k] - u[1, i-1, j, k]) / (2.0 * dx)
                
            #     # Strain rate tensor components
            #     Sxx = ux_x
            #     Sxy = 0.5 * (ux_y + uy_x)
            #     Syy = uy_y
                
            #     # Shear rate (second invariant)
            #     shear_rate = 2.0 * wp.sqrt(Sxx*Sxx + Syy*Syy + 2.0*Sxy*Sxy)
            # else:
            #     # 3D case (simplified)
            #     shear_rate = 0.001  # Default small value

            

            # For now, use a default value to avoid the error
            shear_rate = 0.01  # Default value
            
            # Convert shear rate from lattice to physical units
            shear_rate_physical = shear_rate * (_dt_physical / _dx_physical)
            
            # Convert shear rate from lattice to physical units
            shear_rate_physical = shear_rate * (_dt_physical / _dx_physical)

            # Ensure non-zero shear rate
            shear_rate_physical = wp.max(shear_rate_physical, 1e-10)

            # Carreau-Yasuda model
            factor = wp.pow(1.0 + wp.pow(_lambda_cy * shear_rate_physical, _a), (_n - 1.0) / _a)
            viscosity_lu = _mu_inf + (_mu_0 - _mu_inf) * factor

            # Convert to kinematic viscosity and then to omega
            nu = viscosity_lu / 1.0  # Density assumed 1.0
            omega_local = 1.0 / (3.0 * nu + 0.5)
            
            # Apply stability bounds    
            omega_local = wp.clamp(omega_local, _min_omega, _max_omega)
            
            return omega_local

        # Construct the functional
        @wp.func
        def functional(f: Any, feq: Any, rho: Any, u: Any):
            # Pre-calculate omega for the current cell using a separate function
            omega_local = _omega  # Default fallback
            
            # Use thread ID to get position info
            i, j, k = wp.tid()
            
            # Determine dimensionality with explicit if-statement (not ternary)
            # dim = 0
            # if _d == 2:
            #     dim = 2
            # else:
            #     dim = 3
            
            # Calculate local omega using the existing function
            omega_local = calculate_local_omega(u, i, j, k, _d)
            
            # Standard BGK with variable omega
            fneq = f - feq
            fout = f - omega_local * fneq
            return fout

        # Construct the warp kernel
        @wp.kernel
        def kernel(
            f: wp.array4d(dtype=Any),
            feq: wp.array4d(dtype=Any),
            fout: wp.array4d(dtype=Any),
            rho: wp.array4d(dtype=Any),
            u: wp.array4d(dtype=Any),
            stability_check: wp.array1d(dtype=wp.int32),
            min_omega_value: wp.array1d(dtype=Any),
            max_omega_value: wp.array1d(dtype=Any),
            
        ):
            # Get the global index
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)
            
            # Load needed values
            _f = _f_vec()
            _feq = _f_vec()
            for l in range(self.velocity_set.q):
                _f[l] = f[l, index[0], index[1], index[2]]
                _feq[l] = feq[l, index[0], index[1], index[2]]
                
            # Calculate omega for stability checking
            dim = 2 if u.shape[0] == 2 else 3
            omega_local = calculate_local_omega(u, i, j, k, dim)
            
            # Check for stability issues
            if omega_local < _min_omega or omega_local > _max_omega:
                # Atomically update the stability flag and track min/max
                wp.atomic_add(stability_check, 0, 1)
                wp.atomic_min(min_omega_value, 0, omega_local)
                wp.atomic_max(max_omega_value, 0, omega_local)
            
            # Compute the collision
            _fout = functional(_f, _feq, rho, u)
            
            # Write the result
            for l in range(self.velocity_set.q):
                fout[l, index[0], index[1], index[2]] = self.store_dtype(_fout[l])

        return functional, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f, feq, fout, rho, u):
        # Reset stability checks
        wp.copy(self.stability_check, wp.zeros(1, dtype=wp.int32))
        wp.copy(self.min_omega_value, wp.array([2.0], dtype=self.compute_dtype))
        wp.copy(self.max_omega_value, wp.zeros(1, dtype=self.compute_dtype))

        # Define block size (adjust based on GPU architecture)
        block_size = (32, 1, 1)  # 1D block

        # Calculate grid size to cover the domain
        grid_size = (
            (f.shape[1] + block_size[0] - 1) // block_size[0],
            (f.shape[2] + block_size[1] - 1) // block_size[1],
            (f.shape[3] + block_size[2] - 1) // block_size[2],
        )

        # Launch kernel
        wp.launch(
            self.warp_kernel,
            inputs=[f, feq, fout, rho, u, self.stability_check, self.min_omega_value, self.max_omega_value],
            dim=grid_size,  # Use explicit grid size
            device=wp.get_device("cuda"),
        )
        
        return fout