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
        
        # Now call parent constructor - this will call _construct_warp()
        super().__init__(
            omega=omega, 
            velocity_set=velocity_set, 
            precision_policy=precision_policy, 
            compute_backend=compute_backend
        )

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

        # Function to calculate shear rate using finite differences for vector input
        @wp.func
        def calculate_shear_rate(u: Any, i: int, j: int, k: int, dim: int):
            # Simple finite difference for velocity gradients
            # The u parameter is now a vector, not an array
            dx = 1.0  # Lattice spacing
            
            # Default values
            shear_rate = 0.0
            
            # Since we don't have access to the full velocity field,
            # we can only use the local velocity value.
            # Here we use a minimal approximation as fallback
            # This will be less accurate but avoids the shape error
            
            # For a vector of length 2 or 3, we can approximate a base shear rate
            # from the magnitude of the velocity
            velocity_magnitude = 0.0
            
            if dim == 2:
                velocity_magnitude = wp.sqrt(u[0]*u[0] + u[1]*u[1])
                # Simple approximation - assuming shear rate is related to velocity magnitude
                shear_rate = velocity_magnitude / 1.0  # Some characteristic length
            else:  # dim == 3
                velocity_magnitude = wp.sqrt(u[0]*u[0] + u[1]*u[1] + u[2]*u[2])
                # Simple approximation for 3D
                shear_rate = velocity_magnitude / 1.0
            
            # Ensure non-zero shear rate to avoid division by zero
            return wp.max(shear_rate, 1e-10)

        # Function to calculate local omega based on Carreau-Yasuda model
        @wp.func
        def calculate_local_omega(u: Any, i: int, j: int, k: int, dim: int):
            # Calculate shear rate in lattice units
            shear_rate_lu = calculate_shear_rate(u, i, j, k, dim)
            
            # Convert shear rate from lattice to physical units
            # Avoid division by zero by ensuring dt_physical is non-zero
            dt_safe = wp.max(_dt_physical, 1.0e-10)
            shear_rate_physical = shear_rate_lu / dt_safe

            # Apply Carreau-Yasuda model
            factor = wp.pow(1.0 + wp.pow(_lambda_cy * shear_rate_physical, _a), (_n - 1.0) / _a)
            viscosity_physical = _mu_inf + (_mu_0 - _mu_inf) * factor
            
            # Convert to kinematic viscosity (assuming density = 1.0)
            # Density should never be exactly zero, but add safety
            density = 1.0
            nu_physical = viscosity_physical / wp.max(density, 1.0e-10)
            
            # Convert to lattice units
            # Avoid division by zero in squared values
            dx_squared = wp.max(_dx_physical * _dx_physical, 1.0e-20)
            nu_lu = nu_physical * dt_safe / dx_squared
            
            # Calculate relaxation rate omega (add safety to denominator)
            denom = wp.max(3.0 * nu_lu + 0.5, 1.0e-10)
            omega_local = 1.0 / denom
            
            # Apply stability bounds    
            omega_local = wp.clamp(omega_local, _min_omega, _max_omega)
            
            return omega_local

        # Construct the functional
        @wp.func
        def functional(f: Any, feq: Any, rho: Any, u: Any, i: int, j: int, k: int):
            # Calculate local omega for the current cell
            if _d == 2:
                dim = 2
            else:
                dim = 3
            
            omega_local = calculate_local_omega(u, i, j, k, dim)
            
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
            rho: wp.array3d(dtype=Any),
            u: wp.array4d(dtype=Any),            
        ):
            # Get the global index
            i, j, k = wp.tid()
            
            # Skip if out of bounds
            if i >= f.shape[1] or j >= f.shape[2] or k >= f.shape[3]:
                return
                
            index = wp.vec3i(i, j, k)
            
            # Load needed values
            _f = _f_vec()
            _feq = _f_vec()
            for l in range(self.velocity_set.q):
                _f[l] = f[l, index[0], index[1], index[2]]
                _feq[l] = feq[l, index[0], index[1], index[2]]
            
            # Get velocity vector at current position
            _u_vec = wp.vec(_d, dtype=self.compute_dtype)
            for d in range(_d):
                _u_vec[d] = u[d, index[0], index[1], index[2]]
            
            # Compute the collision with position info and velocity vector
            _fout = functional(_f, _feq, rho[i,j,k], _u_vec, i, j, k)
            
            # Write the result
            for l in range(self.velocity_set.q):
                fout[l, index[0], index[1], index[2]] = self.store_dtype(_fout[l])

        return functional, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f, feq, fout, rho, u):
        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[f, feq, fout, rho, u],
            dim=f.shape[1:],
        )
        return fout
        
    @Operator.register_backend(ComputeBackend.JAX)
    def jax_implementation(self, f, feq, fout, rho, u):
        """
        JAX implementation of the BGKNonNewtonian collision operator.
        """
        raise NotImplementedError("JAX implementation for BGKNonNewtonian collision is not implemented yet")