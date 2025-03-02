from xlb.operator.boundary_condition.bc_zouhe import ZouHeBC
from xlb.compute_backend import ComputeBackend
import warp as wp
import numpy as np
import jax.numpy as jnp
from typing import Any
from functools import partial
from jax import jit, lax

class TimeDependentZouHeBC(ZouHeBC):
    def __init__(self, bc_type, indices=None, **kwargs):
        # DIFFERENCE: Store original profile function to use later with timesteps
        # Original ZouHeBC processes profiles once at initialization
        self._original_profile_func = kwargs.get('profile', None)
        
        # Call parent constructor with original parameters
        super().__init__(bc_type, indices=indices, **kwargs)
        
        # DIFFERENCE: Add timestep tracking for both backends
        # Original ZouHeBC has no concept of time
        self.current_timestep = wp.zeros(1, dtype=wp.int32)
        self.jax_timestep = 0
    
    # DIFFERENCE: Override kernel construction to use timestep
    # Original ZouHeBC's functionals have timestep parameters but don't use them
    def _construct_kernel(self, functional):
        """Override with timestep-aware kernel for WARP"""
        _id = wp.uint8(self.id)
        _q = self.velocity_set.q
        
        @wp.kernel
        def kernel(
            f_pre: wp.array4d(dtype=Any),
            f_post: wp.array4d(dtype=Any),
            bc_mask: wp.array4d(dtype=wp.uint8),
            missing_mask: wp.array4d(dtype=wp.bool),
        ):
            # Get the global index
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)
            
            # read tid data
            _f_pre, _f_post, _boundary_id, _missing_mask = self._get_thread_data(
                f_pre, f_post, bc_mask, missing_mask, index
            )
            
            # DIFFERENCE: Pass timestep to functional
            # Original ZouHeBC passes None or doesn't use the parameter
            if _boundary_id == _id:
                timestep = self.current_timestep[0]  
                _f = functional(index, timestep, _missing_mask, f_pre, f_post, _f_pre, _f_post)
            else:
                _f = _f_post
                
            # Write the result
            for l in range(_q):
                f_post[l, index[0], index[1], index[2]] = self.store_dtype(_f[l])
                
        return kernel
    
    # DIFFERENCE: New method to update timestep for both backends
    # Original ZouHeBC has no time tracking
    def update_timestep(self, timestep):
        """Update the BC's timestep for both backends"""
        temp_array = wp.array([timestep], dtype=wp.int32)
        wp.copy(self.current_timestep, temp_array)
        self.jax_timestep = timestep
        
    # DIFFERENCE: Override warp_implementation is identical in structure to original
    # But relies on custom kernel with timestep awareness
    @ZouHeBC.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f_pre, f_post, bc_mask, missing_mask):
        wp.launch(
            self.warp_kernel,
            inputs=[f_pre, f_post, bc_mask, missing_mask],
            dim=f_pre.shape[1:],
        )
        return f_post
    
    # DIFFERENCE: The JAX implementation is structurally identical
    # But relies on overridden get_rho and get_vel that use timestep
    @ZouHeBC.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0))
    def jax_implementation(self, f_pre, f_post, bc_mask, missing_mask):
        # Create a mask to slice boundary cells
        boundary = bc_mask == self.id
        new_shape = (self.velocity_set.q,) + boundary.shape[1:]
        boundary = lax.broadcast_in_dim(boundary, new_shape, tuple(range(self.velocity_set.d + 1)))
        
        # DIFFERENCE: Comment indicates implicit time-dependence through calculate_equilibrium
        # Original has no concept of time-dependence
        
        # compute the equilibrium based on prescribed values and the type of BC
        feq = self.calculate_equilibrium(f_post, missing_mask)

        # set the unknown f populations based on the non-equilibrium bounce-back method
        f_post_bd = self.bounceback_nonequilibrium(f_post, feq, missing_mask)
        f_post = jnp.where(boundary, f_post_bd, f_post)
        return f_post
    
    # DIFFERENCE: Override get_rho to use time-dependent values
    # Original uses static prescribed_values
    @partial(jit, static_argnums=(0,), inline=True)
    def get_rho(self, fpop, missing_mask):
        """Override to handle time-dependent values with proper reshaping"""
        if self.bc_type == "velocity":
            # DIFFERENCE: Dynamic evaluation with current timestep
            # Original uses static self.prescribed_values
            target_shape = (self.velocity_set.d,) + fpop.shape[1:]
            velocity_fn = self._original_profile_func()
            vel = velocity_fn(self.jax_timestep)
            
            # DIFFERENCE: Uses parent's broadcast function - consistent with original
            vel = self._broadcast_prescribed_values(vel, vel.shape, target_shape)
            
            # Calculate density from velocity - same as original
            rho = self.calculate_rho(fpop, vel, missing_mask)
        elif self.bc_type == "pressure":
            # DIFFERENCE: Get density from time-dependent function
            # Original uses static self.prescribed_values
            density_fn = self._original_profile_func()
            rho = density_fn(self.jax_timestep)
        else:
            raise ValueError(f"type = {self.bc_type} not supported!")
        return rho

    # DIFFERENCE: Override get_vel to use time-dependent values
    # Original uses static prescribed_values
    @partial(jit, static_argnums=(0,), inline=True)
    def get_vel(self, fpop, missing_mask):
        """Override to handle time-dependent values with proper reshaping"""
        if self.bc_type == "velocity":
            # DIFFERENCE: Dynamic evaluation with current timestep
            # Original uses static self.prescribed_values
            target_shape = (self.velocity_set.d,) + fpop.shape[1:]
            velocity_fn = self._original_profile_func()
            vel = velocity_fn(self.jax_timestep)
            
            # DIFFERENCE: Uses parent's broadcast function - consistent with original
            vel = self._broadcast_prescribed_values(vel, vel.shape, target_shape)
        elif self.bc_type == "pressure":
            # DIFFERENCE: Get density from time-dependent function
            # Original uses static self.prescribed_values
            density_fn = self._original_profile_func()
            rho = density_fn(self.jax_timestep)
            
            # Calculate velocity from density - same as original
            vel = self.calculate_vel(fpop, rho, missing_mask)
        else:
            raise ValueError(f"type = {self.bc_type} not supported!")
        return vel