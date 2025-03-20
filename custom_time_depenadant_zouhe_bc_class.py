from xlb.operator.boundary_condition.bc_zouhe import ZouHeBC
from xlb.operator import Operator
from xlb.compute_backend import ComputeBackend
import warp as wp
import numpy as np
import jax.numpy as jnp
from typing import Any
from functools import partial
from jax import jit, lax

class TimeDependentZouHeBC(ZouHeBC):
    def __init__(self, bc_type, indices=None, **kwargs):

        # print(f"Creating TimeDependentZouHeBC with backend: {self.compute_backend}")
        # if self.compute_backend == ComputeBackend.WARP:
        #     print("WARP backend confirmed for this boundary condition")


        # DIFFERENCE: Store original profile function to use later with timesteps
        # Original ZouHeBC processes profiles once at initialization
        self._original_profile_func = kwargs.get('profile', None)
        
        # Call parent constructor with original parameters
        super().__init__(bc_type, indices=indices, **kwargs)
        
        # DIFFERENCE: Add timestep tracking for both backends
        # Original ZouHeBC has no concept of time
        self.current_timestep = wp.zeros(1, dtype=wp.int32)
        self.jax_timestep = 0
        self.current_step = 0  # Use a Python variable to track timestep

    def __call__(self, f_pre, f_post, bc_mask, missing_mask):
        """Override to provide time-dependent behavior"""
        # Get timestep from stepper context using a wrapper approach
        if not hasattr(self, '_last_call_time'):
            self._last_call_time = 0
            print("First BC call")
        else:
            # Increment timestep on each call
            self.current_step += 1
            if self.current_step % 1000 == 0:
                print(f"BC called at step {self.current_step}")
            
            # Update the WARP array with current step
            wp.copy(self.current_timestep, wp.array([self.current_step], dtype=wp.int32))
        
        # Call normal implementation
        if self.compute_backend == ComputeBackend.WARP:
            return self.warp_implementation(f_pre, f_post, bc_mask, missing_mask)
        elif self.compute_backend == ComputeBackend.JAX:
            return self.jax_implementation(f_pre, f_post, bc_mask, missing_mask)
    
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
            timestep_array: wp.array(dtype=wp.int32),  # Added timestep array as kernel input
        ):
            # Get the global index
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)
            
            # read tid data
            _f_pre, _f_post, _boundary_id, _missing_mask = self._get_thread_data(
                f_pre, f_post, bc_mask, missing_mask, index
            )
            
            # DIFFERENCE: Pass timestep to functional
            if _boundary_id == _id:
                timestep = timestep_array[0]  # Get current timestep from input array
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
        # Update WARP array
        temp_array = wp.array([timestep], dtype=wp.int32)
        wp.copy(self.current_timestep, temp_array)
        self.jax_timestep = timestep
        
        # Debug output
        if timestep % 1000 == 0:
            # Calculate expected velocity to verify
            dt = 0.00005  # Use your actual dt
            omega = 2.0 * np.pi * 20.0
            t = dt * timestep
            expected_vel = 0.04 * (0.5 + 0.5 * np.sin(omega * t))
           # print(f"Updating BC timestep to {timestep} (t={t:.6f}s)")
            # print(f"Expected velocity: {expected_vel:.6f}")
            
            # Verify the WARP array was updated
            # print(f"Timestep array contains: {self.current_timestep}")
        
    # DIFFERENCE: Override warp_implementation is identical in structure to original
    # But relies on custom kernel with timestep awareness
    # DIFFERENCE: Override warp_implementation to pass the timestep array
    @ZouHeBC.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f_pre, f_post, bc_mask, missing_mask, timestep=None):
        """Override to capture timestep from stepper before kernel launch"""
        # Update timestep from stepper  
        if timestep is not None:
            wp.copy(self.current_timestep, wp.array([timestep], dtype=wp.int32))
            if timestep % 1000 == 0:
                # More robust debug output
                print(f"\n=== WARP KERNEL LAUNCHING ===")
                print(f"BC ID: {self.id}, Type: {self.bc_type}")
                print(f"Timestep: {timestep}")
                print(f"current_timestep array: {wp.to_numpy(self.current_timestep)[0]}")
                print(f"Launching kernel with shape: {f_pre.shape[1:]}")
                print("============================\n")
    
        # Launch with all required inputs
        wp.launch(
            self.warp_kernel,
            inputs=[f_pre, f_post, bc_mask, missing_mask, self.current_timestep],
            dim=f_pre.shape[1:],
        )
        return f_post
    
    def _construct_warp(self):
        """Override with timestep-aware functional implementation"""
        # Set local constants from parent class
        _d = self.velocity_set.d
        _q = self.velocity_set.q
        _u_vec = wp.vec(self.velocity_set.d, dtype=self.compute_dtype)
        _opp_indices = self.velocity_set.opp_indices
        _c = self.velocity_set.c
        _c_float = self.velocity_set.c_float

        print("testing for call to _construct_warp")

        # Reuse helper functions from parent class
        @wp.func
        def _get_fsum(fpop: Any, missing_mask: Any):
            # Same as parent class implementation
            fsum_known = self.compute_dtype(0.0)
            fsum_middle = self.compute_dtype(0.0)
            for l in range(_q):
                if missing_mask[_opp_indices[l]] == wp.uint8(1):
                    fsum_known += self.compute_dtype(2.0) * fpop[l]
                elif missing_mask[l] != wp.uint8(1):
                    fsum_middle += fpop[l]
            return fsum_known + fsum_middle

        @wp.func
        def get_normal_vectors(missing_mask: Any):
            # Same as parent class implementation
            if wp.static(_d == 3):
                for l in range(_q):
                    if missing_mask[l] == wp.uint8(1) and wp.abs(_c[0, l]) + wp.abs(_c[1, l]) + wp.abs(_c[2, l]) == 1:
                        return -_u_vec(_c_float[0, l], _c_float[1, l], _c_float[2, l])
            else:
                for l in range(_q):
                    if missing_mask[l] == wp.uint8(1) and wp.abs(_c[0, l]) + wp.abs(_c[1, l]) == 1:
                        return -_u_vec(_c_float[0, l], _c_float[1, l])

        @wp.func
        def bounceback_nonequilibrium(fpop: Any, feq: Any, missing_mask: Any):
            # Same as parent class implementation
            for l in range(_q):
                if missing_mask[l] == wp.uint8(1):
                    fpop[l] = fpop[_opp_indices[l]] + feq[l] - feq[_opp_indices[l]]
            return fpop

        @wp.func
        def functional_velocity_time_dependent(
            index: Any,
            timestep: Any,  # This comes from the kernel
            _missing_mask: Any,
            f_pre: Any,
            f_post: Any,
            _f_pre: Any,
            _f_post: Any,
        ):
            # Post-streaming values are only modified at missing direction
            _f = _f_post

            # Find normal vector
            normals = get_normal_vectors(_missing_mask)
            
            # Use hardcoded timestep calculation 
            t = wp.float32(timestep) * 0.00005  # dt value
            omega = wp.float32(2.0 * 3.14159 * 20.0)  # Angular frequency
            
            # Calculate velocity with sinusoidal variation
            prescribed_velocity = wp.float32(0.04) * (wp.float32(0.5) + wp.float32(0.5) * wp.sin(omega * t))
            
            # Create velocity vector aligned with normal direction
            _u = prescribed_velocity * normals
            
            # Calculate density based on velocity
            fsum = _get_fsum(_f, _missing_mask)
            unormal = self.compute_dtype(0.0)
            for d in range(_d):
                unormal += _u[d] * normals[d]
            _rho = fsum / (self.compute_dtype(1.0) + unormal)

            # impose non-equilibrium bounceback
            _feq = self.equilibrium_operator.warp_functional(_rho, _u)
            _f = bounceback_nonequilibrium(_f, _feq, _missing_mask)
            return _f

        # Use our time-dependent functional
        if self.bc_type == "velocity":
            functional = functional_velocity_time_dependent
        elif self.bc_type == "pressure":
            # If needed, implement pressure version too
            raise NotImplementedError("Time-dependent pressure BC not implemented yet")

        # Use parent's kernel construction with our functional
        kernel = self._construct_kernel(functional)

        return functional, kernel
    

    # OVERRIDE BoundaryCondition methods to allow for passsable timestep
    def _construct_aux_data_init_kernel(self, functional):
        """
        Constructs the warp kernel for the auxilary data recovery.
        """
        _id = wp.uint8(self.id)
        _opp_indices = self.velocity_set.opp_indices
        _num_of_aux_data = self.num_of_aux_data

        # Construct the warp kernel
        @wp.kernel
        def aux_data_init_kernel(
            f_0: wp.array4d(dtype=Any),
            f_1: wp.array4d(dtype=Any),
            bc_mask: wp.array4d(dtype=wp.uint8),
            missing_mask: wp.array4d(dtype=wp.bool),
            time_array: wp.array(dtype=wp.int32),  # Added timestep array as kernel input
        ):
            # Get the global index
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)

            # read tid data
            _f_0, _f_1, _boundary_id, _missing_mask = self._get_thread_data(f_0, f_1, bc_mask, missing_mask, index)

            # Apply the functional
            if _boundary_id == _id:
                # prescribed_values is a q-sized vector of type wp.vec
                prescribed_values = functional(index, time_array[0]) # Pass timestep to functional
                # Write the result for all q directions, but only store up to num_of_aux_data
                # TODO: Somehow raise an error if the number of prescribed values does not match the number of missing directions
                counter = wp.int32(0)
                for l in range(self.velocity_set.q):
                    if _missing_mask[l] == wp.uint8(1) and counter < _num_of_aux_data:
                        f_1[_opp_indices[l], index[0], index[1], index[2]] = self.store_dtype(prescribed_values[counter])
                        counter += 1

        return aux_data_init_kernel
    
    def aux_data_init(self, f_0, f_1, bc_mask, missing_mask):
        """Override to initialize auxiliary data with timestep awareness"""
        if self.compute_backend == ComputeBackend.WARP:
            # Use our custom kernel that accepts timestep
            kernel = self._construct_aux_data_init_kernel(self._original_profile_func)
            
            # Launch with current timestep array
            wp.launch(
                kernel,
                inputs=[f_0, f_1, bc_mask, missing_mask, self.current_timestep],  # Pass timestep array
                dim=f_0.shape[1:],
            )
        elif self.compute_backend == ComputeBackend.JAX:
            # For JAX implementation
            # Similar approach - use current jax_timestep
            self.prescribed_values = self._original_profile_func()(self.jax_timestep)
            
        # Mark as initialized
        self.is_initialized_with_aux_data = True
        return f_0, f_1
    
    
    
    # DIFFERENCE: The JAX implementation is structurally identical
    # But relies on overridden get_rho and get_vel that use timestep
    @Operator.register_backend(ComputeBackend.JAX)
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