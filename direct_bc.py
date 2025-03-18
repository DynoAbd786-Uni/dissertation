import warp as wp
import numpy as np
import jax.numpy as jnp
from xlb.operator.boundary_condition.boundary_condition import BoundaryCondition, ImplementationStep
from xlb.operator import Operator
from xlb.compute_backend import ComputeBackend
from typing import Any
from functools import partial
from jax import jit, lax
from xlb.operator.equilibrium import QuadraticEquilibrium


class DirectTimeDependentBC(BoundaryCondition):
    """
    A time-dependent velocity boundary condition that follows ZouHeBC's structure
    but adds time dependence to the velocity.
    """
    
    def __init__(self, dt, dx, u_max, frequency, flow_profile=None,
                 bc_type="velocity", indices=None,
                 **kwargs):  # Keep kwargs for parent class
        """Initialize the DirectTimeDependentBC class"""
        # Verify bc_type is supported
        assert bc_type == "velocity", "DirectTimeDependentBC only supports 'velocity' type boundary conditions"
        self.bc_type = bc_type
        
        # Store time parameters
        self.dt = dt
        self.dx = dx
        self.u_max_physical = u_max
        self.frequency = frequency
        self.u_max = self.u_max_physical * (self.dt/self.dx)
        
        # Initialize variables for flow profile data
        self.use_csv_profile = False
        self.profile_times = None
        self.profile_velocities = None
        self.profile_period = None
        self.times_device = None
        self.velocities_device = None
        self.num_points = 0
        self.period = 0.0
        
        # Add equilibrium operator like ZouHeBC
        self.equilibrium_operator = QuadraticEquilibrium()
        
        # Process flow profile data
        self.flow_profile = flow_profile
        if flow_profile is not None and isinstance(flow_profile, dict) and 'name' in flow_profile:
            self.profile_name = flow_profile['name']
            
            # Check if we have actual data or just using default sinusoidal
            if flow_profile.get('data') is not None and 'x' in flow_profile['data'] and 'y' in flow_profile['data']:
                self.flow_profile_data = flow_profile['data']
                self.profile_times = self.flow_profile_data['x']
                self.profile_velocities = self.flow_profile_data['y']
                
                # Set flag to use CSV profile
                self.use_csv_profile = True
                
                # Calculate period (duration) of the flow profile
                self.profile_period = float(self.profile_times[-1] - self.profile_times[0])
                
                # Find max velocity in profile for normalization
                self.profile_max_vel = np.max(np.abs(self.profile_velocities))
            else:
                self.flow_profile_data = None
        else:
            self.profile_name = "Sinusoidal_1Hz"
            self.flow_profile_data = None
        
        # --- NOW call super().__init__ after all variables are defined ---
        super().__init__(
            ImplementationStep.STREAMING,  # Same as ZouHeBC
            indices=indices,
            **kwargs
        )

        # --- After super().__init__ ---
        # Set needs_aux flags like ZouHeBC
        self.needs_aux_init = True
        self.needs_aux_recovery = True
        self.num_of_aux_data = 1  # One aux data for velocity
        self.needs_padding = True
        
        # Upload flow profile data to device if using CSV (AFTER super().__init__)
        if self.use_csv_profile:
            print(f"Using CSV flow profile: {self.profile_name}")
            print(f"  Profile duration: {self.profile_period:.4f} seconds")
            print(f"  Profile max velocity: {self.profile_max_vel:.4f}")
            print(f"  Profile points: {len(self.profile_times)}")
            
            # Convert to float32 arrays for warp
            times = self.profile_times.astype(np.float32)
            velocities = self.profile_velocities.astype(np.float32)
            
            # Create device arrays
            self.times_device = wp.array(times, dtype=wp.float32)
            self.velocities_device = wp.array(velocities, dtype=wp.float32)
            self.num_points = wp.int32(len(times))
            self.period = wp.float32(self.profile_period)
        else:
            print(f"Using sinusoidal flow profile: {self.profile_name}")
        
    # Modify _construct_warp to accept a pre-calculated velocity
    def _construct_warp(self):

        from constants import Y_VALUES_1, Y_VALUES_2, Y_VALUES_3, Y_VALUES_4
        
        """Construct the WARP kernel and functional"""
        _d = self.velocity_set.d
        _q = self.velocity_set.q
        _u_vec = wp.vec(_d, dtype=self.compute_dtype)
        _opp_indices = self.velocity_set.opp_indices
        _c = self.velocity_set.c
        _c_float = self.velocity_set.c_float
        
        # Store parameters as static constants for the kernel
        _dt = wp.static(self.compute_dtype(self.dt))
        _u_max = wp.static(self.compute_dtype(self.u_max))
        _omega = wp.static(self.compute_dtype(2.0 * np.pi * self.frequency))

        # Using formulas or extrapolation
        _use_csv_profile = wp.static(wp.int32(int(self.use_csv_profile)))
        _y1 = wp.static(Y_VALUES_1)
        _y2 = wp.static(Y_VALUES_2)
        _y3 = wp.static(Y_VALUES_3)
        _y4 = wp.static(Y_VALUES_4)

        print(_y1)
        
        # Helper functions copied from ZouHe
        @wp.func
        def _get_fsum(fpop: Any, missing_mask: Any):
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
            """Get normal vectors at boundary"""
            if wp.static(_d == 3):
                for l in range(_q):
                    if missing_mask[l] == wp.uint8(1) and wp.abs(_c[0, l]) + wp.abs(_c[1, l]) + wp.abs(_c[2, l]) == 1:
                        return -_u_vec(_c_float[0, l], _c_float[1, l], _c_float[2, l])
            else:
                for l in range(_q):
                    if missing_mask[l] == wp.uint8(1) and wp.abs(_c[0, l]) + wp.abs(_c[1, l]) == 1:
                        return -_u_vec(_c_float[0, l], _c_float[1, l])
            
            # Return default if no match found (shouldn't happen)
            if wp.static(_d == 2):
                return _u_vec(0.0, 0.0)
            else:
                return _u_vec(0.0, 0.0, 0.0)
            
        @wp.func
        def bounceback_nonequilibrium(fpop: Any, feq: Any, missing_mask: Any):
            for l in range(_q):
                if missing_mask[l] == wp.uint8(1):
                    fpop[l] = fpop[_opp_indices[l]] + feq[l] - feq[_opp_indices[l]]
            return fpop
            
        @wp.func
        def fmod(a: wp.float32, b: wp.float32) -> wp.float32:
            return a - b * wp.floor(a / b)
        
        @wp.func
        def interpolate_flow_profile(normalized_time: wp.float32) -> wp.float32:
            """
            Interpolate flow profile from 4 4x4 matrices of preloaded values
            
            Args:
                normalized_time: Time value between 0-1 representing position in the 1-second cycle
                
            Returns:
                Interpolated velocity value
            """
            # Step 1: Scale to 0-63 range (64 points)
            # Input time is 0-1, so multiply by 63 to get the float index
            float_index = normalized_time * 63.0
            
            # Step 2: Find the indices to interpolate between
            lower_idx = wp.min(wp.int32(float_index), 62)  # Ensure we don't go out of bounds
            upper_idx = wp.min(lower_idx + 1, 63)
            
            # Step 3: Calculate interpolation factor (0-1) between the two points
            factor = float_index - wp.float32(lower_idx)
            
            # Step 4: Get the matrix and location for each index
            # Matrix 1: indices 0-15
            # Matrix 2: indices 16-31
            # Matrix 3: indices 32-47
            # Matrix 4: indices 48-63
            
            # For lower index
            lower_value = wp.float32(0.0)
            if lower_idx < 16:
                # Use Y_VALUES_1
                row = lower_idx // 4
                col = lower_idx % 4
                lower_value = Y_VALUES_1[row, col]
            elif lower_idx < 32:
                # Use Y_VALUES_2
                row = (lower_idx - 16) // 4
                col = (lower_idx - 16) % 4
                lower_value = Y_VALUES_2[row, col]
            elif lower_idx < 48:
                # Use Y_VALUES_3
                row = (lower_idx - 32) // 4
                col = (lower_idx - 32) % 4
                lower_value = Y_VALUES_3[row, col]
            else:
                # Use Y_VALUES_4
                row = (lower_idx - 48) // 4
                col = (lower_idx - 48) % 4
                lower_value = Y_VALUES_4[row, col]
                
            # For upper index
            upper_value = wp.float32(0.0)
            if upper_idx < 16:
                # Use Y_VALUES_1
                row = upper_idx // 4
                col = upper_idx % 4
                upper_value = Y_VALUES_1[row, col]
            elif upper_idx < 32:
                # Use Y_VALUES_2
                row = (upper_idx - 16) // 4
                col = (upper_idx - 16) % 4
                upper_value = Y_VALUES_2[row, col]
            elif upper_idx < 48:
                # Use Y_VALUES_3
                row = (upper_idx - 32) // 4
                col = (upper_idx - 32) % 4
                upper_value = Y_VALUES_3[row, col]
            else:
                # Use Y_VALUES_4
                row = (upper_idx - 48) // 4
                col = (upper_idx - 48) % 4
                upper_value = Y_VALUES_4[row, col]
                
            # Step 5: Perform linear interpolation
            return lower_value + factor * (upper_value - lower_value)

        # Add helper functions for getting values from matrices
        @wp.func
        def get_matrix_value(index: wp.int32) -> wp.float32:
            """Helper to get a value from the appropriate matrix based on index"""
            if index < 16:
                row = index // 4
                col = index % 4
                return Y_VALUES_1[row, col]
            elif index < 32:
                row = (index - 16) // 4
                col = (index - 16) % 4
                return Y_VALUES_2[row, col]
            elif index < 48:
                row = (index - 32) // 4
                col = (index - 32) % 4
                return Y_VALUES_3[row, col]
            else:
                row = (index - 48) // 4
                col = (index - 48) % 4
                return Y_VALUES_4[row, col]

        @wp.func
        def sinusoidal_flow(t: wp.float32) -> wp.float32:
            """Calculate sinusoidal flow velocity at time t"""
            # Keep angle within [0, 2π] range for numerical stability
            angle = fmod(_omega * t, wp.float32(2.0 * wp.pi))
            return _u_max * (0.5 + 0.5 * wp.sin(angle))
        
        @wp.func
        def time_dependent_velocity_functional(
            index: Any,
            timestep: Any,   
            _missing_mask: Any,     
            f_pre: Any,
            f_post: Any,
            _f_pre: Any,
            _f_post: Any,
        ):
            # Debug print occasionally
            # if timestep % 1000 == 0 and index[0] == 0 and index[1] == 5:
                # wp.printf("[DirectBC] Step %d at (%d,%d)\n", timestep, index[0], index[1])
                
            # Post-streaming values are only modified at missing direction
            _f = _f_post
            
            # Find normal vector
            normals = get_normal_vectors(_missing_mask)
            
            # Calculate time-dependent velocity
            t = _dt * wp.float32(timestep)
            
            
            # if timestep % 1000 == 0 and index[0] == 0 and index[1] == 5:
            #     wp.printf("[DirectBC] t=%f, omega*t=%f, angle=%f\n", t, _omega * t, angle)
            
            if _use_csv_profile == 1:
                # Normalize time to 0-1 range (1-second cycle)
                normalized_time = fmod(t, wp.float32(1.0))
                
                # Use our custom interpolation function
                prescribed_velocity = interpolate_flow_profile(normalized_time)
                
                # Scale by maximum velocity
                # prescribed_velocity = _u_max * 
                
                # Debug print occasionally
                if timestep % 10000 == 0 and index[0] == 0 and index[1] == 5:
                    wp.printf("[DirectBC] t=%f, normalized_time=%f, csv_velocity=%f\n", 
                             t, normalized_time, prescribed_velocity)
            else:
                # Use sinusoidal flow profile
                prescribed_velocity = sinusoidal_flow(t)
            
            # Print debug info occasionally
            # if timestep % 1000 == 0 and index[0] == 0 and index[1] == 5:
            #     wp.printf("[DirectBC] t=%f, velocity=%f, normal=(%f,%f)\n", 
            #               t, prescribed_velocity, normals[0], normals[1])
            
            # Create velocity vector
            _u = prescribed_velocity * normals
            
            # Calculate density based on velocity (like in ZouHeBC)
            fsum = _get_fsum(_f, _missing_mask)
            unormal = self.compute_dtype(0.0)
            for d in range(_d):
                unormal += _u[d] * normals[d]
            _rho = fsum / (self.compute_dtype(1.0) + unormal)
            
            # Impose non-equilibrium bounceback (like in ZouHeBC)
            _feq = self.equilibrium_operator.warp_functional(_rho, _u)
            _f = bounceback_nonequilibrium(_f, _feq, _missing_mask)
            
            return _f
            
        # Use our functional 
        if self.bc_type == "velocity":
            functional = time_dependent_velocity_functional
        else:
            raise NotImplementedError(f"BC type '{self.bc_type}' not supported in DirectTimeDependentBC")
        
        # Use parent's kernel construction with our functional
        kernel = self._construct_kernel(functional)
        
        return functional, kernel
    
    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f_pre, f_post, bc_mask, missing_mask, timestep=None):
        """Implementation for WARP backend that properly uses timestep"""
        # Debug output occasionally
        if timestep is not None and timestep % 10000 == 0:
            # print(f"DirectBC: Applying BC at timestep {timestep}")
            t = timestep * self.dt
            velocity = self.u_max * (0.5 + 0.5 * np.sin(2.0 * np.pi * self.frequency * t))
            # print(f"DirectBC: Expected velocity at t={t:.6f}s: {velocity:.6f}")
        
        # Default timestep if not provided
        if timestep is None:
            timestep = 0


        # Launch the kernel with timestep parameter
        wp.launch(
            self.warp_kernel,
            inputs=[f_pre, f_post, bc_mask, missing_mask, wp.int32(timestep)],
            dim=f_pre.shape[1:],
        )
        return f_post
    
    def aux_data_init(self, f_0, f_1, bc_mask, missing_mask):
        """Initialize with constant velocity (timestep 0)"""
        self.is_initialized_with_aux_data = True
        return f_0, f_1
    

    @partial(jit, static_argnums=(0,), inline=True)
    def _get_known_middle_mask(self, missing_mask):
        known_mask = missing_mask[self.velocity_set.opp_indices]
        middle_mask = ~(missing_mask | known_mask)
        return known_mask, middle_mask

    @partial(jit, static_argnums=(0,), inline=True)
    def _get_normal_vec(self, missing_mask):
        main_c = self.velocity_set.c[:, self.velocity_set.main_indices]
        m = missing_mask[self.velocity_set.main_indices]
        normals = -jnp.tensordot(main_c, m, axes=(-1, 0))
        return normals
    
    @partial(jit, static_argnums=(0, 2, 3), inline=True)
    def _broadcast_prescribed_values(self, prescribed_values, prescribed_values_shape, target_shape):
        """
        Broadcasts `prescribed_values` to `target_shape` following specific rules:

        - If `prescribed_values_shape` is (2, 1) or (3, 1) (for constant profiles),
          broadcast along the last 2 or 3 dimensions of `target_shape` respectively.
        - For other shapes, identify mismatched dimensions and broadcast only in that direction.
        """
        # Determine the number of dimensions to match
        num_dims_prescribed = len(prescribed_values_shape)
        num_dims_target = len(target_shape)

        if num_dims_prescribed > num_dims_target:
            raise ValueError("prescribed_values has more dimensions than target_shape")

        # Insert singleton dimensions after the first dimension to match target_shape
        if num_dims_prescribed < num_dims_target:
            # Number of singleton dimensions to add
            num_singleton = num_dims_target - num_dims_prescribed

            if num_dims_prescribed == 0:
                # If prescribed_values is scalar, reshape to all singleton dimensions
                prescribed_values_shape = (1,) * num_dims_target
            else:
                # Insert singleton dimensions after the first dimension
                prescribed_values_shape = (prescribed_values_shape[0], *(1,) * num_singleton, *prescribed_values_shape[1:])
                prescribed_values = prescribed_values.reshape(prescribed_values_shape)

        # Create broadcast shape based on the rules
        broadcast_shape = []
        for pv_dim, tgt_dim in zip(prescribed_values_shape, target_shape):
            if pv_dim == 1 or pv_dim == tgt_dim:
                broadcast_shape.append(tgt_dim)
            else:
                raise ValueError(f"Cannot broadcast dimension {pv_dim} to {tgt_dim}")

        return jnp.broadcast_to(prescribed_values, target_shape)
    
    @partial(jit, static_argnums=(0,), inline=True)
    def get_rho(self, fpop, missing_mask):
        if self.bc_type == "velocity":
            target_shape = (self.velocity_set.d,) + fpop.shape[1:]
            vel = self._broadcast_prescribed_values(self.prescribed_values, self.prescribed_values.shape, target_shape)
            rho = self.calculate_rho(fpop, vel, missing_mask)
        elif self.bc_type == "pressure":
            rho = self.prescribed_values
        else:
            raise ValueError(f"type = {self.bc_type} not supported! Use 'pressure' or 'velocity'.")
        return rho

    @partial(jit, static_argnums=(0,), inline=True)
    def get_vel(self, fpop, missing_mask):
        if self.bc_type == "velocity":
            target_shape = (self.velocity_set.d,) + fpop.shape[1:]
            vel = self._broadcast_prescribed_values(self.prescribed_values, self.prescribed_values.shape, target_shape)
        elif self.bc_type == "pressure":
            rho = self.prescribed_values
            vel = self.calculate_vel(fpop, rho, missing_mask)
        else:
            raise ValueError(f"type = {self.bc_type} not supported! Use 'pressure' or 'velocity'.")
        return vel
    
    @partial(jit, static_argnums=(0,), inline=True)
    def calculate_vel(self, fpop, rho, missing_mask):
        """
        Calculate velocity based on the prescribed pressure/density (Zou/He BC)
        """

        normals = self._get_normal_vec(missing_mask)
        known_mask, middle_mask = self._get_known_middle_mask(missing_mask)
        fsum = jnp.sum(fpop * middle_mask, axis=0, keepdims=True) + 2.0 * jnp.sum(fpop * known_mask, axis=0, keepdims=True)
        unormal = -1.0 + fsum / rho

        # Return the above unormal as a normal vector which sets the tangential velocities to zero
        vel = unormal * normals
        return vel

    @partial(jit, static_argnums=(0,), inline=True)
    def calculate_rho(self, fpop, vel, missing_mask):
        """
        Calculate density based on the prescribed velocity (Zou/He BC)
        """
        normals = self._get_normal_vec(missing_mask)
        known_mask, middle_mask = self._get_known_middle_mask(missing_mask)
        unormal = jnp.sum(normals * vel, keepdims=True, axis=0)
        fsum = jnp.sum(fpop * middle_mask, axis=0, keepdims=True) + 2.0 * jnp.sum(fpop * known_mask, axis=0, keepdims=True)
        rho = fsum / (1.0 + unormal)
        return rho
    
    @partial(jit, static_argnums=(0,), inline=True)
    def calculate_equilibrium(self, f_post, missing_mask):
        """
        This is the ZouHe method of calculating the missing macroscopic variables at the boundary.
        """
        rho = self.get_rho(f_post, missing_mask)
        vel = self.get_vel(f_post, missing_mask)

        feq = self.equilibrium_operator(rho, vel)
        return feq

    @partial(jit, static_argnums=(0,), inline=True)
    def bounceback_nonequilibrium(self, fpop, feq, missing_mask):
        """
        Calculate unknown populations using bounce-back of non-equilibrium populations
        a la original Zou & He formulation
        """
        opp = self.velocity_set.opp_indices
        fknown = fpop[opp] + feq - feq[opp]
        fpop = jnp.where(missing_mask, fknown, fpop)
        return fpop
    
    @partial(jit, static_argnums=(0))
    def _get_velocity_jax(self, timestep):
        """Calculate velocity at given timestep"""
        t = self.dt * timestep

        print(t)
        
        if self.use_csv_profile:
            # Normalize time to profile period
            t_normalized = t % self.profile_period
            
            # Find indices to interpolate between
            idx = jnp.searchsorted(jnp.array(self.profile_times), t_normalized) - 1
            idx = jnp.clip(idx, 0, len(self.profile_times) - 2)
            
            # Linear interpolation
            t0, t1 = self.profile_times[idx], self.profile_times[idx + 1]
            v0, v1 = self.profile_velocities[idx], self.profile_velocities[idx + 1]
            weight = (t_normalized - t0) / (t1 - t0)
            
            velocity = v0 + weight * (v1 - v0)
            return velocity * self.u_max
        else:
            # Simple sinusoidal flow
            angle = (2.0 * jnp.pi * self.frequency * t) % (2.0 * jnp.pi)
            return self.u_max * (0.5 + 0.5 * jnp.sin(angle))

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0))
    def jax_implementation(self, f_pre, f_post, bc_mask, missing_mask, timestep=None):
        # Default timestep if not provided
        if timestep is None:
            timestep = 0
        
        # Update prescribed_values for this timestep
        velocity = self._get_velocity_jax(timestep)
        
        # Convert scalar velocity to vector using normals
        normals = self._get_normal_vec(missing_mask)
        self.prescribed_values = velocity * normals
        
        # Rest of implementation remains the same
        boundary = bc_mask == self.id
        new_shape = (self.velocity_set.q,) + boundary.shape[1:]
        boundary = lax.broadcast_in_dim(boundary, new_shape, tuple(range(self.velocity_set.d + 1)))

        feq = self.calculate_equilibrium(f_post, missing_mask)
        f_post_bd = self.bounceback_nonequilibrium(f_post, feq, missing_mask)
        f_post = jnp.where(boundary, f_post_bd, f_post)
        return f_post