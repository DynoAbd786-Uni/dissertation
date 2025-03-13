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
from scipy.interpolate import interp1d

class DirectTimeDependentBC(BoundaryCondition):
    """
    A time-dependent velocity boundary condition that follows ZouHeBC's structure
    but adds time dependence to the velocity.
    """
    
    def __init__(self, dt, dx, u_max, frequency, flow_profile=None,
                 bc_type="velocity", indices=None,
                 **kwargs):  # Keep kwargs for parent class
        """Initialize the DirectTimeDependentBC class
        
        Args:
            bc_type: Must be 'velocity' for this BC
            indices: Boundary indices
            dt: Time step size (default: 0.00005)
            dx: Spatial step size (default: 0.0001)
            u_max: Maximum velocity (default: 0.04)
            frequency: Oscillation frequency in Hz (default: 1.0)
            flow_profile: Optional dictionary with flow profile data
                         {'name': profile_name, 'data': {'x': time_array, 'y': velocity_array}}
            **kwargs: Optional parameters
        """
        # Verify bc_type is supported
        assert bc_type == "velocity", "DirectTimeDependentBC only supports 'velocity' type boundary conditions"
        self.bc_type = bc_type
        
        # Store time parameters
        self.dt = dt
        self.dx = dx
        self.u_max_physical = u_max
        self.frequency = frequency
        self.u_max = self.u_max_physical * (self.dt/self.dx)
        
        # Add equilibrium operator like ZouHeBC
        self.equilibrium_operator = QuadraticEquilibrium()
        
        # Call parent constructor with proper implementation step
        super().__init__(
            ImplementationStep.STREAMING,  # Same as ZouHeBC
            indices=indices,
            **kwargs
        )

        # Print all parameters
        # print(f"Created DirectTimeDependentBC with:")
        # print(f"  dt={self.dt}, dx={self.dx}")
        # print(f"  u_max_physical={self.u_max_physical} m/s (physical)")
        # print(f"  u_max_lattice={self.u_max} LU (lattice)")
        # print(f"  freq={self.frequency} Hz (physical)")
        # print(f"  One oscillation should take {1.0/(self.frequency*self.dt):,.0f} timesteps")
        
        
        # Set needs_aux flags like ZouHeBC
        self.needs_aux_init = True
        self.needs_aux_recovery = True
        self.num_of_aux_data = 1  # One aux data for velocity
        self.needs_padding = True

        # Store flow profile data if supplied
        self.flow_profile = flow_profile
        self.use_csv_profile = False
        self.profile_times = None
        self.profile_velocities = None
        self.interpolator = None
        self.profile_period = None
        
        if flow_profile is not None and isinstance(flow_profile, dict) and 'name' in flow_profile:
            self.profile_name = flow_profile['name']
            
            # Check if we have actual data or just using default sinusoidal
            if flow_profile.get('data') is not None and 'x' in flow_profile['data'] and 'y' in flow_profile['data']:
                self.flow_profile_data = flow_profile['data']
                self.profile_times = self.flow_profile_data['x']
                self.profile_velocities = self.flow_profile_data['y']
                
                # Create interpolator for velocity values
                self.use_csv_profile = True
                
                # Calculate period (duration) of the flow profile
                self.profile_period = float(self.profile_times[-1] - self.profile_times[0])
                
                # Create interpolator (cyclic/periodic)
                # We'll handle the periodicity in the kernel
                self.interpolator = interp1d(
                    self.profile_times, 
                    self.profile_velocities,
                    bounds_error=False,
                    fill_value=(self.profile_velocities[0], self.profile_velocities[-1])
                )
                
                # Find max velocity in profile for normalization
                self.profile_max_vel = np.max(np.abs(self.profile_velocities))
                
                print(f"Using CSV flow profile: {self.profile_name}")
                print(f"  Profile duration: {self.profile_period:.4f} seconds")
                print(f"  Profile max velocity: {self.profile_max_vel:.4f}")
                print(f"  Profile points: {len(self.profile_times)}")
            else:
                print(f"Using sinusoidal flow profile: {self.profile_name}")
                self.flow_profile_data = None
        else:
            self.profile_name = "Sinusoidal_1Hz"
            self.flow_profile_data = None
            print(f"Using default sinusoidal flow profile")
        
        # Upload flow profile data to device if using CSV
        if self.use_csv_profile:
            # Convert to float32 arrays for warp
            times = self.profile_times.astype(np.float32)
            velocities = self.profile_velocities.astype(np.float32)
            
            # Create device arrays
            self.times_device = wp.array(times, dtype=wp.float32)
            self.velocities_device = wp.array(velocities, dtype=wp.float32)
            self.num_points = wp.int32(len(times))
            self.period = wp.float32(self.profile_period)            
        
        # print(f"Created DirectTimeDependentBC with dt={self.dt}, u_max={self.u_max}, freq={self.frequency}")
        # print(f"BC ID: {self.id}")
        
    def _construct_warp(self):
        """Construct the WARP kernel and functional"""
        # Set local constants
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
            
            # Keep angle within [0, 2π] range for numerical stability
            angle = fmod(_omega * t, wp.float32(2.0 * wp.pi))
            # if timestep % 1000 == 0 and index[0] == 0 and index[1] == 5:
            #     wp.printf("[DirectBC] t=%f, omega*t=%f, angle=%f\n", t, _omega * t, angle)
            
            prescribed_velocity = _u_max * (wp.float32(0.5) + wp.float32(0.5) * wp.sin(angle))
            
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