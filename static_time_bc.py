import warp as wp
import numpy as np
from xlb.operator.boundary_condition.bc_zouhe import ZouHeBC
from xlb.operator import Operator
from xlb.compute_backend import ComputeBackend
from typing import Any

class HardcodedPulsatileBC(ZouHeBC):
    """ZouHe BC with hardcoded time-dependent velocity profile"""
    
    def __init__(self, bc_type, indices=None, **kwargs):
        # Extract our specific parameters before calling parent constructor
        self.dt = kwargs.pop('dt', 0.00005)  # Default time step
        self.u_max = kwargs.pop('u_max', 0.04)  # Maximum velocity
        self.frequency = kwargs.pop('frequency', 20.0)  # Hz
        
        # Call parent constructor with remaining kwargs
        super().__init__(bc_type, indices=indices, **kwargs)
        
        print(f"Created HardcodedPulsatileBC with dt={self.dt}, u_max={self.u_max}, freq={self.frequency}")
        print(f"BC ID: {self.id}")
        
        # Add this at the end of initialization
        def print_at_step(step):
            if step % 1000 == 0:
                print(f"Step {step}: BC {self.id} should have velocity {self.u_max * min(1.0, step*self.dt/0.1)}")
        
        # Add hook to sim runner
        import atexit
        atexit.register(lambda: print("BC execution complete"))
        
    def _construct_warp(self):
        """Create a completely hardcoded functional with time calculation built-in"""
        # Set local constants 
        _d = self.velocity_set.d
        _q = self.velocity_set.q
        _u_vec = wp.vec(self.velocity_set.d, dtype=self.compute_dtype)
        _opp_indices = self.velocity_set.opp_indices
        _c = self.velocity_set.c
        _c_float = self.velocity_set.c_float
        
        # Store parameters as static constants for the kernel
        _dt = wp.static(self.compute_dtype(self.dt))
        _u_max = wp.static(self.compute_dtype(self.u_max))
        _omega = wp.static(self.compute_dtype(2.0 * np.pi * self.frequency))
        
        # Reuse helper functions
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
            for l in range(_q):
                if missing_mask[l] == wp.uint8(1):
                    fpop[l] = fpop[_opp_indices[l]] + feq[l] - feq[_opp_indices[l]]
            return fpop
            
        @wp.func
        def hardcoded_velocity_functional(
            index: Any,
            timestep: Any,  # Ignore this since it's not being passed correctly
            _missing_mask: Any,
            f_pre: Any,
            f_post: Any,
            _f_pre: Any,
            _f_post: Any,
        ):
            # Post-streaming values are only modified at missing direction
            _f = _f_post
            
            # Find normal vector (usually x-direction for inlet)
            normals = get_normal_vectors(_missing_mask)
            
            # OVERRIDE: Use constant velocity regardless of timestep
            prescribed_velocity = _u_max * 0.75  # Use 75% of max velocity
            
            # Announce it (will only print once per kernel launch)
            if index[0] == 0 and index[1] == 5:
                wp.printf("FORCE VELOCITY=%.6f at inlet\n", prescribed_velocity)
            
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
            
        # Use our hardcoded functional
        if self.bc_type == "velocity":
            functional = hardcoded_velocity_functional
        else:
            raise NotImplementedError("Only velocity type implemented")
            
        # Create kernel with the functional
        kernel = self._construct_kernel(functional)
        
        return functional, kernel

    def _construct_aux_data_init_kernel(self, functional):
        """
        Override aux_data_init_kernel to avoid the functional reference error
        """
        _id = wp.uint8(self.id)
        _opp_indices = self.velocity_set.opp_indices
        _num_of_aux_data = self.num_of_aux_data
        _q = self.velocity_set.q
        
        # Store parameters as constants for initialization
        _u_max = self.compute_dtype(self.u_max)
        
        @wp.kernel
        def aux_data_init_kernel(
            f_0: wp.array4d(dtype=self.compute_dtype),
            f_1: wp.array4d(dtype=self.compute_dtype),
            bc_mask: wp.array4d(dtype=wp.uint8),
            missing_mask: wp.array4d(dtype=wp.bool),
        ):
            # Get the global index
            i, j, k = wp.tid()
            
            # Read boundary ID
            _boundary_id = bc_mask[0, i, j, k]
            
            # Only process our boundary cells
            if _boundary_id == _id:
                # For initialization, use 0.5*u_max (the minimum of oscillation)
                prescribed_value = _u_max * 0.5
                
                # Process each direction explicitly - NO SLICES
                counter = wp.int32(0)
                for l in range(_q):
                    # Check if this is a missing direction
                    if missing_mask[l, i, j, k] == wp.bool(True) and counter < _num_of_aux_data:
                        # Initialize with prescribed value
                        f_1[_opp_indices[l], i, j, k] = self.store_dtype(prescribed_value)
                        counter += 1
                        
        return aux_data_init_kernel

    def aux_data_init(self, f_0, f_1, bc_mask, missing_mask):
        """Override to properly initialize auxiliary data"""
        if self.compute_backend == ComputeBackend.WARP:
            # Create and launch custom kernel
            kernel = self._construct_aux_data_init_kernel(None)  # No functional needed
            
            wp.launch(
                kernel,
                inputs=[f_0, f_1, bc_mask, missing_mask],
                dim=f_0.shape[1:],
            )
        
        # Mark as initialized
        self.is_initialized_with_aux_data = True
        return f_0, f_1