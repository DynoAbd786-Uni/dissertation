"""
Spatial Flow Profiles Module

This module contains different spatial velocity profile implementations
for fluid flow simulations, supporting both JAX and Warp backends.
"""

import warp as wp
import jax.numpy as jnp
from abc import ABC, abstractmethod
from xlb.compute_backend import ComputeBackend

class SpatialFlowProfile(ABC):
    """Base class for spatial flow profiles"""
    
    def __init__(self, vessel_diameter=None, vessel_centre=None, grid_shape=None, backend=None, precision_policy=None, u_max=0.04):
        # Vessel-specific parameters (preferred)
        self.vessel_diameter = vessel_diameter
        self.vessel_centre = vessel_centre
        
        # Grid parameters (for backward compatibility)
        self.grid_shape = grid_shape
        self.backend = backend
        self.precision_policy = precision_policy
        self.u_max = u_max
        
        # Calculate effective channel height
        if vessel_diameter is not None:
            self.channel_height = vessel_diameter
        elif grid_shape is not None:
            self.channel_height = grid_shape[1] - 1  # Backward compatibility
        else:
            raise ValueError("Either vessel_diameter or grid_shape must be provided")
    
    @abstractmethod
    def get_profile_function(self):
        """Return the appropriate profile function for the selected backend"""
        pass


class PoiseuilleProfile(SpatialFlowProfile):
    """
    Poiseuille Flow Profile
    
    Implements a parabolic velocity distribution characteristic of 
    fully developed laminar flow in a channel or pipe.
    """
    
    def get_profile_function(self):
        """Return Poiseuille flow profile function"""
        u_max = self.u_max
        H_y = float(self.channel_height)  # Use vessel diameter or grid height
        vessel_centre_y = float(self.vessel_centre) if self.vessel_centre is not None else H_y / 2.0

        @wp.func
        def poiseuille_profile_warp(index: wp.vec3i):
            # Poiseuille flow profile: parabolic velocity distribution
            y = self.precision_policy.store_precision.wp_dtype(index[1])

            # Calculate distance from vessel centerline
            y_from_center = y - vessel_centre_y
            
            # Normalized radial distance from centerline (r/R where R = H_y/2)
            r_normalized = wp.abs(2.0 * y_from_center / H_y)

            # Parabolic profile: u = u_max * (1 - (r/R)²)
            # Only apply profile within the vessel diameter
            if wp.abs(y_from_center) <= H_y / 2.0:
                velocity = u_max * wp.max(0.0, 1.0 - r_normalized * r_normalized)
            else:
                velocity = 0.0

            return wp.vec(velocity, length=1)

        def poiseuille_profile_jax():
            if self.grid_shape is not None:
                y = jnp.arange(self.grid_shape[1])
            else:
                # If no grid shape, create array for vessel region
                y = jnp.arange(int(vessel_centre_y - H_y/2), int(vessel_centre_y + H_y/2) + 1)

            # Calculate distance from vessel centerline
            y_from_center = y - vessel_centre_y
            
            # Normalized radial distance from centerline
            r_normalized = jnp.abs(2.0 * y_from_center / H_y)

            # Parabolic profile for x velocity, zero for y
            # Only apply profile within the vessel diameter
            mask = jnp.abs(y_from_center) <= H_y / 2.0
            u_x = jnp.where(mask, u_max * jnp.maximum(0.0, 1.0 - r_normalized**2), 0.0)
            u_y = jnp.zeros_like(u_x)

            return jnp.stack([u_x, u_y])

        if self.backend == ComputeBackend.JAX:
            return poiseuille_profile_jax
        elif self.backend == ComputeBackend.WARP:
            return poiseuille_profile_warp


class BluntedParaboloidProfile(SpatialFlowProfile):
    """
    Blunted Paraboloid Flow Profile (Power-Law)
    
    Implements a power-law velocity distribution characteristic of 
    fully developed non-Newtonian flow (e.g., blood flow).
    
    Profile: v(r) = v_max * (1 - (r/R)^n)
    Where n ∈ [1.5, 1.9] for blood flow applications.
    """
    
    def __init__(self, vessel_diameter=None, vessel_centre=None, grid_shape=None, backend=None, precision_policy=None, u_max=0.04, n=1.7, scale_factor=1.0):
        """
        Initialize blunted paraboloid profile
        
        Args:
            vessel_diameter: Diameter of the vessel in lattice units
            vessel_centre: Centre position of the vessel in lattice units
            grid_shape: Grid dimensions (for backward compatibility)
            backend: Compute backend (JAX or WARP)
            precision_policy: Precision policy for computations
            u_max: Maximum velocity at center
            n: Power-law exponent (1.5-1.9 for blood)
            scale_factor: Time-varying scaling factor Scale(t)
        """
        super().__init__(vessel_diameter, vessel_centre, grid_shape, backend, precision_policy, u_max)
        self.n = n
        self.scale_factor = scale_factor
    
    def update_scale_factor(self, new_scale_factor):
        """Update the time-varying scale factor"""
        self.scale_factor = new_scale_factor
    
    def get_profile_function(self):
        """Return blunted paraboloid flow profile function"""
        u_max = self.u_max
        n = self.n
        scale_factor = self.scale_factor
        H_y = float(self.channel_height)  # Use vessel diameter or grid height
        R = H_y / 2.0  # Radius (half of vessel diameter)
        vessel_centre_y = float(self.vessel_centre) if self.vessel_centre is not None else H_y / 2.0

        @wp.func
        def blunted_paraboloid_profile_warp(index: wp.vec3i):
            # Blunted paraboloid profile: power-law velocity distribution
            y = self.precision_policy.store_precision.wp_dtype(index[1])

            # Calculate distance from vessel centerline
            y_from_center = y - vessel_centre_y
            r = wp.abs(y_from_center)  # Radial distance from centerline
            
            # Normalized radius r/R
            r_normalized = r / R
            
            # Power-law profile: v(r) = Scale(t) * v_max * (1 - (r/R)^n)
            # Only apply profile within the vessel diameter
            if r <= R:
                velocity_magnitude = scale_factor * u_max * wp.max(0.0, 1.0 - wp.pow(r_normalized, n))
            else:
                velocity_magnitude = 0.0
            
            return wp.vec(velocity_magnitude, length=1)

        def blunted_paraboloid_profile_jax():
            if self.grid_shape is not None:
                y = jnp.arange(self.grid_shape[1])
            else:
                # If no grid shape, create array for vessel region
                y = jnp.arange(int(vessel_centre_y - R), int(vessel_centre_y + R) + 1)

            # Calculate distance from vessel centerline
            y_from_center = y - vessel_centre_y
            r = jnp.abs(y_from_center)  # Radial distance from centerline
            
            # Normalized radius r/R
            r_normalized = r / R
            
            # Power-law profile for x velocity: v(r) = Scale(t) * v_max * (1 - (r/R)^n)
            # Only apply profile within the vessel diameter
            mask = r <= R
            u_x = jnp.where(mask, scale_factor * u_max * jnp.maximum(0.0, 1.0 - jnp.power(r_normalized, n)), 0.0)
            u_y = jnp.zeros_like(u_x)

            return jnp.stack([u_x, u_y])

        if self.backend == ComputeBackend.JAX:
            return blunted_paraboloid_profile_jax
        elif self.backend == ComputeBackend.WARP:
            return blunted_paraboloid_profile_warp


# Legacy function for backward compatibility
def bc_profile(self):
    """
    Legacy boundary condition profile function
    
    This function maintains backward compatibility while redirecting
    to the new PoiseuilleProfile class.
    """
    profile = PoiseuilleProfile(
        grid_shape=self.grid_shape,
        backend=self.backend,
        precision_policy=self.precision_policy,
        u_max=self.u_max
    )
    return profile.get_profile_function()