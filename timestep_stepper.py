from functools import partial
from jax import jit
from xlb.operator.stepper.nse_stepper import IncompressibleNavierStokesStepper
from xlb.operator import Operator
from xlb.compute_backend import ComputeBackend
from xlb.operator.boundary_condition.boundary_condition import ImplementationStep
from direct_bc import DirectTimeDependentBC  # Import your custom BC

class INSETimestepStepper(IncompressibleNavierStokesStepper):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Identify any time-dependent BCs for later use
        self.time_dependent_bcs = [bc for bc in self.boundary_conditions 
                                  if isinstance(bc, DirectTimeDependentBC)]
        print(f"Registered {len(self.time_dependent_bcs)} time-dependent boundary conditions")


    """
    Custom Navier-Stokes stepper that passes timestep to DirectTimeDependentBC instances
    while maintaining original behavior for other boundary conditions.
    """
    
    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0))
    def jax_implementation(self, f_0, f_1, bc_mask, missing_mask, timestep):
        """
        Override the JAX implementation to pass timestep to DirectTimeDependentBC
        """
        # Cast to compute precision
        f_0 = self.precision_policy.cast_to_compute_jax(f_0)
        f_1 = self.precision_policy.cast_to_compute_jax(f_1)

        # Apply streaming
        f_post_stream = self.stream(f_0)

        # Apply boundary conditions - MODIFIED to pass timestep to DirectTimeDependentBC
        for bc in self.boundary_conditions:
            if bc.implementation_step == ImplementationStep.STREAMING:
                if isinstance(bc, DirectTimeDependentBC):
                    # Pass timestep to our custom BC
                    f_post_stream = bc(
                        f_0,
                        f_post_stream,
                        bc_mask,
                        missing_mask,
                        timestep=timestep,
                    )
                else:
                    # Original call for other BCs
                    f_post_stream = bc(
                        f_0,
                        f_post_stream,
                        bc_mask,
                        missing_mask,
                    )

        # Compute the macroscopic variables
        rho, u = self.macroscopic(f_post_stream)

        # Compute equilibrium
        feq = self.equilibrium(rho, u)

        # Apply collision
        f_post_collision = self.collision(f_post_stream, feq, rho, u)

        # Apply collision type boundary conditions - MODIFIED for timestep
        for bc in self.boundary_conditions:
            f_post_collision = bc.update_bc_auxilary_data(f_post_stream, f_post_collision, bc_mask, missing_mask)
            if bc.implementation_step == ImplementationStep.COLLISION:
                if isinstance(bc, DirectTimeDependentBC):
                    # Pass timestep to our custom BC
                    f_post_collision = bc(
                        f_post_stream,
                        f_post_collision,
                        bc_mask,
                        missing_mask,
                        timestep=timestep,
                    )
                else:
                    # Original call for other BCs
                    f_post_collision = bc(
                        f_post_stream,
                        f_post_collision,
                        bc_mask,
                        missing_mask,
                    )

        # Copy back to store precision
        f_1 = self.precision_policy.cast_to_store_jax(f_post_collision)

        return f_0, f_1
    
    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f_0, f_1, bc_mask, missing_mask, timestep):
        """
        For WARP, we'll delegate to the parent class implementation.
        """
        # Call the parent implementation
        return super().warp_implementation(f_0, f_1, bc_mask, missing_mask, timestep)