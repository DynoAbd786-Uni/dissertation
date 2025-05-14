from dataclasses import dataclass
from typing import Dict, Any
import numpy as np

@dataclass
class BloodFlowProfile:
    """Physical parameters for blood flow simulation"""
    density: float  # kg/m³
    kinematic_viscosity: float  # m²/s
    characteristic_velocity: float  # m/s
    vessel_diameter: float  # m
    heart_rate: int  # beats per minute
    
    @property
    def reynolds_number(self) -> float:
        """Calculate Reynolds number"""
        return (self.characteristic_velocity * self.vessel_diameter) / self.kinematic_viscosity

    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary"""
        return {
            "physical_parameters": {
                "density": self.density,
                "kinematic_viscosity": self.kinematic_viscosity,
                "characteristic_velocity": self.characteristic_velocity,
                "vessel_diameter": self.vessel_diameter,
                "heart_rate": self.heart_rate,
                "reynolds_number": self.reynolds_number
            }
        }

class BloodFlowProfiles:
    """Standard blood flow profiles for different vessel types"""
    
    @staticmethod
    def cerebral_artery() -> BloodFlowProfile:
        """Standard profile for cerebral artery"""
        return BloodFlowProfile(
            density=1060,  # kg/m³
            kinematic_viscosity=3.3e-6,  # m²/s
            characteristic_velocity=0.4,  # m/s
            vessel_diameter=2e-3,  # 2mm
            heart_rate=60
        )
    
    @staticmethod
    def aneurysm() -> BloodFlowProfile:
        """Profile for aneurysm simulation"""
        return BloodFlowProfile(
            density=1060,  # kg/m³
            kinematic_viscosity=3.3e-6,  # m²/s
            characteristic_velocity=0.3,  # m/s (reduced flow)
            vessel_diameter=4e-3,  # 4mm (dilated vessel)
            heart_rate=60
        )
    
    @staticmethod
    def custom(
        density: float = 1060,
        kinematic_viscosity: float = 3.3e-6,
        characteristic_velocity: float = 0.4,
        vessel_diameter: float = 2e-3,
        heart_rate: int = 60
    ) -> BloodFlowProfile:
        """Create custom blood flow profile"""
        return BloodFlowProfile(
            density=density,
            kinematic_viscosity=kinematic_viscosity,
            characteristic_velocity=characteristic_velocity,
            vessel_diameter=vessel_diameter,
            heart_rate=heart_rate
        )