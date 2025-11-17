# visualization/height_profile_data.py
"""
Height Profile Data Container
=============================

Dataclass for storing height-dependent structural and wind properties
for visualization purposes.
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class HeightProfileData:
    """
    Container for height-dependent properties used in VIV analysis.

    All arrays should have the same length corresponding to discretization
    points along the structure.

    Attributes:
    -----------
    z : np.ndarray
        Height coordinates [m]
    
    Structural Properties:
    d : np.ndarray
        Diameter profile [m]
    phi : np.ndarray
        Mode shape values [-]
    
    Wind Properties:
    u : np.ndarray
        Wind velocity profile [m/s]
    Iv : np.ndarray
        Turbulence intensity profile [-]
    
    Spectral Model Parameters:
    B : np.ndarray
        Bandwidth parameter [-]
    f_s : np.ndarray
        Vortex shedding frequency profile [Hz]
    U_cr : np.ndarray
        Critical velocity profile [m/s]
    velocity_ratio : np.ndarray
        Velocity ratio u/U_cr [-]
    
    Optional (for future):
    sigma_cl : Optional[np.ndarray]
        RMS lift coefficient profile [-]
    Ka : Optional[np.ndarray]
        Aerodynamic damping parameter profile [-]
    """
    # Height array
    z: np.ndarray

    # Structural properties
    d: np.ndarray
    phi: np.ndarray

    # Wind properties
    u: np.ndarray
    Iv: np.ndarray

    # Spectral model parameters
    B: np.ndarray
    f_s: np.ndarray
    U_cr: np.ndarray
    velocity_ratio: np.ndarray

    # Optional future parameters
    sigma_cl: Optional[np.ndarray] = None
    Ka: Optional[np.ndarray] = None

    def __post_init__(self):
        """Validate that all arrays have consistent shapes."""
        n_points = len(self.z)

        arrays_to_check = {
            'diameter': self.d,
            'mode_shape': self.phi,
            'velocity': self.u,
            'turbulence_intensity': self.Iv,
            'shedding_frequency': self.f_s,
            'critical_velocity': self.U_cr,
            'velocity_ratio': self.velocity_ratio
        }

        for name, array in arrays_to_check.items():
            if len(array) != n_points:
                raise ValueError(
                    f"Array '{name}' has length {len(array)}, "
                    f"expected {n_points} to match height array"
                )
            
    @property
    def n_points(self) -> int:
        """Number of discretization points."""
        return len(self.z)
    
    @property
    def height(self) -> float:
        """Total structure height [m]."""
        return float(self.z[-1])
    
    def __repr__(self) -> str:
        return (f"HeightProfileData(n_points={self.n_points}, "
                f"height={self.height:.1f}m)")