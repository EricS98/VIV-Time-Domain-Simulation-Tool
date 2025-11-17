# config/aerodynamic_parameters/damping_parameter.py
"""
Aerodynamic Damping Parameter Configuration
===========================================

Defines how Ka_max varies with Reynolds number.
Different standards use different curves.
"""

from abc import ABC, abstractmethod
from typing import Union
import numpy as np
from common.math_utils import loglin

class DampingParameter(ABC):
    """
    Abstract base class for aerodynamic damping parameter.
    
    Defines Ka_max(Re) relationship.
    """
    
    @abstractmethod
    def get_ka_max(self, Re: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Get maximum aerodynamic damping parameter.
        
        Parameters:
        -----------
        Re : float or np.ndarray
            Reynolds number(s)
            
        Returns:
        --------
        float or np.ndarray
            Ka_max value(s)
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get name of damping parameter model."""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
    
class CICINDDampingParameter(DampingParameter):
    """
    CICIND Model Code aerodynamic damping parameter.
    
    Ka_max(Re):
    - Re ≤ 2e5: Ka_max = 2.8
    - 2e5 < Re < 5e5: log-linear interpolation
    - Re ≥ 5e5: Ka_max = 0.9
    """
    
    def get_ka_max(self, Re: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """CICIND Ka_max(Re) curve with log-linear interpolation."""
        Re_arr = np.atleast_1d(Re)
        Ka_max = np.zeros_like(Re_arr, dtype=float)

        # Subcritical
        mask_sub = Re_arr <= 2e5
        Ka_max[mask_sub] = 2.8

        # Transition - log-linear
        mask_trans = (Re_arr > 2e5) & (Re_arr < 5e5)
        Ka_max[mask_trans] = loglin(Re_arr[mask_trans], 2e5, 2.8, 5e5, 0.9)
        
        # Supercritical
        mask_super = Re_arr >= 5e5
        Ka_max[mask_super] = 0.9
        
        return float(Ka_max[0]) if np.isscalar(Re) else Ka_max
    
    def get_name(self) -> str:
        return 'CICIND'
    
    def __repr__(self) -> str:
        return "CICINDDampingParameter()"
    
class EurocodeDampingParameter(DampingParameter):
    """
    Eurocode aerodynamic damping parameter.
    
    Ka_max(Re):
    - Re ≤ 1e5: Ka_max = 2.0
    - 1e5 < Re < 5e5: log-linear to 0.5
    - 5e5 ≤ Re < 1e6: log-linear to 1.0
    - Re ≥ 1e6: Ka_max = 1.0
    """

    def get_ka_max(self, Re: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Eurocode Ka_max(Re) with log-linear interpolation."""
        Re_arr = np.atleast_1d(Re)
        Ka_max = np.zeros_like(Re_arr, dtype=float)
        
        # Subcritical
        mask_sub = Re_arr <= 1e5
        Ka_max[mask_sub] = 2.0
        
        # First transition
        mask_trans1 = (Re_arr > 1e5) & (Re_arr < 5e5)
        Ka_max[mask_trans1] = loglin(Re_arr[mask_trans1], 1e5, 2.0, 5e5, 0.5)
        
        # Second transition
        mask_trans2 = (Re_arr >= 5e5) & (Re_arr < 1e6)
        Ka_max[mask_trans2] = loglin(Re_arr[mask_trans2], 5e5, 0.5, 1e6, 1.0)
        
        # Supercritical
        mask_super = Re_arr >= 1e6
        Ka_max[mask_super] = 1.0
        
        return float(Ka_max[0]) if np.isscalar(Re) else Ka_max
    
    def get_name(self) -> str:
        return 'Eurocode'
    
# Factory functions
def create_cicind_damping_parameter() -> CICINDDampingParameter:
    """Create CICIND damping parameter model."""
    return CICINDDampingParameter()

def create_eurocode_damping_parameter() -> EurocodeDampingParameter:
    """Create Eurocode damping parameter model."""
    return EurocodeDampingParameter()