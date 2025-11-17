# config/aerodynamic_parameters/lift_coefficient.py
"""
RMS Lift Coefficient Configuration
===================================

Defines how the RMS lift coefficient σ_CL varies with Reynolds number.
Different standards use different curves.
"""

from abc import ABC, abstractmethod
from typing import Union
import numpy as np
from common.math_utils import loglin

class LiftCoefficient(ABC):
    """
    Abstract base class for RMS lift coefficient.

    Defines σ_CL(Re) relationship.
    """
    @abstractmethod
    def get_sigma_cl(self, Re: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Get RMS lift coefficient as function of Reynolds number.

        Parameters:
        -----------
        Re : float or np.ndarray
            Reynolds number(s)
            
        Returns:
        --------
        float or np.ndarray
            σ_CL value(s)
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get name of lift coefficient model."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
    
class CICINDLiftCoefficient(LiftCoefficient):
    """
    CICIND Model Code RMS lift coefficient

    σ_CL(Re):
    - Re ≤ 2e5: σ_CL = 0.7
    - 2e5 < Re < 5e5: log-linear interpolation
    - Re ≥ 5e5: σ_CL = 0.2
    """
    def get_sigma_cl(self, Re: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """CICIND σ_CL(Re) curve with log-linear interpolation."""
        Re_arr = np.atleast_1d(Re)
        sigma_cl = np.zeros_like(Re_arr, dtype=float)

        # Subcritical regime (Re < 2e5)
        mask_sub = Re_arr < 2e5
        sigma_cl[mask_sub] = 0.7
        
        # Transition regime (2e5 < Re < 5e5) - log-linear interpolation
        mask_trans = (Re_arr > 2e5) & (Re_arr < 5e5)
        sigma_cl[mask_trans] = loglin(Re_arr[mask_trans], 2e5, 0.7, 5e5, 0.2)
        
        # Supercritical regime (Re >= 5e5)
        mask_super = Re_arr >= 5e5
        sigma_cl[mask_super] = 0.2
        
        return float(sigma_cl[0]) if np.isscalar(Re) else sigma_cl
    
    def get_name(self) -> str:
        return 'CICIND'
    
    def __repr__(self) -> str:
        return "CICINDLiftCoefficient()"
    
# Factory functions
def create_cicind_lift_coefficient() -> CICINDLiftCoefficient:
    """Create CICIND lift coefficient model."""
    return CICINDLiftCoefficient()