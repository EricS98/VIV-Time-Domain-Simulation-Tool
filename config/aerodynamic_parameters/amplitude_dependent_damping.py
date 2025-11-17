# config/amplitude_dependent_damping.py
"""
Amplitude Dependent Damping Configuration
=========================================

Defines how aerodynamic damping Ka varies with response amplitude.

IMPORTANT: Different formulations have different input requirements:
- Vickery-Basu: Ka_r_n = Ka_max * K_a0 (height-integrated reference value)
- Lupi DMSM: Ka_r_n = K_a0,modal (height-integrated turbulence reduction factor only)
- Eurocode: Ka_r_n = Ka_max * K_a0 (similar to Vickery-Basu)
"""
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass

class AmplitudeDependentDamping(ABC):
    """Abstract base class for amplitude-dependent damping formulations."""

    @abstractmethod
    def calculate_ka(self, sigma_y: float, Ka_r_n: float, d_ref: float) -> float:
        """
        Calculate Ka as a function of amplitude.
        
        Parameters:
        -----------
        sigma_y : float
            RMS displacement [m]
        Ka_r_n : float
            Reference damping parameter (interpretation depends on model):
            - Vickery-Basu/Eurocode: Ka_max * K_a0
            - Lupi: K_a0,modal (turbulence reduction factor only)
        d_ref : float
            Reference diameter [m]
            
        Returns:
        --------
        float
            Aerodynamic damping parameter Ka
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get name of damping formulation."""
        pass

@dataclass
class VickeryBasuDamping(AmplitudeDependentDamping):
    """
    Vickery & Basu (1983) aerodynamic damping model.

    Ka(σy) = Ka_r,n * (1 - (σ_y/(aL,n*d_ref))²)

    Where Ka:r,n is the height-integrated reference value:
        Ka_r,n = ∫[Ka_max(Re(z)) * Ka_norm(U/Ucr(z), Iv(z)) * d²(z) * φ²(z)] dz 
                 / (d_ref² * ∫φ²(z) dz)

    Parameters:
    -----------
    a_L : float
        Non-dimensional limiting amplitude (default: 0.4)
    """
    a_L: float = 0.4    # Non-dimensional limiting amplitude

    def calculate_ka(self, sigma_y, Ka_r_n, d_ref):
        ratio = sigma_y / (self.a_L * d_ref)
        return max(Ka_r_n * (1 - ratio**2), 0.0)
    
    def get_name(self) -> str:
        return 'Vickery & Basu aerodynamic damping'
    
@dataclass
class LupiDamping(AmplitudeDependentDamping):
    """
    Lupi et al. (2018, 2021) aerodynamic damping model (DMSM).

    Ka(σy) = a * exp(-b*σy/d) / (σy/d)^c

    This envelope curve represents the maximum Ka values across all V/Vcr ratios
    for each amplitude. The 3D behavior Ka(σy/d, V/Vcr) is "folded" into this
    single curve.

    Turbulence modification:
    -----------------------
    The parameter Ka_r_n passed to calculate_ka() should be the height-integrated
    turbulence reduction factor K_a0,modal from simplified turbulence reduction:
    
        K_a0,modal = ∫[max(1 - 3·Iv(z), 0.25) * d²(z) * φ²(z)] dz 
                     / (d_ref² * ∫φ²(z) dz)
    
    Then: Ka(σy/d) = [a * exp(-b*σy/d) / (σy/d)^c] * K_a0,modal

    Default (recommended) parameters from Lupi et al. (2018):
    a = 0.220, b = 2.000, c = 0.500
    
    Alternative conservative parameters (Lupi et al. 2017):
    a = 0.3475, b = 5.808, c = 0.3582
    """
    a: float = 0.220
    b: float = 2.000
    c: float = 0.500

    def calculate_ka(self, sigma_y, Ka_r_n, d_ref):
        """
        Calculate Ka with turbulence modification.
        
        Parameters:
        -----------
        sigma_y : float
            RMS displacement [m]
        Ka_r_n : float
            Height-integrated turbulence reduction factor K_a0,modal
            (NOT Ka_max * K_a0 as in Vickery-Basu!)
        d_ref : float
            Reference diameter [m]
        
        Returns:
        --------
        float
            Aerodynamic damping parameter Ka
        """
        if sigma_y == 0:
            # Handle limit as σy→0
            return 1e6
        else:
            ratio = sigma_y / d_ref
            # Lupi's aerodynamic damping law
            Ka_smooth = self.a * np.exp(-self.b * ratio) / ratio**self.c
            return max(Ka_smooth * Ka_r_n, 0.0)
        
    def get_lockin_ratio(self, Sc: float) -> float:
        """
        Calculate V/Vcr ratio where maximum response occurs.
        
        Based on Lupi et al. (2017):
        - Sc ≤ 10: Wide lock-in, ymax at V ≈ 1.25·Vcr
        - Sc > 10: Narrow lock-in, ymax at V ≈ Vcr
        
        The lock-in factor (V/Vcr)² accounts for the fact that maximum 
        amplitude is reached in the lock-in range for V ≠ Vcr, not at 
        the critical velocity itself.
        
        Parameters:
        -----------
        Sc : float
            Scruton number
            
        Returns:
        --------
        float
            Velocity ratio V/Vcr where maximum response occurs
        """
        if Sc <= 10:
            return 1.25  # Wide lock-in range
        else:
            return 1.0   # Narrow lock-in range
        
    def get_name(self) -> str:
        return 'Damping-modified spectral model (Lupi) aerodynamic damping'
        
@dataclass
class EurocodeDamping(AmplitudeDependentDamping):
    """
    Eurocode draft aerodynamic damping model.

    Ka(σy) = Ka_r,n * (1 - (σy/(aL,n*d_ref))^Θ)

    Where Ka_r,n is the height-integrated reference value (similar to Vickery-Basu).
    
    Parameters:
    -----------
    a_L : float
        Non-dimensional limiting amplitude (default: 0.4)
    theta : float
        Exponent for amplitude dependency (default: 2.0)
    """
    a_L: float = 0.4
    theta: float = 2.0 

    def calculate_ka(self, sigma_y, Ka_r_n, d_ref):
        ratio = sigma_y / (self.a_L * d_ref)
        return max(Ka_r_n * (1 - ratio**self.theta), 0.0)
    
    def get_name(self) -> str:
        return 'Eurocode draft aerodynamic damping'
    
# Factory functions
def create_vickery_basu_damping(a_L: float = 0.4) -> VickeryBasuDamping:
    """Create Vickery & Basu amplitude-dependent damping."""
    return VickeryBasuDamping(a_L=a_L)

def create_lupi_damping(a: float = 0.220, b: float = 2.000, c: float = 0.500) -> LupiDamping:
    """Create Lupi amplitude-dependent damping."""
    return LupiDamping(a=a, b=b, c=c)

def create_eurocode_damping(a_L: float = 0.4, theta: float = 2.0) -> EurocodeDamping:
    """Create Eurocode amplitude-dependent damping."""
    return EurocodeDamping(a_L=a_L, theta=theta)