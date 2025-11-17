# config/reference_diameter.py
"""
Reference Diameter Calculation Module
======================================

Provides different methods for calculating the reference diameter used
in VIV analysis normalization.

Background
----------
In spectral VIV analysis, many parameters are normalized by a reference
diameter. Different standards and researchers use different definitions:

1. TOP: Top diameter (Vickery & Basu 1983)
   - Simple and traditional
   - May overestimate tip effects for tapered structures

2. TOP_THIRD_AVG: Average diameter over top third (CICIND Model Code)
   - Reduces sensitivity to tip effects
   - Good practical compromise

3. EFFECTIVE: Mode-shape weighted effective diameter (Livanos 2024)
   - Theoretically most accurate
   - Accounts for modal participation
   - Formula: d_eff = sqrt(∫₀ʰ d²(z) Φₙ²(z) dz / ∫₀ʰ Φₙ²(z) dz)
"""

from enum import Enum
from typing import Union, Optional, TYPE_CHECKING
import numpy as np
from dataclasses import dataclass

# Import types for type checking only
if TYPE_CHECKING:
    from common.structures import StructureProperties
    from config.geometry import CrossSection
    from config.mode_shape import ModeShape


class ReferenceDiameterMethod(Enum):
    """
    Methods for calculating reference diameter.
    
    Attributes
    ----------
    TOP : Use top diameter (original Vickery & Basu)
    TOP_THIRD_AVG : Average over top third of height (CICIND)
    EFFECTIVE : Mode-shape weighted effective diameter (Livanos)
    """
    TOP = "top"
    TOP_THIRD_AVG = "top_third"
    EFFECTIVE = "effective"
    
    def __str__(self):
        return self.value
    
    @classmethod
    def from_string(cls, method_str: str):
        """Create enum from string."""
        method_map = {
            'top': cls.TOP,
            'top_third': cls.TOP_THIRD_AVG,
            'top_third_avg': cls.TOP_THIRD_AVG,
            'effective': cls.EFFECTIVE,
        }
        method_lower = method_str.lower()
        if method_lower not in method_map:
            raise ValueError(
                f"Unknown reference diameter method: {method_str}. "
                f"Options: {list(method_map.keys())}"
            )
        return method_map[method_lower]


@dataclass
class ReferenceDiameterConfig:
    """
    Configuration for reference diameter calculation.
    
    Parameters
    ----------
    method : ReferenceDiameterMethod
        Calculation method to use
    n_integration_points : int
        Number of points for numerical integration (for EFFECTIVE method)
    """
    method: ReferenceDiameterMethod = ReferenceDiameterMethod.TOP_THIRD_AVG
    n_integration_points: int = 100
    
    def __post_init__(self):
        """Validate configuration."""
        if self.n_integration_points < 10:
            raise ValueError("n_integration_points must be at least 10")
    
    def __repr__(self):
        return f"ReferenceDiameterConfig(method={self.method.value})"


class ReferenceDiameterCalculator:
    """
    Calculate reference diameter using various methods.
    
    This class provides a unified interface for computing the reference
    diameter used in VIV analysis normalization.
    """
    
    @staticmethod
    def calculate(
        config: ReferenceDiameterConfig,
        cross_section: 'CrossSection',
        height: float,
        d_nominal: float,
        mode_shape: Optional['ModeShape'] = None,
        mode_number: int = 1
    ) -> float:
        """
        Calculate reference diameter based on configuration.
        
        Parameters
        ----------
        config : ReferenceDiameterConfig
            Configuration specifying calculation method
        cross_section : CrossSection
            Cross-section geometry definition
        height : float
            Total structure height [m]
        d_nominal : float
            Nominal diameter [m] (used by cross_section interface)
        mode_shape : ModeShape, optional
            Mode shape definition (required for EFFECTIVE method)
        mode_number : int
            Mode number for mode shape (default: 1)
            
        Returns
        -------
        float
            Reference diameter [m]
            
        Raises
        ------
        ValueError
            If required inputs are missing for selected method
        """
        # For constant diameter, all methods return d_nominal
        if cross_section.is_constant():
            return d_nominal
        
        # For variable diameter, proceed with selected method
        method = config.method
        
        if method == ReferenceDiameterMethod.TOP:
            return ReferenceDiameterCalculator._calculate_top(
                cross_section, height, d_nominal
            )
        
        elif method == ReferenceDiameterMethod.TOP_THIRD_AVG:
            return ReferenceDiameterCalculator._calculate_top_third_avg(
                cross_section, height, d_nominal, config.n_integration_points
            )
        
        elif method == ReferenceDiameterMethod.EFFECTIVE:
            if mode_shape is None:
                raise ValueError("mode_shape is required for EFFECTIVE method")
            return ReferenceDiameterCalculator._calculate_effective(
                cross_section, height, d_nominal, mode_shape, 
                mode_number, config.n_integration_points
            )
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    @staticmethod
    def _calculate_top(
        cross_section: 'CrossSection',
        height: float,
        d_nominal: float
    ) -> float:
        """
        Calculate top diameter.
        
        Returns the diameter at the top of the structure (z=h).
        This is the traditional Vickery & Basu (1983) approach.
        """
        return cross_section.get_diameter(height, height, d_nominal)
    
    @staticmethod
    def _calculate_top_third_avg(
        cross_section: 'CrossSection',
        height: float,
        d_nominal: float,
        n_points: int = 100
    ) -> float:
        """
        Calculate average diameter over top third of structure.
        
        This approach is recommended by CICIND Model Code to reduce
        sensitivity to tip effects while remaining computationally simple.
        
        Formula: d_ref = (1/H_top) ∫_{2h/3}^h d(z) dz
        """
        # Integration over top third: z ∈ [2h/3, h]
        z_start = 2 * height / 3
        z_end = height
        z_points = np.linspace(z_start, z_end, n_points)
        
        # Get diameters at integration points
        d_z = cross_section.get_diameter(z_points, height, d_nominal)
        
        # Compute average using trapezoidal rule
        d_avg = np.trapz(d_z, z_points) / (z_end - z_start)
        
        return float(d_avg)
    
    @staticmethod
    def _calculate_effective(
        cross_section: 'CrossSection',
        height: float,
        d_nominal: float,
        mode_shape: 'ModeShape',
        mode_number: int = 1,
        n_points: int = 100
    ) -> float:
        """
        Calculate mode-shape weighted effective diameter (Livanos 2024).
        
        This method accounts for the fact that different sections of the
        structure contribute differently to the response based on the
        mode shape amplitude.
        
        Formula: d_eff = sqrt(∫₀ʰ d²(z) Φₙ²(z) dz / ∫₀ʰ Φₙ²(z) dz)
        
        Where:
        - d(z) is the diameter at height z
        - Φₙ(z) is the mode shape (normalized to 1 at top)
        - n is the mode number
        
        References
        ----------
        Livanos et al. (2024): "An enhanced analytical calculation model 
        based on sectional calculation using a 3D contour map of aerodynamic 
        damping for vortex induced vibrations of wind turbine towers"
        """
        # Create height discretization
        z_points = np.linspace(0, height, n_points)
        
        # Get diameters at each height
        d_z = cross_section.get_diameter(z_points, height, d_nominal)
        
        # Get mode shape at each height
        phi_z = mode_shape.evaluate(z_points, height, mode_number=mode_number)
        
        # Compute weighted integral: ∫ d²(z) Φ²(z) dz
        numerator = np.trapezoid(d_z**2 * phi_z**2, z_points)
        
        # Compute mode shape integral: ∫ Φ²(z) dz
        denominator = np.trapezoid(phi_z**2, z_points)
        
        # Effective diameter
        d_eff = np.sqrt(numerator / denominator)
        
        return float(d_eff)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_top_diameter_config() -> ReferenceDiameterConfig:
    """Create config for top diameter method."""
    return ReferenceDiameterConfig(method=ReferenceDiameterMethod.TOP)

def create_top_third_config(n_points: int = 100) -> ReferenceDiameterConfig:
    """Create config for top-third average method (CICIND)."""
    return ReferenceDiameterConfig(
        method=ReferenceDiameterMethod.TOP_THIRD_AVG,
        n_integration_points=n_points
    )

def create_effective_diameter_config(n_points: int = 100) -> ReferenceDiameterConfig:
    """Create config for effective diameter method (Livanos)."""
    return ReferenceDiameterConfig(
        method=ReferenceDiameterMethod.EFFECTIVE,
        n_integration_points=n_points
    )