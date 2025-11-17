# config/geometry.py
"""
Geometry Configuration - Cross-Section Variation
=================================================

Abstract base classes for defining how cross-sectional properties
(diameter) vary with height.
"""

from abc import ABC, abstractmethod
from typing import Union
import numpy as np


class CrossSection(ABC):
    """
    Abstract base class for cross-sectional geometry.

    Defines how diameter (and potentially other properties) vary with height.
    """

    @abstractmethod
    def get_diameter(self, z: Union[float, np.ndarray], h: float,
                     d_nominal: float) -> Union[float, np.ndarray]:
        """
        Get diameter at height z.
        
        Parameters:
        -----------
        z : float or np.ndarray
            Height(s) above ground [m]
        h : float
            Total structure height [m]
        d_nominal : float
            Nominal/reference diameter [m]
            
        Returns:
        --------
        float or np.ndarray
            Diameter at height z [m]
        """
        pass

    @abstractmethod
    def is_constant(self) -> bool:
        """
        Check if diameter is constant (for optimization).
        
        Returns:
        --------
        bool
            True if diameter is constant with height
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
    

class ConstantDiameter(CrossSection):
    """
    Constant diameter: d(z) = d_nominal for all z.
    
    Most common case - cylindrical structure without taper.
    """

    def __init__(self):
        """Initialize constant diameter cross-section."""
        pass

    def get_diameter(self, z: Union[float, np.ndarray], h: float,
                     d_nominal: float) -> Union[float, np.ndarray]:
        """Return constant diameter."""
        z_array = np.asarray(z)
        return np.full_like(z_array, d_nominal, dtype=float)
    
    def is_constant(self) -> bool:
        return True
    
    def __repr__(self) -> str:
        return "ConstantDiameter()"
    
class LinearTaper(CrossSection):
    """
    Linear taper: d(z) = d_base + (d_top - d_base) * (z/h)
    """
    def __init__(self, d_base: float, d_top: float):
        """
        Initialize linear taper.

        Parameters:
        -----------
        d_base : float
            Diameter at base (z=0) [m]
        d_top : float
            Diameter at top (z=h) [m]
        """
        if d_base <= 0:
            raise ValueError("Base diameter must be positive")
        if d_top <= 0:
            raise ValueError("Top diameter must be positive")
        if d_top > d_base:
            raise ValueError("Top diameter cannot exceed base diameter for taper")
        
        self.d_base = d_base
        self.d_top = d_top
        self.taper_ratio = (d_top - d_base)  # negative for normal taper

    def get_diameter(self, z: Union[float, np.ndarray], h: float, 
                    d_nominal: float) -> Union[float, np.ndarray]:
        """
        Calculate diameter using linear interpolation.
        
        Note: d_nominal is ignored for LinearTaper as d_base and d_top 
              fully define the geometry.
        """
        z_array = np.asarray(z)

        # Linear interpolation
        return self.d_base + self.taper_ratio * (z_array / h)
    
    def is_constant(self) -> bool:
        return False
    
    def __repr__(self) -> str:
        return f"LinearTaper(d_base={self.d_base:.3f} m, d_top={self.d_top:.3f} m)"
    
class BaseTaper(CrossSection):
    """
    Tapered base:
    d(z) = d_base - (d_base - d_top) / h1 * z   for 0 <= z <= h1
    d(z) = d_top                                for h1 < z <= h
    """
    def __init__(self, d_base: float, d_top: float, h1: float):
        """
        Initialize base taper.

        Parameters:
        -----------
        d_base : float
            Diameter at base (z=0) [m]
        d_top : float
            Diameter above h1 [m]
        h1 : float
            Height from base where taper ends (constant diameter starts) [m]
        """
        if d_base <= 0 or d_top <= 0 or h1 <= 0:
            raise ValueError("Diameters and h1 must be positive")
        if d_base < d_top:
            raise ValueError("Base diameter must be >= top diameter for BaseTaper")

        self.d_base = d_base
        self.d_top = d_top
        self.h1 = h1
        self.taper_slope = (d_base - d_top) / h1

    def get_diameter(self, z: Union[float, np.ndarray], h: float,
                     d_nominal: float) -> Union[float, np.ndarray]:
        """Calculate diameter with tapered base and constant top."""
        z_array = np.asarray(z)

        # Calculate diameter using the piecewise function
        diameter_tapered = self.d_base - self.taper_slope * z_array

        return np.maximum(diameter_tapered, self.d_top)
    
    def is_constant(self) -> bool:
        return False
    
    def __repr__(self) -> str:
        return (f"BaseTaper(d_base={self.d_base:.3f} m, d_top={self.d_top:.3f} m, "
                f"h1={self.h1:.3f} m)")
    
class TopTaper(CrossSection):
    """
    Tapered top:
    d(z) = d_base                                        for 0 <= z <= (h - h1)
    d(z) = d_base - (d_base - d_top)/h1 * (z - (h-h1))  for (h - h1) < z <= h
    
    Structure has constant diameter at base, then tapers to smaller diameter at top.
    """
    def __init__(self, d_base: float, d_top: float, h1: float):
        """
        Initialize top taper.

        Parameters:
        -----------
        d_base : float
            Diameter at base (constant section) [m]
        d_top : float
            Diameter at top (z=h) [m]
        h1 : float
            Height over which tapering occurs (measured from top) [m]
        """
        if d_base <= 0 or d_top <= 0 or h1 <= 0:
            raise ValueError("Diameters and h1 must be positive")
        if d_base < d_top:
            raise ValueError("Base diameter must be >= top diameter for TopTaper")

        self.d_base = d_base
        self.d_top = d_top
        self.h1 = h1
        self.taper_slope = (d_base - d_top) / h1

    def get_diameter(self, z: Union[float, np.ndarray], h: float,
                     d_nominal: float) -> Union[float, np.ndarray]:
        """Calculate diameter with constant base and tapered top."""
        z_array = np.asarray(z)
        
        # Height where tapering starts
        z_taper_start = h - self.h1
        
        # Initialize with base diameter
        d = np.full_like(z_array, self.d_base, dtype=float)
        
        # Apply taper for z > z_taper_start
        mask = z_array > z_taper_start
        d[mask] = self.d_base - self.taper_slope * (z_array[mask] - z_taper_start)
        
        return d
    
    def is_constant(self) -> bool:
        return False
    
    def __repr__(self) -> str:
        return (f"TopTaper(d_base={self.d_base:.3f} m, d_top={self.d_top:.3f} m, "
                f"h1={self.h1:.3f} m)")

    
# Factory functions

def create_constant_diameter() -> ConstantDiameter:
    """Create constant diameter cross-section."""
    return ConstantDiameter()

def create_linear_taper(d_base: float, d_top: float) -> LinearTaper:
    """Create linear taper cross-section."""
    return LinearTaper(d_base, d_top)

def create_base_taper(d_base: float, d_top: float, h1: float) -> BaseTaper:
    """Create base taper cross-section."""
    return BaseTaper(d_base, d_top, h1)

def create_top_taper(d_base: float, d_top: float, h1: float) -> TopTaper:
    """Create top taper cross-section."""
    return TopTaper(d_base, d_top, h1)