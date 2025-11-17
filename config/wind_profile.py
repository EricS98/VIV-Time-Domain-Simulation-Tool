# config/wind_profile.py
"""
Wind Profile Configuration - Base Classes

Abstract Base classes for wind profiles with concrete implementations.
Easily extensible for new profile types (e.g., logarithmic).
"""

from abc import ABC, abstractmethod
from typing import Union, Optional
import numpy as np

# ============================================================================
# BASE CLASS
# ============================================================================

class WindProfile(ABC):
    """
    Abstract base class for wind profiles.

    Defines the interface that all wind profiles must implement.
    Includes both velocity and turbulence intensity.
    """

    @abstractmethod
    def get_velocity(self, z: Union[float, np.ndarray], h: float) -> Union[float, np.ndarray]:
        """
        Get wind velocity at height z.

        Parameters:
        -----------
        z : float or np.ndarray
            Height(s) above ground [m]
        h : float
            Total structure height [m] (for reference)
            
        Returns:
        --------
        float or np.ndarray
            Wind velocity at height z [m/s]
        """
        pass

    @abstractmethod
    def get_turbulence_intensity(self, z: Union[float, np.ndarray], h: float = None) -> Union[float, np.ndarray]:
        """
        Get turbulence intensity at height z.

        Parameters:
        -----------
        z : float or np.ndarray
            Height(s) above ground [m]
            
        Returns:
        --------
        float or np.ndarray
            Turbulence intensity at height z [-]
        """

    @abstractmethod
    def is_constant(self) -> bool:
        """
        Check if wind profile is constant.

        Returns:
        --------
        bool
            True if wind velocity is constant with height
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
    
# ============================================================================
# CONCRETE IMPLEMENTATIONS
# ============================================================================

class ConstantWindProfile(WindProfile):
    """
    Constant wind profile: u(z) = u_ref for all z.

    Simplest case - uniform wind velocity over height.
    """

    def __init__(self, u_ref: float, Iv: float = 0.10):
        """
        Initialize constant wind profile.
        
        Parameters:
        -----------
        u_ref : float
            Reference wind velocity [m/s]
        Iv: float
            Turbulence intensity [-]
        """
        if u_ref <= 0:
            raise ValueError("Reference wind velocity must be positive")
        if not (0 <= Iv <= 1):
            raise ValueError("Turbulence intensity must be in [0, 1]")
        
        self.u_ref = u_ref
        self.Iv = Iv

    def get_velocity(self, z: Union[float, np.ndarray], h: float) -> Union[float, np.ndarray]:
        """Return constant velocity."""
        z_array = np.asarray(z)
        return np.full_like(z_array, self.u_ref, dtype=float)
    
    def get_turbulence_intensity(self, z: Union[float, np.ndarray], h: float = None) -> Union[float, np.ndarray]:
        """Return constant turbulence intensity."""
        z_array = np.asarray(z)
        return np.full_like(z_array, self.Iv, dtype=float)
    
    def is_constant(self) -> bool:
        return True
    
    def __repr__(self) -> str:
        return (f"ConstantWindProfile(u_ref={self.u_ref:.2f} m/s, "
                f"Iv={self.Iv:.2f})")

    
class PowerLawWindProfile(WindProfile):
    """
    Power law wind profile: 
    - u(z) = u_ref * (z / z_ref)^alpha
    - Iv(z) = gamma * (z / 10)^(-alpha)

    Most common wind profile for engineering purposes.
    """

    def __init__(self, u_ref: float, z_ref: float = 10.0, alpha: float = 0.16,
                gamma: float = 0.19, z_min: Optional[float] = None):
        """
        Initialize power law wind profile.
        
        Parameters:
        -----------
        u_ref : float
            Reference wind velocity at z_ref [m/s]
        z_ref : float
            Reference height [m]
        alpha : float
            Power law exponent (terrain-dependent):
            - 0.12: Terrain I (sea, flat)
            - 0.16: Terrain II (low vegetation)
            - 0.22: Terrain III (urban, forest)
        gamma : float
            Turbulence intensity factor (terrain-dependent):
            - 0.14: Terrain I
            - 0.19: Terrain II
            - 0.28: Terrain III
        z_min : float, optional
            Minimum height below which profile is constant (plateau)
        """
        # Validations
        if u_ref <= 0:
            raise ValueError("Reference wind velocity must be positive")
        if z_ref <= 0:
            raise ValueError("Reference height must be positive")
        if not (0 < alpha < 1):
            raise ValueError("Power law exponent alpha must be in (0, 1)")
        if gamma <= 0:
            raise ValueError("Turbulence factor gamma must be positive")
        if z_min is not None and z_min < 0:
            raise ValueError("Minimum height z_min must be non-negative")
        
        self.u_ref = u_ref
        self.z_ref = z_ref
        self.alpha = alpha
        self.gamma = gamma
        self.z_min = z_min if z_min is not None else 0.0

    def get_velocity(self, z: Union[float, np.ndarray], h: float) -> Union[float, np.ndarray]:
        """Calculate wind velocity using power law."""
        z_array = np.asarray(z)

        # Apply plateau below z_min
        z_effective = np.maximum(z_array, self.z_min) if self.z_min > 0 else z_array

        # Power law
        return self.u_ref * (z_effective / self.z_ref)**self.alpha
    
    def get_turbulence_intensity(self, z: Union[float, np.ndarray], h: float = None) -> Union[float, np.ndarray]:
        """Calculate turbulence intensity using power law."""
        z_array = np.asarray(z)

        # Apply plateau below z_min
        z_effective = np.maximum(z_array, self.z_min) if self.z_min > 0 else z_array

        # Power law turbulence: Iv(z) = gamma * (z / 10.0)^(-alpha)
        return self.gamma * (z_effective / 10.0)**(-self.alpha)
    
    def is_constant(self) -> bool:
        return False
    
    def __repr__(self) -> str:
        return (f"PowerLawWindProfile(u_ref={self.u_ref:.2f} m/s, "
                f"z_ref={self.z_ref:.1f} m, alpha={self.alpha:.3f}), "
                f"gamma={self.gamma:.2f}")
    
class TerrainBasedWindProfile(PowerLawWindProfile):
    """
    Power law wind profile using predefined terrain categories.

    Convenient wrapper around PowerLawWindProfile with terrain presets.
    Based on CICIND Commentaries / Eurocode terrain categories.
    Includes both velocity and turbulence profiles.
    """

    # Terrain category definitions
    TERRAIN_CATEGORIES = {
        'I': {
            'description': 'Sea, lakes, or flat horizontal area',
            'z0': 0.01,     # Roughness length
            'z_min': 2.0,   # Minimum height
            'alpha': 0.12,  # Power law exponent
            'gamma': 0.14   # Turbulence factor
        },
        'II': {
            'description': 'Area with low vegetation',
            'z0': 0.05,
            'z_min': 4.0,
            'alpha': 0.16,
            'gamma': 0.19
        },
        'III': {
            'description': 'Area with regular cover of vegetation or buildings',
            'z0': 0.30,
            'z_min': 8.0,
            'alpha': 0.22,
            'gamma': 0.28
        }
    }

    def __init__(self, u_ref: float, z_ref: float = 10.0,
                 terrain_category: str = 'II'):
        """
        Initialize terrain-based wind profile.
        
        Parameters:
        -----------
        u_ref : float
            Reference wind velocity at z_ref [m/s]
        z_ref : float
            Reference height [m]
        terrain_category : str
            Terrain category: 'I', 'II', or 'III'
        """
        if terrain_category not in self.TERRAIN_CATEGORIES:
            raise ValueError(
                f"Unknown terrain category '{terrain_category}'. "
                f"Available: {list(self.TERRAIN_CATEGORIES.keys())}"
            )
        
        self.terrain_category = terrain_category
        self.terrain_data = self.TERRAIN_CATEGORIES[terrain_category]

        # Initialize parent with terrain parameters
        super().__init__(
            u_ref=u_ref,
            z_ref=z_ref,
            alpha=self.terrain_data['alpha'],
            gamma=self.terrain_data['gamma'],
            z_min=self.terrain_data['z_min']
        )

    def __repr__(self) -> str:
        return (f"TerrainBasedWindProfile(terrain={self.terrain_category}, "
                f"u_ref={self.u_ref:.2f} m/s, z_ref={self.z_ref:.1f} m)")
    
class LogarithmicWindProfile(WindProfile):
    """
    Logarithmic wind profile: u(z) = (u_star / kappa) * ln(z / z0)
    
    Based on atmospheric boundary layer theory.
    Easy to add when needed - demonstrates extensibility.
    """
    
    def __init__(self, u_ref: float, z_ref: float = 10.0, 
                 z0: float = 0.05, kappa: float = 0.4):
        """
        Initialize logarithmic wind profile.
        
        Parameters:
        -----------
        u_ref : float
            Reference wind velocity at z_ref [m/s]
        z_ref : float
            Reference height [m]
        z0 : float
            Roughness length [m]
        kappa : float
            von Karman constant (typically 0.4)
        """
        if u_ref <= 0:
            raise ValueError("Reference wind velocity must be positive")
        if z_ref <= 0:
            raise ValueError("Reference height must be positive")
        if z0 <= 0 or z0 >= z_ref:
            raise ValueError("Roughness length must satisfy 0 < z0 < z_ref")
        if kappa <= 0:
            raise ValueError("von Karman constant must be positive")
        
        self.u_ref = u_ref
        self.z_ref = z_ref
        self.z0 = z0
        self.kappa = kappa
        
        # Calculate friction velocity from reference conditions
        self.u_star = (kappa * u_ref) / np.log(z_ref / z0)
    
    def get_velocity(self, z: Union[float, np.ndarray], h: float) -> Union[float, np.ndarray]:
        """Calculate wind velocity using logarithmic law."""
        z_array = np.asarray(z)
        
        # Avoid log(0) - use z0 as minimum
        z_safe = np.maximum(z_array, self.z0 * 1.01)
        
        # Logarithmic law
        return (self.u_star / self.kappa) * np.log(z_safe / self.z0)
    
    def is_constant(self) -> bool:
        return False
    
    def __repr__(self) -> str:
        return (f"LogarithmicWindProfile(u_ref={self.u_ref:.2f} m/s, "
                f"z_ref={self.z_ref:.1f} m, z0={self.z0:.3f} m)")
    
# ============================================================================
# FACTORY FUNCTIONS (for convenience)
# ============================================================================

def create_constant_profile(u_ref: float, Iv: float = 0.15) -> ConstantWindProfile:
    """Create constant wind profile."""
    return ConstantWindProfile(u_ref, Iv)

def create_power_law_profile(u_ref: float, z_ref: float = 10.0,
                            alpha: float = 0.16, gamma: float = 0.19
                            ) -> PowerLawWindProfile:
    """Create power law wind profile with custom parameters."""
    return PowerLawWindProfile(u_ref, z_ref, alpha, gamma)

def create_terrain_profile(u_ref: float, z_ref: float = 10.0,
                           terrain: str = 'II') -> TerrainBasedWindProfile:
    """Create terrain-based wind profile."""
    return TerrainBasedWindProfile(u_ref, z_ref, terrain)

def create_logarithmic_profile(u_ref: float, z_ref: float = 10.0,
                               z0: float = 0.05) -> LogarithmicWindProfile:
    """Create logarithmic wind profile."""
    return LogarithmicWindProfile(u_ref, z_ref, z0)