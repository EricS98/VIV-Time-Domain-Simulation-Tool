# common/wind_model.py
"""
Wind Model - Unified Wind Spectrum and Profile Module
====================================================

This module merges the functionality of wind_spectrum and wind_profile
to provide a unified interface for wind modeling including:
- Terrain categories and wind profiles (power law)
- Turbulence intensity profiles
- Wind velocity spectra (Eurocode)
- Time series generation using spectral methods

Classes:
--------
- TerrainCategory: Terrain parameters for wind modeling
- WindProfile: Wind speed and turbulence calculations  
- WindSpectrum: Wind velocity spectrum calculations
- WindTimeSeriesGenerator: Generate turbulent wind time series
- WindModel: Unified interface combining all functionality
"""

import numpy as np
from typing import Union, Optional, Tuple
from dataclasses import dataclass

@dataclass
class TerrainCategory:
    """Terrain category parameters based on CICIND Commentaries."""
    category: str
    z0: float  # Roughness length [m]
    z_min: float  # Minimum height for the terrain category [m]
    prefactor: float
    alpha: float  # Power law exponent [-]
    turb_factor: float
    description: str
    
    @classmethod
    def get_category(cls, category: str):
        """Get predefined terrain categories."""
        categories = {
            'I': cls('I', 0.01, 2.0, 1.18, 0.12, 0.14, "Sea, lakes or flat and horizontal area."),
            'II': cls('II', 0.05, 4.0, 1.00, 0.16, 0.19, "Area with low vegetation."),
            'III': cls('III', 0.30, 8.0, 0.77, 0.22, 0.28, "Area with regular cover of vegetation or buildings.")
        }
        
        if category not in categories:
            raise ValueError(f"Unknown terrain category: {category}. "
                           f"Available categories: {list(categories.keys())}")
        
        return categories[category]
    
    @classmethod
    def get_all_categories(cls) -> dict:
        """Get all terrain categories as a dictionary."""
        return {cat: cls.get_category(cat) for cat in ['I', 'II', 'III']}


class WindProfile:
    """
    Wind profile calculations using power law model.
    Handles both mean wind speed and turbulence intensity profiles.
    """

    def __init__(self, u_ref: float, terrain_category: Union[str, TerrainCategory],
                 z_ref: float = 10.0, profile_type: str = "power_law",
                 Iv_constant: Optional[float] = None):
        """
        Initialize the wind profile calculator.

        Parameters:
        -----------
        u_ref: float
            Reference wind speed at height z_ref [m/s]
        terrain_category: str or TerrainCategory
            Terrain category (I, II, III) or TerrainCategory object
        z_ref: float
            Reference height [m]
        """
        if u_ref <= 0:
            raise ValueError("Reference wind speed u_ref must be positive.")
        if Iv_constant is not None and not (0 <= Iv_constant <= 1):
            raise ValueError("Turbulence intensity Iv_ref must be in [0, 1].")
        if z_ref <= 0:
            raise ValueError("Reference height z_ref must be positive.")
        
        if isinstance(terrain_category, str):
            self.terrain = TerrainCategory.get_category(terrain_category)
        else:
            self.terrain = terrain_category
        
        self.u_ref = u_ref
        self.z_ref = z_ref
        self.profile_type = profile_type
        self.Iv_constant = Iv_constant
        
        # Validate z_ref meets minimum height requirement
        if z_ref < self.terrain.z_min:
            raise ValueError(f"Reference height z_ref ({z_ref}m) is below minimum "
                           f"height for terrain category {self.terrain.category} ({self.terrain.z_min}m)")
        
        if profile_type == "constant" and Iv_constant is None:
            raise ValueError("Iv_constant must be provided for constant profiles")
    
    def wind_speed(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate wind speed at height z using power law.
        
        U(z) = prefactor * u_ref * (z / z_ref) ** alpha

        Parameters:
        -----------
        z: float or np.ndarray
            Height(s) at which to calculate wind speed [m]

        Returns:
        --------
        float or np.ndarray: Wind speed at height z [m/s]
        """
        if self.profile_type == "constant":
            z = np.asarray(z)
            return np.full_like(z, self.u_ref, dtype=float)
        else:
            z = np.asarray(z)
            if np.any(z <= 0):
                raise ValueError("Height z must be positive.")
            # clamp to z_min (plateau)
            z_eff = np.maximum(z, self.terrain.z_min)
            return self.terrain.prefactor * self.u_ref * (z_eff / self.z_ref) ** self.terrain.alpha
    
    def turbulence_intensity(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate turbulence intensity using CICIND model.

        Parameters:
        -----------
        z: float or np.ndarray
            Height(s) at which to calculate turbulence intensity [m]

        Returns:
        --------
        float or np.ndarray: Turbulence intensity at height z [-]
        """
        if self.profile_type == "constant":
            z = np.asarray(z)
            return np.full_like(z, self.Iv_constant, dtype=float)
        else:
            z = np.asarray(z)
            if np.any(z <= 0):
                raise ValueError("Height z must be positive.")
            # clamp to z_min (plateau) and use the same boundary as wind_speed
            z_eff = np.maximum(z, self.terrain.z_min)
            return self.terrain.turb_factor * (z_eff / 10.0) ** (-self.terrain.alpha)
    
    def generate_profile(self, z_min: Optional[float] = None, z_max: float = 300.0,
                         n_points: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate wind profile over a height range.

        Parameters:
        -----------
        z_min: float, optional
            Minimum height [m]. If None, uses terrain category minimum height.
        z_max: float
            Maximum height [m]
        n_points: int
            Number of points in the profile

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 
            Heights (m), wind speeds (m/s), turbulence intensities (-)
        """
        if z_min is None:
            z_min = self.terrain.z_min
        if z_min <= 0 or z_max <= z_min:
            raise ValueError("Invalid height range for wind profile generation.")
        
        heights = np.linspace(z_min, z_max, n_points)
        wind_speeds = self.wind_speed(heights)
        turb_intensities = self.turbulence_intensity(heights)
        
        return heights, wind_speeds, turb_intensities


class WindSpectrum:
    """
    Wind velocity spectrum calculations using Eurocode model.
    """
    
    def __init__(self, terrain_category: Union[str, TerrainCategory]):
        """Initialize wind spectrum calculator."""
        if isinstance(terrain_category, str):
            self.terrain = TerrainCategory.get_category(terrain_category)
        else:
            self.terrain = terrain_category

        # Eurocode parameters
        self.z_t = 200.0    # Reference height [m]
        self.L_t = 300.0    # Reference integral length scale [m]

        # Calculate terrain-dependent parameter
        self.alpha = 0.67 + 0.05 * np.log(self.terrain.z0)

    def integral_length_scale(self, z: float) -> float:
        """
        Calculate integral length scale L(z).

        Parameters:
        -----------
        z: float
            Height above ground [m]

        Returns:
        --------
        float: Integral length scale L(z) [m]
        """
        if z < self.terrain.z_min:
            z_calc = self.terrain.z_min
        else: 
            z_calc = z

        return self.L_t * (z_calc / self.z_t)**self.alpha
    
    def dimensionless_frequency(self, f: np.ndarray, z: float, U: float) -> np.ndarray:
        """
        Calculate dimensionless frequency f_L(z,f).

        Parameters:
        -----------
        f: np.ndarray
            Frequency array [Hz]
        z: float
            Height above ground [m]
        U: float
            Mean wind velocity at height z [m/s]

        Returns:
        --------
        np.ndarray: Dimensionless frequency f_L(z,f) [-]
        """
        if U <= 0:
            raise ValueError("Mean wind speed U must be positive.")
        L_z = self.integral_length_scale(z)

        return f * L_z / U
    
    def normalized_spectrum(self, f: np.ndarray, z: float, U: float) -> np.ndarray:
        """
        Calculate normalized wind velocity spectrum S(z,f).

        Parameters:
        -----------
        f: np.ndarray
            Frequency array [Hz]
        z: float
            Height above ground [m]
        U: float
            Mean wind velocity at height z [m/s]
            
        Returns:
        --------
        np.ndarray: Normalized spectrum S(z,f) [-]
        """
        f_L = self.dimensionless_frequency(f, z, U)
        return 6.8 * f_L / (1 + 10.2 * f_L)**(5/3)
    
    def power_spectral_density(self, f: np.ndarray, z: float, U: float,
                               sigma_u: float) -> np.ndarray:
        """
        Calculate power spectral density S_u(z,f).

        Parameters:
        -----------
        f: np.ndarray
            Frequency array [Hz]
        z: float
            Height above ground [m]
        U: float
            Mean wind velocity at height z [m/s]
        sigma_u: float
            Standard deviation of wind fluctuations [m/s]
            
        Returns:
        --------
        np.ndarray: Power spectral density S_u(z,f) [m²/s²/Hz]
        """
        S_normalized = self.normalized_spectrum(f, z, U)
        S = S_normalized * sigma_u**2 / np.maximum(f, 1e-12)
        S[0] = 0.0
        return S


class WindTimeSeriesGenerator:
    """Generator for turbulent wind time series using spectral methods."""

    def __init__(self, spectrum_calculator: WindSpectrum):
        """Initialize with a wind spectrum calculator."""
        self.spectrum = spectrum_calculator

    def generate_wind_series(self, t: np.ndarray, z: float, U_mean: float, 
                            Iv: float = 0.15, random_seed: int = None) -> np.ndarray:
        """
        Generate wind time series using IFFT synthesis.
        
        Parameters:
        -----------
        t: np.ndarray
            Time array [s]
        z: float
            Height above ground [m]
        U_mean: float
            Mean wind velocity [m/s]
        Iv: float
            Turbulence intensity [-]
        random_seed: int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        np.ndarray: Wind velocity time series [m/s]
        """
        # Time/frequency grids
        dt = t[1] - t[0]
        N = len(t)
        df = 1.0 / (N * dt)
        f = np.arange(0, N // 2 + 1) * df

        # Calculate standard deviation of the wind velocity
        sigma_u = Iv * U_mean

        # Calculate full wind spectrum
        S_u = self.spectrum.power_spectral_density(f, z, U_mean, sigma_u)

        # Generate fluctuating component
        from common.time_domain_base import TimeSeriesSynthesizer
        u_fluct = TimeSeriesSynthesizer.synthesize_from_psd(S_u, df, N, random_seed)

        return U_mean + u_fluct


class WindModel:
    """
    Unified wind model combining profile and spectrum functionality.
    
    This class provides a single interface to all wind modeling capabilities:
    - Wind profiles and turbulence intensity 
    - Wind velocity spectra
    - Time series generation
    """
    
    def __init__(self, terrain_category: Union[str, TerrainCategory] = 'II'):
        """
        Initialize the unified wind model.
        
        Parameters:
        -----------
        terrain_category: str or TerrainCategory
            Terrain category (I, II, III) or TerrainCategory object
        """
        if isinstance(terrain_category, str):
            self.terrain = TerrainCategory.get_category(terrain_category)
        else:
            self.terrain = terrain_category
            
        # Initialize spectrum calculator
        self.spectrum = WindSpectrum(self.terrain)
        
        # Wind profile will be created when needed
        self._profile = None
        
    def setup_profile(self, u_ref: float, z_ref: float = 10.0,
                      profile_type: str = "power_law", Iv_constant: Optional[float] = None) -> None:
        """
        Setup wind profile calculator.
        
        Parameters:
        -----------
        u_ref: float
            Reference wind speed at height z_ref [m/s]
        z_ref: float
            Reference height [m]
        profile_type: str
            "power_law" or "constant"
        """
        self._profile = WindProfile(
            u_ref=u_ref,
            terrain_category=self.terrain,
            z_ref=z_ref,
            profile_type=profile_type,
            Iv_constant=Iv_constant
        )
    
    def wind_speed(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Calculate wind speed at height z."""
        if self._profile is None:
            raise ValueError("Wind profile not setup. Call setup_profile() first.")
        return self._profile.wind_speed(z)
    
    def turbulence_intensity(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Calculate turbulence intensity at height z."""
        if self._profile is None:
            raise ValueError("Wind profile not setup. Call setup_profile() first.")
        return self._profile.turbulence_intensity(z)
    
    def generate_profile(self, z_min: Optional[float] = None, z_max: float = 300.0,
                        n_points: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate complete wind profile."""
        if self._profile is None:
            raise ValueError("Wind profile not setup. Call setup_profile() first.")
        return self._profile.generate_profile(z_min, z_max, n_points)
    
    def power_spectral_density(self, f: np.ndarray, z: float, U: float,
                              sigma_u: float) -> np.ndarray:
        """Calculate wind velocity power spectral density."""
        return self.spectrum.power_spectral_density(f, z, U, sigma_u)
    
    def integral_length_scale(self, z: float) -> float:
        """Calculate integral length scale at height z."""
        return self.spectrum.integral_length_scale(z)
    
    def generate_time_series(self, t: np.ndarray, z: float, U_mean: float,
                           Iv: float = 0.15, random_seed: int = None) -> np.ndarray:
        """
        Generate wind velocity time series.
        
        Parameters:
        -----------
        t: np.ndarray
            Time array [s]
        z: float
            Height above ground [m]
        U_mean: float
            Mean wind velocity [m/s]
        Iv: float
            Turbulence intensity [-]
        random_seed: int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        np.ndarray: Wind velocity time series [m/s]
        """
        generator = WindTimeSeriesGenerator(self.spectrum)
        return generator.generate_wind_series(t, z, U_mean, Iv, random_seed)
    
    def get_terrain_info(self) -> dict:
        """Get terrain category information."""
        return {
            'category': self.terrain.category,
            'description': self.terrain.description,
            'roughness_length': self.terrain.z0,
            'min_height': self.terrain.z_min,
            'power_law_exponent': self.terrain.alpha,
            'turbulence_factor': self.terrain.turb_factor
        }