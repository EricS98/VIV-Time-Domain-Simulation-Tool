# calculators/lift_spectrum.py
"""
Lift Spectrum Calculator Module
===============================

Calculates the sectional lift coefficient spectrum and 
sectional lift force spectrum based on the Vickery & Basu spectral model.

Formulas:
---------
Sectional lift coefficient spectrum:
    S_CL(f,z) = (σ_CL²(z)) / (√π * B(z) * f_s(z)) * exp(-(1 - f/f_s(z))² / B(z)²)

Sectional lift force spectrum:
    S_L(f,z) = [0.5 * ρ * U²(z) * d(z)]² * S_CL(f,z)

where:
    - σ_CL(z): RMS lift coefficient at height z
    - B(z): Bandwidth parameter at height z
    - f_s(z): Shedding frequency at height z = St * U(z) / d(z)
    - ρ: Air density
    - U(z): Wind velocity at height z
    - d(z): Diameter at height z

References:
- E. Simon, "Development and Application of a Time-Domain Simulation Tool for Spectral Modeling of Vortex-Induced Vibrations", Master's Thesis, RWTH Aachen, 2025.
"""
import numpy as np
from typing import Union, Optional, Tuple
from dataclasses import dataclass

from common.structures import StructureProperties
from config import VIVAnalysisConfig
from calculators.base_calculator import BaseCalculator

@dataclass
class LiftSpectrumResults:
    """Results from lift spectrum calculation."""

    # Height coordinate
    z: Union[float, np.ndarray]

    # Frequency array
    frequency: np.ndarray

    # Lift coefficient spectrum S_CL(f,z) [1/Hz]
    S_CL: np.ndarray

    # Lift force spectrum S_L(f,z) [N²·s/m²]
    S_L: np.ndarray

    # Parameters used
    sigma_CL: Union[float, np.ndarray]  # RMS lift coefficient
    B: Union[float, np.ndarray]          # Bandwidth parameter
    f_s: Union[float, np.ndarray]        # Shedding frequency [Hz]
    U: Union[float, np.ndarray]          # Wind velocity [m/s]
    d: Union[float, np.ndarray]          # Diameter [m]

class LiftSpectrumCalculator(BaseCalculator):
    """
    Calculator for sectional lift coefficient and lift force spectra.

    Supports both constant and height-varying properties.
    """
    
    def __init__(self, St: float = 0.2):
        """
        Initialize lift spectrum calculator.

        Parameters:
        -----------
        St : float
            Strouhal number (default: 0.2)
        """
        super().__init__(St=St)

    def calculate_spectrum(
        self,
        structure: StructureProperties,
        config: VIVAnalysisConfig,
        frequency: np.ndarray,
        z: Optional[Union[float, np.ndarray]] = None
    ) -> LiftSpectrumResults:
        """
        Calculate lift coefficient and lift force spectra.

        Parameters:
        -----------
        structure : StructureProperties
            Structure properties
        config : VIVAnalysisConfig
            Analysis configuration
        frequency : np.ndarray
            Frequency array [Hz]
        z : float or np.ndarray, optional
            Height(s) at which to evaluate spectrum [m].
            If None, uses structure top height.
            
        Returns:
        --------
        LiftSpectrumResults
            Complete spectrum results
        """
        # Determine height(s)
        if z is None:
            z = structure.height

        z_array = np.atleast_1d(z)
        is_single_height = np.isscalar(z)

        # Check if properties are constant
        is_constant = (config.cross_section.is_constant() and 
                       config.wind_profile.is_constant())
        
        if is_constant and is_single_height:
            # Use analytical approach for constant properties at single height
            results = self._calculate_constant(structure, config, frequency, z_array[0])
        else:
            # Use numerical approach for height-varying properties
            results = self._calculate_varying(structure, config, frequency, z_array)

        return results
    
    def _calculate_constant(
        self,
        structure: StructureProperties,
        config: VIVAnalysisConfig, 
        frequency: np.ndarray,
        z: float
    ) -> LiftSpectrumResults:
        """
        Calculate spectra for constant properties (analytical).

        Parameters:
        -----------
        structure : StructureProperties
            Structure properties
        config : VIVAnalysisConfig
            Analysis configuration
        frequency : np.ndarray
            Frequency array [Hz]
        z : float
            Height at which to evaluate [m]
            
        Returns:
        --------
        LiftSpectrumResults
            Spectrum results
        """
        h = structure.height

        # Get constant properties
        d = config.cross_section.get_diameter(z, h, structure.diameter)
        U = config.wind_profile.get_velocity(z, h)
        Iv = config.wind_profile.get_turbulence_intensity(z, h)

        # Calculate Reynolds number
        # Simon (2025), Eq. 2.8
        Re = U * d / self.nu_air
        
        # Get aerodynamic parameters
        sigma_CL = config.lift_coefficient.get_sigma_cl(Re)

        # Bandwidth parameter (turbulence-dependent)
        # Simon (2025), Eq. 3.5
        B = min(0.1 + Iv, 0.35)
        
        # Shedding frequency
        # Simon (2025), Eq. 2.7b
        f_s = self.St * U / d
        
        # Calculate lift coefficient spectrum
        S_CL = self._calculate_S_CL(frequency, sigma_CL, B, f_s)
        
        # Calculate lift force spectrum
        # Simon (2025), Eq. 3.8c
        force_scaling = (0.5 * self.rho_air * U**2 * d)**2
        S_L = force_scaling * S_CL
        
        return LiftSpectrumResults(
            z=z,
            frequency=frequency,
            S_CL=S_CL,
            S_L=S_L,
            sigma_CL=sigma_CL,
            B=B,
            f_s=f_s,
            U=U,
            d=d,
        )
    
    def _calculate_varying(
        self,
        structure: StructureProperties,
        config: VIVAnalysisConfig,
        frequency: np.ndarray,
        z_array: np.ndarray
    ) -> LiftSpectrumResults:
        """
        Calculate spectra for height-varying properties.
        
        Parameters:
        -----------
        structure : StructureProperties
            Structure properties
        config : VIVAnalysisConfig
            Analysis configuration
        frequency : np.ndarray
            Frequency array [Hz]
        z_array : np.ndarray
            Heights at which to evaluate [m]
            
        Returns:
        --------
        LiftSpectrumResults
            Spectrum results (arrays will have shape [n_heights, n_frequencies])
        """
        h = structure.height
        d_top = structure.diameter
        n_z = len(z_array)
        n_f = len(frequency)

        # Initialize arrays
        S_CL = np.zeros((n_z, n_f))
        S_L = np.zeros((n_z, n_f))

        # Get height-varying properties
        d_z = config.cross_section.get_diameter(z_array, h, d_top)
        U_z = config.wind_profile.get_velocity(z_array, h)

        # Calculate turbulence intensity at each height
        Iv_z = np.array([config.wind_profile.get_turbulence_intensity(z, h)
                         for z in z_array])
        
        # Calculate Reynolds number at each height
        # Simon (2025), Eq. 2.8
        Re_z = U_z * d_z / self.nu_air

        # Get aerodynamic parameters at each height
        sigma_CL_z = np.array([config.lift_coefficient.get_sigma_cl(Re)
                               for Re in Re_z])
        
        # Bandwidth parameter at each height
        # Simon (2025), Eq. 3.5
        B_z = np.minimum(0.1 + Iv_z, 0.35)

        # Shedding frequency at each height
        # Simon (2025), Eq. 2.7b
        f_s_z = self.St * U_z / d_z

        # Calculate spectra at each height
        for i in range(n_z):
            # Lift coefficient spectrum
            S_CL[i,:] = self._calculate_S_CL(
                frequency, sigma_CL_z[i], B_z[i], f_s_z[i]
            )

            # Lift force spectrum
            # Simon (2025), Eq. 3.8c
            force_scaling = (0.5 * self.rho_air * U_z[i]**2 * d_z[i])**2
            S_L[i,:] = force_scaling * S_CL[i,:]

        return LiftSpectrumResults(
            z=z_array,
            frequency=frequency,
            S_CL=S_CL,
            S_L=S_L,
            sigma_CL=sigma_CL_z,
            B=B_z,
            f_s=f_s_z,
            U=U_z,
            d=d_z,
        )
    
    def _calculate_S_CL(
        self,
        frequency: np.ndarray,
        sigma_CL: float,
        B: float,
        f_s: float
    ) -> np.ndarray:
        """
        Calculate lift coefficient spectrum using Gaussian spectral model.

        Formula:
            S_CL(f) = (σ_CL²) / (√π * B * f_s) * exp(-((1 - f/f_s) / B)²)
        
        Parameters:
        -----------
        frequency : np.ndarray
            Frequency array [Hz]
        sigma_CL : float
            RMS lift coefficient
        B : float
            Bandwidth parameter
        f_s : float
            Shedding frequency [Hz]
            
        Returns:
        --------
        np.ndarray
            Lift coefficient spectrum [1/Hz]
        """
        # ====================================
        # Simon (2025), Eq. 3.3
        # ====================================

        # Normalization factor
        normalization = sigma_CL**2 / (np.sqrt(np.pi) * B * f_s)

        # Gaussian shape centered at shedding frequency
        frequency_deviation = (1.0 - frequency / f_s) / B
        exponential_term = np.exp(-frequency_deviation**2)

        S_CL = normalization * exponential_term

        return S_CL
    
    def calculate_at_natural_frequency(
        self,
        structure: StructureProperties,
        config: VIVAnalysisConfig,
        z: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Calculate spectrum values at the natural frequency.

        Parameters:
        -----------
        structure : StructureProperties
            Structure properties
        config : VIVAnalysisConfig
            Analysis configuration
        z : float, optional
            Height at which to evaluate [m].
            If None, uses structure top height.
            
        Returns:
        --------
        Tuple[float, float]
            (S_CL at f_n, S_L at f_n)
        """
        if z is None:
            z = structure.height

        f_n = structure.f_n
 
        results = self.calculate_spectrum(structure, config, np.array([f_n]), z)

        S_CL_at_fn = results.S_CL[0] if results.S_CL.ndim > 0 else results.S_CL
        S_L_at_fn = results.S_L[0] if results.S_L.ndim > 0 else results.S_L
        
        return S_CL_at_fn, S_L_at_fn