# calculators/generalized_force_spectrum.py
"""
Generalized Force Spectrum Calculator
======================================

Calculates the generalized force spectrum S_Qn(f) using the spectral approach.

Mathematical formulation:
-------------------------
Full double integral with coherence:
    S_Qn(f) = ∫₀ʰ ∫₀ʰ g_n(f,z₁) g_n(f,z₂) Coh(f; z₁, z₂) dz₁ dz₂

Where:
    g_n(f,z) = √S_L(f,z) · Φ_n(z)
    S_L(f,z) = [0.5 ρ U²(z) d(z)]² · S_CL(f,z)

Simplified for constant correlation length λ:
    S_Qn(f) = 2λd ∫₀ʰ S_L(f,z) Φ_n²(z) dz
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

from calculators.base_calculator import BaseCalculator
from calculators.lift_spectrum import LiftSpectrumCalculator
from common.structures import StructureProperties
from common.time_domain_base import TimeGridBuilder, SimulationGrid
from config.analysis_config import VIVAnalysisConfig

@dataclass
class GeneralizedForceSpectrum:
    """Results from generalized force spectrum calculation."""

    # Frequency data
    frequency: np.ndarray       # Frequency array [Hz]
    spectrum: np.ndarray        # S_Qn(f) [N²·s]

    # Key values
    natural_frequency: float    # Natural frequency [Hz]
    spectrum_at_fn: float       # S_Qn(f_n) [N²·s]

    # Statistics
    modal_variance: float       # ∫S_Qn df [N²]
    modal_rms: float            # √(modal_variance) [N]

    # Parameters used
    sigma_cl: float                      # RMS lift coefficient
    bandwidth: float                     # Bandwidth parameter B
    shedding_frequency: float            # Vortex shedding frequency [Hz]
    critical_velocity: float             # Critical velocity [m/s]
    correlation_length: float            # Correlation length [m]
    mode_shape_integral: float           # ∫Φ²dz

    # Computation info
    method: str = "analytical"           # 'analytical' or 'numerical'
    coherence_method: str = "constant"   # Coherence function type used

class GeneralizedForceSpectrumCalculator(BaseCalculator):
    """
    Calculator for generalized force spectrum S_Qn(f).
    
    Uses the LiftSpectrumCalculator for sectional spectra and integrates
    over height with proper coherence functions.
    """
    def __init__(self, St: float = 0.2):
        """
        Initialize calculator.
        
        Parameters:
        -----------
        St : float
            Strouhal number (default: 0.2)
        """
        super().__init__(St=St)
        self.lift_calculator = LiftSpectrumCalculator(St=St)
        self.grid_builder = TimeGridBuilder()

    def calculate_spectrum(
        self, 
        structure: StructureProperties,
        config: VIVAnalysisConfig,
        frequency: Optional[np.ndarray] = None,
        grid: Optional[SimulationGrid] = None
    ) -> GeneralizedForceSpectrum:
        """
        Calculate generalized force spectrum S_Qn(f).

        Automatically selects between:
        - Analytical (single integral) for constant correlation length
        - Numerical (double integral) for full coherence functions
        
        Parameters:
        -----------
        structure : StructureProperties
            Structure properties
        config : VIVAnalysisConfig
            Analysis configuration
        frequency : np.ndarray, optional
            Frequency array [Hz]. If None, auto-generated.
        grid : SimulationGrid, optional
            Pre-computed simulation grid. If provided, uses grid.f.
            
        Returns:
        --------
        GeneralizedForceSpectrum
            Complete spectrum results
        """
        h = structure.height
        f_n = structure.f_n

        # Step 1: Determine frequency array
        if grid is not None:
            frequency = grid.f
            print(f"\nUsing frequency grid from SimulationGrid:")
            print(f"  df = {grid.df:.4f} Hz, f_max = {grid.f[-1]:.2f} Hz")
        elif frequency is None:
            raise ValueError(
                "Either 'frequency' array or 'grid' must be provided. "
                "Use create_grid() to generate an appropriate grid."
            )

        # Step 2: Check coherence function type
        use_simple_coherence = config.coherence_function.is_constant_correlation_length()

        print(f"\n{'='*70}")
        print(f"GENERALIZED FORCE SPECTRUM CALCULATION")
        print(f"{'='*70}")
        print(f"Structure: {structure.name}")
        print(f"  Natural frequency: {f_n:.4f} Hz")
        print(f"  Coherence method: {config.coherence_function.get_name()}")

        # Step 3: Calculate spectrum
        if use_simple_coherence:
            # Fast path: single integral with constant correlation length
            result = self._calculate_single_integral(
                structure, config, frequency
            )
        else:
            # Full path: double integral with coherence function
            result = self._calculate_double_integral(
                structure, config, frequency
            )

        return result

    def _calculate_single_integral(
        self, 
        structure: StructureProperties,
        config: VIVAnalysisConfig,
        frequency: np.ndarray
    ) -> GeneralizedForceSpectrum:
        """
        Calculate using single integral (constant correlation length).

        Formula: S_Qn(f) = 2λd ∫₀ʰ S_L(f,z) Φ_n²(z) dz
        """
        print("\nUsing SINGLE INTEGRAL (constant correlation length)")

        h = structure.height
        f_n = structure.f_n

        # Get correlation length
        lambda_corr = config.coherence_function.lambda_factor

        # Check if properties are constant
        is_constant = (config.cross_section.is_constant() and
                       config.wind_profile.is_constant())
        
        if is_constant:
            # Analytical solution
            print("  Properties are constant → using analytical formula")
            S_Qn, params = self._analytical_constant_properties(
                structure, config, frequency, lambda_corr
            )
            method = "analytical"
        else:
            # Numerical integration over height
            print("  Properties vary with height → using numerical integration")
            S_Qn, params = self._numerical_single_integral(
                structure, config, frequency, lambda_corr
            )
            method = "numerical"

        # Calculate statistics
        df = frequency[1] - frequency[0] if len(frequency) > 1 else 0.01
        modal_variance = np.trapezoid(S_Qn, dx=df)
        modal_rms = np.sqrt(max(modal_variance, 0))

        # Find spectrum at natural frequency
        fn_idx = np.argmin(np.abs(frequency - f_n))
        spectrum_at_fn = S_Qn[fn_idx]

        # Mode shape integral
        mode_shape_integral = config.mode_shape.compute_modal_integral(h, mode_number=1)

        print(f"\n{'='*70}")
        print(f"RESULTS:")
        print(f"  Modal variance: {modal_variance:.3e} N²")
        print(f"  Modal RMS: {modal_rms:.3e} N")
        print(f"  S_Qn(f_n): {spectrum_at_fn:.3e} N²·s")
        print(f"{'='*70}")
        
        return GeneralizedForceSpectrum(
            frequency=frequency,
            spectrum=S_Qn,
            natural_frequency=f_n,
            spectrum_at_fn=spectrum_at_fn,
            modal_variance=modal_variance,
            modal_rms=modal_rms,
            sigma_cl=params['sigma_cl'],
            bandwidth=params['bandwidth'],
            shedding_frequency=params['shedding_frequency'],
            critical_velocity=params['critical_velocity'],
            correlation_length=lambda_corr,
            mode_shape_integral=mode_shape_integral,
            method=method,
            coherence_method="constant_correlation_length"
        )
    
    def _calculate_double_integral(
        self, 
        structure: StructureProperties,
        config: VIVAnalysisConfig,
        frequency: np.ndarray
    ) -> GeneralizedForceSpectrum:
        """
        Calculate using double integral with coherence function.
        
        Formula: S_Qn(f) = ∫₀ʰ ∫₀ʰ g_n(f,z₁) g_n(f,z₂) Coh(z₁,z₂) dz₁ dz₂
        """
        print("\nUsing DOUBLE INTEGRAL (full coherence function)")
        
        h = structure.height
        f_n = structure.f_n
        n_points = 100  # Height discretization

        # Create height grid
        z_points = np.linspace(0, h, n_points)
        dz = z_points[1] - z_points[0]

        # Get properties at each height
        d_z = config.cross_section.get_diameter(z_points, h, structure.diameter)
        phi_z = config.mode_shape.evaluate(z_points, h, mode_number=1)

        # Calculate lift force spectrum at all heights
        print(f"  Calculating S_L(f,z) at {n_points} heights...")
        lift_results = self.lift_calculator.calculate_spectrum(
            structure, config, frequency, z_points
        )

        # S_L should have shape [n_heights, n_frequencies]
        S_L_matrix = lift_results.S_L

        # Handle case where constant properties return 1D array
        if S_L_matrix.ndim == 1:
            # For constant properties, spectrum is same at all heights
            # Replicate to create [n_heights, n_frequencies] array
            S_L_matrix = np.tile(S_L_matrix, (n_points, 1))

        # Get parameters for reporting
        u_ref = config.wind_profile.get_velocity(h, h)
        d_ref = config.cross_section.get_diameter(h, h, structure.diameter)
        v_crit = f_n * d_ref / self.St
        Re = v_crit * d_ref / self.nu_air
        sigma_cl = config.lift_coefficient.get_sigma_cl(Re)
        Iv_ref = config.wind_profile.get_turbulence_intensity(h)
        B = min(0.1 + Iv_ref, 0.35)
        f_s = self.St * v_crit / d_ref

        # Initialize generalized spectrum
        n_freq = len(frequency)
        S_Qn = np.zeros(n_freq)

        # Check if coherence is frequency-independent for optimization
        is_freq_independent = config.coherence_function.is_frequency_independent()

        if is_freq_independent:
            # Pre-calculate coherence matrix (only depends on heights, not frequency)
            coh_matrix = np.zeros((n_points, n_points))
            for i in range(n_points):
                for j in range(i, n_points):
                    z1, z2 = z_points[i], z_points[j]
                    d1, d2 = d_z[i], d_z[j]
                    
                    # Calculate coherence (frequency parameter doesn't matter)
                    coh = config.coherence_function.get_coherence(
                        z1, z2, 0.0, d1, d2  # f=0.0 is arbitrary since it's not used
                    )
                    
                    # Store in both positions due to symmetry
                    coh_matrix[i, j] = coh
                    coh_matrix[j, i] = coh

            print(f"  Computing double integral for {n_freq} frequencies...")

            # Now loop over frequencies with pre-calculated coherence
            for k in range(n_freq):
                # g_n(f,z) = √S_L(f,z) · Φ_n(z)
                g_n = np.sqrt(S_L_matrix[:, k]) * phi_z
                
                # Use vectorized operations with pre-calculated coherence
                # S_Qn(f) = ∫∫ g_n(i) * g_n(j) * coh(i,j) dz dz
                # With discretization: sum over all i,j
                g_outer = np.outer(g_n, g_n)  # g_n(i) * g_n(j) for all i,j
                integral = np.sum(g_outer * coh_matrix) * dz * dz
                
                S_Qn[k] = integral

        else:

            # For each frequency, compute double integral
            for k in range(n_freq):
                # g_n(f,z) = √S_L(f,z) · Φ_n(z)
                g_n = np.sqrt(S_L_matrix[:, k]) * phi_z
            
                integral = 0.0

                # Double integration with symmetry
                for i in range(n_points):
                    for j in range(i, n_points):    # Only upper triangle
                        z1, z2 = z_points[i], z_points[j]
                        d1, d2 = d_z[i], d_z[j]

                        # Get coherence
                        coh = config.coherence_function.get_coherence(
                            z1, z2, frequency[k], d1, d2
                        )

                        # Product of g_n values
                        g_product = g_n[i] * g_n[j]

                        # Add to integral (factor of 2 for off-diagonal due to symmetry)
                        weight = 2.0 if i != j else 1.0
                        integral += weight * g_product * coh * dz * dz

                S_Qn[k] = integral

        # Calculate statistics
        df = frequency[1] - frequency[0] if len(frequency) > 1 else 0.01
        modal_variance = np.trapezoid(S_Qn, dx=df)
        modal_rms = np.sqrt(max(modal_variance, 0))

        # Find spectrum at natural frequency
        fn_idx = np.argmin(np.abs(frequency - f_n))
        spectrum_at_fn = S_Qn[fn_idx]
        
        # Mode shape integral
        mode_shape_integral = config.mode_shape.compute_modal_integral(h, mode_number=1)
        
        print(f"\n{'='*70}")
        print(f"RESULTS:")
        print(f"  Modal variance: {modal_variance:.3e} N²")
        print(f"  Modal RMS: {modal_rms:.3e} N")
        print(f"  S_Qn(f_n): {spectrum_at_fn:.3e} N²·s")
        print(f"{'='*70}")

        return GeneralizedForceSpectrum(
            frequency=frequency,
            spectrum=S_Qn,
            natural_frequency=f_n,
            spectrum_at_fn=spectrum_at_fn,
            modal_variance=modal_variance,
            modal_rms=modal_rms,
            sigma_cl=sigma_cl,
            bandwidth=B,
            shedding_frequency=f_s,
            critical_velocity=v_crit,
            correlation_length=float("nan"),
            mode_shape_integral=mode_shape_integral,
            method="numerical",
            coherence_method=config.coherence_function.get_name()
        )
    
    def _analytical_constant_properties(
        self,
        structure: StructureProperties,
        config: VIVAnalysisConfig,
        frequency: np.ndarray,
        lambda_corr: float
    ) -> Tuple[np.ndarray, dict]:
        """
        Analytical solution for constant properties.

        Formula: S_Qn(f) = 2λd (0.5ρU²d)² S_CL(f) ∫Φ²dz
        """
        h = structure.height
        f_n = structure.f_n

        # Calculate lift force spectrum single height
        lift_results = self.lift_calculator.calculate_spectrum(
            structure, config, frequency, z=h
        )

        # Lift force spectrum
        S_L = lift_results.S_L

        # Mode shape integral
        mode_shape_integral = config.mode_shape.compute_modal_integral(h, mode_number=1)

        # Generalized force spectrum
        d = structure.diameter
        S_Qn = 2.0 * lambda_corr * d * S_L * mode_shape_integral

        params = {
            'sigma_cl': lift_results.sigma_CL,
            'bandwidth': lift_results.B,
            'shedding_frequency': lift_results.f_s,
            'critical_velocity': f_n * d / self.St
        }
        
        return S_Qn, params
    
    def _numerical_single_integral(
        self,
        structure: StructureProperties,
        config: VIVAnalysisConfig,
        frequency: np.ndarray,
        lambda_corr: float
    ) -> Tuple[np.ndarray, float]:
        """
        Numerical integration for height-varying properties.
        
        Formula: S_Qn(f) = 2λd ∫₀ʰ S_L(f,z) Φ_n²(z) dz
        """
        h = structure.height
        n_points = 100
        
        # Create height grid
        z_points = np.linspace(0, h, n_points)
        
        # Get mode shape at each height
        phi_z = config.mode_shape.evaluate(z_points, h, mode_number=1)

        # Calculate lift force spectrum at all heights
        lift_results = self.lift_calculator.calculate_spectrum(
            structure, config, frequency, z_points
        )

        # S_L has shape [n_heights, n_frequencies]
        S_L_matrix = lift_results.S_L

        # For each frequency, integrate over height
        S_Qn = np.zeros(len(frequency))
        for i, f in enumerate(frequency):
            integrand = S_L_matrix[:, i] * phi_z**2
            S_Qn[i] = 2.0 * lambda_corr * lift_results.d[-1] * np.trapezoid(integrand, z_points)

        # Get parameters for reporting (uses values at top)
        f_n = structure.f_n
        d_ref = lift_results.d[-1]
        U_ref = lift_results.U[-1]
        v_crit = f_n * d_ref / self.St

        params = {
            'sigma_cl': lift_results.sigma_CL[-1] if hasattr(lift_results.sigma_CL, '__len__') else lift_results.sigma_CL,
            'bandwidth': lift_results.B[-1] if hasattr(lift_results.B, '__len__') else lift_results.B,
            'shedding_frequency': lift_results.f_s[-1] if hasattr(lift_results.f_s, '__len__') else lift_results.f_s,
            'critical_velocity': v_crit
        }

        return S_Qn, params
    
    def create_grid(
        self,
        structure: StructureProperties,
        config: VIVAnalysisConfig,
        duration: float = 600.0,
        dt: Optional[float] = None,
        min_natural_periods: int = 1000,
    ) -> SimulationGrid:
        """
        Create simulation grid suitable for spectrum calculation.
        
        Parameters:
        -----------
        structure : StructureProperties
            Structure properties
        duration : float
            Simulation duration [s] (default: 600)
        dt : float, optional
            Time step [s]. If None, auto-calculated.
        min_natural_periods : int
            Minimum number of natural periods (default: 1000)
        f_max : float
            Maximum frequency [Hz] (default: 10.0)
            
        Returns:
        --------
        SimulationGrid
            Grid with appropriate parameters
        """
        return self.grid_builder.create_grid(
            structure=structure,
            config=config,
            duration=duration,
            dt=dt,
            min_natural_periods=min_natural_periods,
            force_power_of_2=True,
            z_points=200
        )