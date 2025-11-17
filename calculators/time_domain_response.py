# calculators/time_domain_response.py
"""
Time-Domain Response Calculator Module
======================================

Calculates time-domain VIV response using:
1. Generalized force spectrum synthesis to time series
2. Non-linear Newmark time integration with amplitude-dependent damping
3. Statistical analysis of response time series
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

from common.structures import StructureProperties
from config.analysis_config import VIVAnalysisConfig
from calculators.base_calculator import BaseCalculator
from calculators.generalized_force_spectrum import GeneralizedForceSpectrumCalculator
from common.time_domain_base import TimeGridBuilder, SimulationGrid, TimeSeriesSynthesizer
from common.time_integration import NewmarkIntegrator, NewmarkParameters, NonLinearConvergenceParams, RungeKuttaIntegrator
from config.aerodynamic_parameters.amplitude_dependent_damping import VickeryBasuDamping
from config.time_domain import NewmarkMethod, RungeKuttaMethod

@dataclass
class TimeDomainResults:
    """Results from time-domain response calculation."""

    # Time series
    time: np.ndarray
    displacement: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    force: np.ndarray

    # Statistics
    sigma_y: float
    sigma_ydot: float
    y_max: float
    y_min: float
    peak_factor: float

    # Modal parameters
    modal_mass: float
    modal_stiffness: float
    natural_frequency: float

    # Damping evolution
    damping_history: Optional[Dict[str, np.ndarray]] = None

    # Grid info
    grid: SimulationGrid = None

    # Steady-state / limit-cycle onset
    steady_start_index: int = 0
    steady_start_time: float = 0.0

    # Convergence
    converged: bool = True

    def __repr__(self) -> str:
        return (f"TimeDomainResults(σ_y={self.sigma_y*1000:.2f}mm, "
                f"y_max={self.y_max*1000:.2f}mm, "
                f"k_p={self.peak_factor:.3f}, "
                f"converged={self.converged})")
    
class TimeDomainResponseCalculator(BaseCalculator):
    """
    Calculate time-domain VIV response with amplitude-dependent damping.
    """

    def __init__(self, St: float = 0.2):
        """Initialize the calculator."""
        super().__init__(St=St)
        self.spectrum_calculator = GeneralizedForceSpectrumCalculator(St=St)
        self.grid_builder = TimeGridBuilder()

    def calculate_response(
        self,
        structure: StructureProperties,
        config: VIVAnalysisConfig,
        d_ref: float,
        grid: Optional[SimulationGrid] = None,
        duration: float = 600.0,
        dt: Optional[float] = None,
        min_natural_periods: int = 1000,
        random_seed: Optional[int] = None,
        u0: float = 0.0,
        udot0: float = 0.0,
        convergence_params: Optional[NonLinearConvergenceParams] = None,
        store_damping_history: bool = True
    ) -> TimeDomainResults:
        """
        Calculate time-domain VIV response.

        Parameters:
        -----------
        structure : StructureProperties
            Structure properties
        config : VIVAnalysisConfig
            Analysis configuration
        d_ref : float
            Reference diameter [m]
        grid : SimulationGrid, optional
            Pre-computed simulation grid. If None, created automatically.
        duration : float
            Simulation duration [s] (default: 600)
        dt : float, optional
            Time step [s]. If None, auto-calculated.
        min_natural_periods : int
            Minimum number of natural periods (default: 1000)
        random_seed : int, optional
            Random seed for force synthesis
        u0, udot0 : float
            Initial displacement [m] and velocity [m/s]
        convergence_params : NonLinearConvergenceParams, optional
            Convergence parameters for non-linear integration
        store_damping_history : bool
            Whether to store damping evolution history
            
        Returns:
        --------
        TimeDomainResults
            Complete time-domain response results
        """
        print(f"\n{'='*70}")
        print(f"TIME-DOMAIN VIV RESPONSE CALCULATION")
        print(f"{'='*70}")
        print(f"Structure: {structure.name}")
        print(f"Damping model: {config.damping_formulation.get_name()}")
        print(f"Reference diameter d_ref: {d_ref:.3f} m")

        # Step 1: Create or validate simulation grid
        if grid is None:
            print(f"\nCreating simulation grid...")
            grid = self._create_grid(
                structure, config, duration, dt, min_natural_periods
            )
        else:
            print(f"\nUsing provided simulation grid:")
            print(f"  Duration: {grid.T:.1f} s")
            print(f"  Time step: {grid.dt:.4f} s")
            print(f"  N samples: {grid.N}")

        # Step 2: Calculate generalized force spectrum
        print(f"\nCalculating generalized force spectrum...")
        spectrum_result = self.spectrum_calculator.calculate_spectrum(
            structure, config, grid=grid
        )

        # Step 3: Synthesize force time series
        force_series = self._synthesize_force(
            spectrum_result.spectrum, grid, random_seed
        )

        # Step 4: Set up modal equation of motion
        print(f"\nSetting up modal equation of motion...")
        M_n, C_func, C_deriv_func, K_n = self._setup_modal_system(
            structure, config, d_ref, grid.dt
        )

        # --- Step 5: Time integration ---
        # Determine which integrator to use
        integration_method = config.time_domain_config.integration_method

        print(f"\nPerforming time integration...")
        print(f"  Method: {integration_method.get_name()}")

        if isinstance(integration_method, RungeKuttaMethod):
            # Use Runge-Kutta integrator
            integrator_rk = RungeKuttaIntegrator()
            y, ydot, ydotdot = integrator_rk.integrate_nonlinear(
                M=M_n,
                K=K_n,
                damping_func=C_func,
                f=force_series,
                u0=u0,
                udot0=udot0,
                dt=grid.dt
            )
            damping_history = None
            store_damping_history = False
        else:
            # Use Newmark integration
            gamma, beta = config.time_domain_config.get_newmark_parameters()
            integrator = NewmarkIntegrator(NewmarkParameters(gamma=gamma, beta=beta))

            if convergence_params is None:
                convergence_params = NonLinearConvergenceParams()

            result = integrator.integrate_nonlinear_krenk(
                M=M_n,
                K=K_n,
                damping_func=C_func,
                damping_derivative_func=C_deriv_func,
                f=force_series,
                u0=u0,
                udot0=udot0,
                dt=grid.dt,
                convergence_params=convergence_params,
                store_rms_history=store_damping_history
            )

            # Unpack results
            if store_damping_history:
                y, ydot, ydotdot, damping_history = result
            else:
                y, ydot, ydotdot = result
                damping_history = None

        # Step 6: Calculate statistics
        print(f"\nCalculating response statistics...")
        stats = self._calculate_statistics(y, ydot, grid.t, structure, config, damping_history=damping_history)

        print(f"\n{'='*70}")
        print(f"TIME-DOMAIN RESULTS")
        print(f"{'='*70}")
        print(f"  RMS displacement: σ_y = {stats['sigma_y']*1000:.2f} mm")
        print(f"  Max displacement: y_max = {stats['y_max']*1000:.2f} mm")
        print(f"  Min displacement: y_min = {stats['y_min']*1000:.2f} mm")
        print(f"  Peak factor: k_p = {stats['peak_factor']:.3f}")
        print(f"  RMS velocity: σ_ẏ = {stats['sigma_ydot']*1000:.2f} mm/s")
        print(f"{'='*70}")
        
        # Create results object
        return TimeDomainResults(
            time=grid.t,
            displacement=y,
            velocity=ydot,
            acceleration=ydotdot,
            force=force_series,
            sigma_y=stats['sigma_y'],
            sigma_ydot=stats['sigma_ydot'],
            y_max=stats['y_max'],
            y_min=stats['y_min'],
            peak_factor=stats['peak_factor'],
            modal_mass=M_n,
            modal_stiffness=K_n,
            natural_frequency=structure.f_n,
            damping_history=damping_history,
            grid=grid,
            converged=True,
            steady_start_index=stats.get('steady_start_index', 0),
            steady_start_time=stats.get('steady_start_time', 0.0)
        )
    
    def _create_grid(
        self,
        structure: StructureProperties,
        config: VIVAnalysisConfig,
        duration: float,
        dt: Optional[float],
        min_natural_periods: int
    ) -> SimulationGrid:
        """Create simulation grid for time-domain analysis."""

        return self.grid_builder.create_grid(
            structure=structure,
            config=config,
            duration=duration,
            dt=dt,
            min_natural_periods=min_natural_periods,
            force_power_of_2=True
        )
    
    def _synthesize_force(
        self,
        S_Qn: np.ndarray,
        grid: SimulationGrid,
        random_seed: Optional[int]
    ) -> np.ndarray:
        """Synthesize generalized force time series from spectrum."""

        force_series = TimeSeriesSynthesizer.synthesize_from_psd(
            S_xx=S_Qn,
            df=grid.df,
            N=grid.N,
            random_seed=random_seed
        )

        print(f"  Force RMS: {np.std(force_series):.2e} N")
        print(f"  Force peak: {np.max(np.abs(force_series)):.2e} N")
        
        return force_series
    
    def _setup_modal_system(
        self,
        structure: StructureProperties,
        config: VIVAnalysisConfig,
        d_ref: float,
        dt: float
    ) -> Tuple[float, callable, callable, float]:
        """
        Set up modal equation of motion parameters.

        Returns:
        --------
        M_n : float
            Modal mass [kg]
        C_func : callable
            Damping function C(y, ẏ)
        C_deriv_func : callable
            Damping derivative ∂C/∂y
        K_n : float
            Modal stiffness [N/m]
        """
        h = structure.height
        f_n = structure.f_n
        m_eq = structure.m_eq
        delta_s = structure.delta_s

        # Modal mass
        mode_shape_integral = config.mode_shape.compute_modal_integral(h, mode_number=1)
        M_n = m_eq * mode_shape_integral

        # Modal stiffness
        omega_n = 2 * np.pi * f_n
        K_n = M_n * omega_n**2

        # Structural damping coefficient
        zeta_s = delta_s / (2 * np.pi)
        C_s = 2 * zeta_s * np.sqrt(K_n * M_n)

        print(f"  Modal mass: M_n = {M_n:.1f} kg")
        print(f"  Modal stiffness: K_n = {K_n:.2e} N/m")
        print(f"  Structural damping: C_s = {C_s:.2e} Ns/m (ζ_s = {zeta_s:.4f})")

        # Set up amplitude-dependent damping

        # IMPORTANT: Ka_r_n interpretation depends on damping model:
        # - Vickery-Basu/Eurocode: Ka_r,n = Ka_max(Re) * K_a0
        # - Lupi DMSM: K_a0,modal = K_a0 only
        from common.time_domain_damping import create_time_domain_damping
        # Create generic time-domain damping
        td_damping = create_time_domain_damping(
            structure=structure,
            config=config,
            d_ref=d_ref,
            dt=dt,
            window_cycles=2.0
        )

        def C_func(y, ydot):
            """Total damping function."""
            C_aero = td_damping.get_damping_coefficient(y, ydot)
            return C_s - C_aero
            
        def C_deriv_func(y, ydot):
            """Derivative of damping w.r.t. displacement."""
            return td_damping.get_damping_derivative(y, ydot)
            
        # Attach the damping object for history tracking
        C_func.vb_damping = td_damping

        print(f"  Aerodynamic damping: {config.damping_formulation.get_name()}")
        
        return M_n, C_func, C_deriv_func, K_n
    
    def _calculate_statistics(
        self,
        y: np.ndarray,
        ydot: np.ndarray,
        t: np.ndarray,
        structure: StructureProperties,
        config: VIVAnalysisConfig,
        damping_history: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, float]:
        """
        Calculate response statistics, excluding initial transient.
        If no damping history is available, fallback to simple fixed
        trim fraction.

        Parameters:
        -----------
        y : np.ndarray
            Displacement time series [m]
        ydot : np.ndarray
            Velocity time series [m/s]
        t : np.ndarray
            Time array [s]
        structure : StructureProperties
            Structure properties (used for f_n)
        config : VIVAnalysisConfig
            Full analysis config (used for time-domain settings)
            
        Returns:
        --------
        dict : Statistics dictionary
        """
        n_samples = len(y)

        # Calculate samples per period for period-based trimming
        T_n = 1.0 / structure.f_n
        dt = t[1] - t[0] if len(t) > 1 else 0.01
        samples_per_period = max(1, int(round(T_n / dt)))

        # Minimum startup period (always applied)
        min_transient_periods = 20
        min_samples = min_transient_periods * samples_per_period

        # Default fallback: simple fixed trim
        max_trim_fraction = config.time_domain_config.max_trim_fraction
        n_skip = int(max_trim_fraction * n_samples)

        # Try to use damping history if available
        if damping_history is not None and 'zeta_total' in damping_history:
            zeta_total = np.asarray(damping_history['zeta_total'])
            zeta_struct = np.asarray(
                damping_history.get('zeta_struct', np.zeros_like(zeta_total))
            )

            # Make sure available length is not exceeded
            n_samples = min(n_samples, len(zeta_total))

            # Calculate mean total damping after minimum startup
            start_idx = min(min_samples, n_samples - 1)
            zeta_total_mean = np.mean(zeta_total[start_idx:])
            zeta_total_std = np.std(zeta_total[start_idx:])
            zeta_s_mean = np.mean(zeta_struct[start_idx:]) if zeta_struct.size else 0.0

            # Check if damping is essentially constant
            if abs(zeta_total_mean) > 1e-6:
                relative_variation = zeta_total_std / abs(zeta_total_mean)
            else:
                relative_variation = float('inf')

            is_damping_stable = relative_variation < 0.01

            # Case 1: Damping is stable and positive
            if is_damping_stable and zeta_total_mean > 0:
                n_skip = min_samples    # Around 20 periods

                print(f"  ✅ Stable damping detected:")
                print(f"     ζ_total = {zeta_total_mean*100:.3f}% (±{zeta_total_std*100:.3f}%)")
                print(f"     System at steady-state, using minimal trim: {min_transient_periods} periods")

            # Case 2: Damping evolving (negative or approaching zero) -> Limit-cycle detection
            else:
                if abs(zeta_s_mean) > 0.0:
                    zeta_tol = 0.05 * abs(zeta_s_mean)
                else:
                    zeta_tol = 1e-3

                indices = np.where(np.abs(zeta_total[start_idx:n_samples]) <= zeta_tol)[0]

                if indices.size > 0:
                    n_skip_candidate = start_idx + int(indices[0])
                    n_skip = min(n_skip_candidate, int(max_trim_fraction * n_samples))
                    print(f"  ✅ Limit-cycle detected: |ζ_total| ≈ 0 at t = {t[n_skip]:.1f}s")
                else:
                    # Criterion not met: use maximum allowed trim
                    print(f"  ⚠️  Limit-cycle criterion not met (|ζ_total| > {zeta_tol*100:.3f}%)")
                    print(f"     ζ_total mean = {zeta_total_mean*100:.3f}%, std = {zeta_total_std*100:.3f}%")
                    print(f"     Using max trim = {max_trim_fraction*100:.0f}%")

        else:
            # No damping history available: use default fallback
            print(f"  ℹ️  No damping history available, using default trim = {max_trim_fraction*100:.0f}%")

        # Ensure valid range
        n_skip = max(0, min(n_skip, n_samples - 1))

        # Use only steady-state for statistics
        y_steady = y[n_skip:]
        ydot_steady = ydot[n_skip:]

        # Calculate statistics 
        sigma_y = np.std(y_steady)
        sigma_ydot = np.std(ydot_steady)
        y_max = np.max(np.abs(y_steady))
        y_min = np.min(y_steady)

        # Peak factor
        peak_factor = y_max / sigma_y if sigma_y > 0 else 0.0

        # Steady-state / limit-cycle onset information
        steady_start_index = int(n_skip)
        # Guard against indexing off the end in some edge case
        if steady_start_index >= len(t):
            steady_start_index = len(t) - 1
        steady_start_time = float(t[steady_start_index])

        return {
            'sigma_y': sigma_y,
            'sigma_ydot': sigma_ydot,
            'y_max': y_max,
            'y_min': y_min,
            'peak_factor': peak_factor,
            'steady_start_index': steady_start_index,
            'steady_start_time': steady_start_time
        }
    
    def compare_with_frequency_domain(
        self,
        time_domain_result: TimeDomainResults,
        freq_domain_result
    ) -> Dict[str, Any]:
        """
        Compare time-domain results with frequency-domain results.

        Parameters:
        -----------
        time_domain_result : TimeDomainResults
            Time-domain results
        freq_domain_result : FrequencyDomainResults
            Frequency-domain results for comparison
            
        Returns:
        --------
        dict : Comparison metrics
        """
        sigma_y_td = time_domain_result.sigma_y
        sigma_y_fd = freq_domain_result.sigma_y

        y_max_td = time_domain_result.y_max
        y_max_fd = freq_domain_result.y_max

        kp_td = time_domain_result.peak_factor
        kp_fd = freq_domain_result.peak_factor

        comparison = {
            'sigma_y_ratio': sigma_y_td / sigma_y_fd if sigma_y_fd > 0 else np.nan,
            'y_max_ratio': y_max_td / y_max_fd if y_max_fd > 0 else np.nan,
            'peak_factor_diff': kp_td - kp_fd,
            'sigma_y_td': sigma_y_td,
            'sigma_y_fd': sigma_y_fd,
            'y_max_td': y_max_td,
            'y_max_fd': y_max_fd,
            'kp_td': kp_td,
            'kp_fd': kp_fd
        }

        print(f"\n{'='*70}")
        print(f"TIME-DOMAIN vs FREQUENCY-DOMAIN COMPARISON")
        print(f"{'='*70}")
        print(f"RMS displacement:")
        print(f"  Time-domain:      σ_y = {sigma_y_td*1000:.2f} mm")
        print(f"  Frequency-domain: σ_y = {sigma_y_fd*1000:.2f} mm")
        print(f"  Ratio (TD/FD):    {comparison['sigma_y_ratio']:.3f}")
        print(f"\nPeak displacement:")
        print(f"  Time-domain:      y_max = {y_max_td*1000:.2f} mm")
        print(f"  Frequency-domain: y_max = {y_max_fd*1000:.2f} mm")
        print(f"  Ratio (TD/FD):    {comparison['y_max_ratio']:.3f}")
        print(f"\nPeak factor:")
        print(f"  Time-domain:      k_p = {kp_td:.3f}")
        print(f"  Frequency-domain: k_p = {kp_fd:.3f}")
        print(f"  Difference:       Δk_p = {comparison['peak_factor_diff']:.3f}")
        print(f"{'='*70}")
        
        return comparison