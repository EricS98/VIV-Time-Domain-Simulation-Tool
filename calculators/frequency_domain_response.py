# calculators/frequency_domain_response.py
"""
Frequency-Domain Response Calculator Module
===========================================

Calculates frequency-domain VIV response with amplitude-dependent damping.

IMPORTANT: Proper handling of aerodynamic damping parameters:
- Vickery-Basu/Eurocode: Ka_r,n = Ka_max(Re) * K_a0(U/Ucr, Iv)
- Lupi DMSM: K_a0,modal = K_a0(U/Ucr, Iv) only (NO Ka_max!)

The ka_integrator automatically returns the correct value for each model.
"""

import numpy as np
from scipy.optimize import fsolve
from typing import Optional, Dict, Any
from dataclasses import dataclass

from common.structures import StructureProperties
from config.analysis_config import VIVAnalysisConfig
from config.aerodynamic_parameters.amplitude_dependent_damping import VickeryBasuDamping, LupiDamping, EurocodeDamping
from calculators.base_calculator import BaseCalculator
from calculators.generalized_force_spectrum import GeneralizedForceSpectrumCalculator
from calculators.ka_integrator import KaHeightIntegrator
from calculators.structure_properties_calculator import StructurePropertiesCalculator

@dataclass
class FrequencyDomainResults:
    """
    Results from frequency-domain response calculation.
    
    Note on Ka_r_n field:
    - For Vickery-Basu/Eurocode: Contains Ka_r,n = Ka_max(Re) * K_a0
    - For Lupi DMSM: Contains K_a0,modal (turbulence reduction factor only)
      The actual Ka is computed as: Ka(σy/d) = envelope_curve(σy/d) * K_a0,modal
    """

    # Response
    sigma_y_squared: float  # Response variance
    sigma_y: float          # RMS response
    peak_factor: float      # Peak factor k_p
    y_max: float            # Physical peak response [m]

    # Damping components
    Ka_r_n: float           # Height-integrated Ka (interpretation depends on model)
    #delta_s: float           # Structural log. damping decrement
    delta_a: float           # Aerodynamic log. damping decrement
    delta_total: float       # Total log. damping decrement

    # Parameters
    Scruton_number: float
    modal_mass: float       # Modal mass [kg]
    natural_frequency: float
    d_ref: float               # Reference diameter

    # Spectrum
    S_Qn_fn: float          # Generalized force spectrum at fn

    # Iteration info
    converged: bool

    # Optional: equivalent linear response spectrum
    response_frequency: Optional[np.ndarray] = None
    response_spectrum: Optional[np.ndarray] = None
    transfer_function: Optional[np.ndarray] = None

    def __repr__(self) -> str:
        return (f"FrequencyDomainResult(σy={self.sigma_y*1000:.2f}mm, "
                f"y_max={self.y_max*1000:.2f}mm, "
                f"delta_total={self.delta_total:.3f}, converged={self.converged})")

class FrequencyDomainResponseCalculator(BaseCalculator):
    """
    Calculate frequency-domain VIV response with amplitude-dependent damping.

    Uses iterative solution of:
    σy² = S_Qn(fn) / (8 * (2πfn)³ * Mn² * ζn)
    """
    def __init__(self, St: float = 0.2):
        """Initialize the calculator."""
        super().__init__(St=St)
        self.ka_integrator = KaHeightIntegrator(St=St)
        self.spectrum_calculator = GeneralizedForceSpectrumCalculator(St=St)
        self.structure_calculator = StructurePropertiesCalculator(rho_air=1.25)

    def calculate_response(
            self, 
            structure: StructureProperties,
            config: VIVAnalysisConfig, 
            d_ref: float, 
            a_L: float, 
            grid = None,
            tolerance: float = 1e-6,
            verbose: bool = True
    ) -> FrequencyDomainResults:
        """
        Calculate frequency-domain VIV response.

        Parameters:
        -----------
        structure : StructureProperties
            Structure properties
        config : VIVAnalysisConfig
            Analysis configuration
        d_ref : float
            Reference diameter [m]
        grid : SimulationGrid, optional
            Pre-computed frequency grid
        tolerance : float
            Convergence tolerance for σy
        verbose : bool
            If False, suppress console output
            
        Returns:
        --------
        FrequencyDomainResult
            Complete response results
        """
        h = structure.height
        f_n = structure.f_n
        m_eq = structure.m_eq
        delta_s = structure.delta_s

        if verbose:
            print(f"\n{'='*70}")
            print(f"FREQUENCY-DOMAIN VIV RESPONSE")
            print(f"{'='*70}")
            print(f"Structure: {structure.name}")
            print(f"Formulation: {config.damping_formulation.get_name()}")
            print(f"Reference diameter d_ref: {d_ref:.3f} m")

        # 1. Calculate generalized force spectrum
        if grid is None:
            grid = self.spectrum_calculator.create_grid(structure, config)
        
        spectrum_result = self.spectrum_calculator.calculate_spectrum(
            structure, config, grid=grid
        )
        
        S_Qn_fn = spectrum_result.spectrum_at_fn
        if verbose:
            print(f"\nGeneralized Force Spectrum:")
            print(f"  S_Qn(fn) = {S_Qn_fn:.3e} N²·s")

        # 2. Calculate Scruton number
        Sc = self.structure_calculator.calculate_scruton_number(m_eq, delta_s, d_ref)
        if verbose:
            print(f"\nScruton Number:")
            print(f"  Sc = {Sc:.2f} (with d_ref = {d_ref:.3f} m)")

        # 3. Modal parameters
        omega_n = 2 * np.pi * f_n
        mode_shape_integral = config.mode_shape.compute_modal_integral(h, mode_number=1)
        M_n = m_eq * mode_shape_integral

        if verbose:
            print(f"\nModal Parameters:")
            print(f"  fn = {f_n:.4f} Hz")
            print(f"  Mn = {M_n:.1f} kg")

        # 4. Calculate load parameter
        C_an = S_Qn_fn / (8 * omega_n**3 * M_n**2 * (self.rho_air * d_ref**2 / m_eq))

        # Apply lock-in factor for Lupi damping
        if isinstance(config.damping_formulation, LupiDamping):
            # Get lock-in velocity ratio
            v_vcr_lockin = config.damping_formulation.get_lockin_ratio(Sc)
            C_an *= v_vcr_lockin**2

            if verbose:
                print(f"\nLupi Lock-in Factor:")
                print(f"  Sc = {Sc:.2f}")
                print(f"  V/Vcr at max response = {v_vcr_lockin:.2f}")
                print(f"  Lock-in factor (V/Vcr)² = {v_vcr_lockin**2:.3f}")
                print(f"  C_a,n (with lock-in) = {C_an:.3e} m²")

        if verbose and not isinstance(config.damping_formulation, LupiDamping):
            print(f"\nLoad Parameter:")
            print(f"  C_a,n = {C_an:.3e} m²")

        # 5. Calculate height-integrated aerodynamic damping parameter
        # 
        # IMPORTANT: The interpretation differs by damping model:
        # - Vickery-Basu/Eurocode: Ka_r,n = Ka_max(Re) * K_a0(U/Ucr, Iv)
        # - Lupi DMSM: K_a0,modal = K_a0(U/Ucr, Iv) only (NO Ka_max!)
        #
        # The ka_integrator.calculate_integrated_ka() automatically returns
        # the correct value for each model (see ka_integrator.py lines 92-143).
        #
        # We use different variable names to clarify:
        # - ka_reduction_factor: For Lupi (contains K_a0,modal)
        # - Ka_r_n: For Vickery-Basu/Eurocode (contains Ka_max * K_a0)
        if isinstance(config.damping_formulation, LupiDamping):
            # For Lupi: ka_integrator returns K_a0,modal (turbulence reduction factor)
            ka_reduction_factor = self._calculate_integrated_ka(
                structure, config, d_ref
            )
            Ka_r_n = None
            print(f"\n  Lupi DMSM: K_a0,modal = {ka_reduction_factor:.4f}")
            print(f"  (This will be multiplied with envelope curve, NOT with Ka_max)")
        else:
            # For Vickery-Basu/Eurocode: ka_integrator returns Ka_r,n = Ka_max * K_a0
            Ka_r_n = self._calculate_integrated_ka(
                structure, config, d_ref
            )
            ka_reduction_factor = None
            print(f"\n  Vickery-Basu/Eurocode: Ka_r,n = {Ka_r_n:.3f}")

        # 6. Iterative solution for sigma_y
        result = self._solve_response(
            structure, config, C_an, 
            Sc, Ka_r_n, a_L, d_ref,
            ka_reduction_factor=ka_reduction_factor
        )

        # 7. Calculate peak factor and peak response
        if isinstance(config.damping_formulation, LupiDamping):
            # For Lupi, use the converged Ka value
            Ka_peak_factor = result['Ka_final']
        else:
            Ka_peak_factor = Ka_r_n
        sc_over_4pi_ka = Sc / (4 * np.pi * Ka_peak_factor)
        peak_factor = self.calculate_peak_factor(sc_over_4pi_ka=sc_over_4pi_ka, formula="cicind")

        y_max = peak_factor * result['sigma_y']
        y_max_over_d = y_max / d_ref

        print(f"\nPeak Response:")
        print(f"  Peak factor: k_p = {peak_factor:.3f}")
        print(f"  Peak displacement: y_max = {y_max*1000:.2f} mm = {y_max:.4f} m")
        print(f"  Normalized peak displacement y_max/d_ref = {y_max_over_d:.3f}")

        # 8. Post-process: compute equivalent linear response spectrum S_y(f)
        response_frequency = None
        response_spectrum = None
        transfer_function = None

        try:
            # Total damping ratio
            zeta_total = result['delta_total'] / (2.0 * np.pi)

            # Frequency array and generalized force spectrum
            f = spectrum_result.frequency
            S_Qn = spectrum_result.spectrum

            # Modal parameters
            omega = 2.0 * np.pi * f               # [rad/s]
            omega_n = 2.0 * np.pi * f_n           # already defined above
            K_n = M_n * omega_n**2               # [N/m]
            C_eq = 2.0 * zeta_total * M_n * omega_n  # [Ns/m]

            # Transfer function H(ω) from generalized force to displacement
            denominator = K_n - M_n * omega**2 + 1j * C_eq * omega
            H = 1.0 / denominator                       # [m/N]

            # Response PSD
            S_y = np.abs(H)**2 * S_Qn       # [m^2·s]

            # Attach to result object
            response_frequency = f
            response_spectrum = S_y
            transfer_function = H

        except Exception as e:
            print(f"⚠️  Could not compute FD response spectrum: {e}")

        # 9. Create results object
        fd_results = FrequencyDomainResults(
            sigma_y_squared=result['sigma_y_squared'],
            sigma_y=result['sigma_y'],
            peak_factor=peak_factor,
            y_max=y_max,
            Ka_r_n=Ka_r_n if Ka_r_n is not None else ka_reduction_factor,
            delta_a=result['delta_a'],
            delta_total=result['delta_total'],
            Scruton_number=Sc,
            modal_mass=M_n,
            natural_frequency=f_n,
            d_ref=d_ref,
            S_Qn_fn=S_Qn_fn,
            converged=result['converged'],
        )

        fd_results.response_frequency = response_frequency
        fd_results.response_spectrum = response_spectrum
        fd_results.transfer_function = transfer_function

        return fd_results
    
    def _calculate_integrated_ka(
        self,
        structure: StructureProperties,
        config: VIVAnalysisConfig,
        d_ref: float
    ) -> float:
        """
        Calculate height-integrated aerodynamic damping parameter.
        
        Returns:
        --------
        float
            For Vickery-Basu/Eurocode: Ka_r,n = Ka_max * K_a0
            For Lupi DMSM: K_a0,modal (turbulence reduction factor only)
        """
        
        # Discretize height
        n_points = 100
        z_points = np.linspace(0, structure.height, n_points)

        # Get height-dependent properties
        d_z = config.cross_section.get_diameter(z_points, structure.height, structure.diameter)
        u_z = config.wind_profile.get_velocity(z_points, h=structure.height)
        phi_z = config.mode_shape.evaluate(z_points, structure.height, mode_number=1)
        Re_z = self.calculate_reynolds_number(u_z, d_z)
        Ka_max_z = config.damping_parameter.get_ka_max(Re_z)

        # Calculate height-integrated parameter
        # The ka_integrator automatically handles Lupi vs Vickery-Basu
        Ka_r_n = self.ka_integrator.calculate_integrated_ka(
            structure, config, z_points, d_z, u_z, phi_z,
            Ka_max_z, d_ref
        )

        return Ka_r_n      
    
    def _solve_response(
        self,
        structure: StructureProperties,
        config: VIVAnalysisConfig,
        C_an: float,
        Sc: float,
        Ka_r_n: float,
        a_L: float,
        d_ref: float,
        ka_reduction_factor: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Solve for response variance.

        Governing relation: σ_y² = C_a,n / (Sc/(4π) - Ka(σ_y))
        
        Parameters:
        -----------
        Ka_r_n : float
            For Vickery-Basu/Eurocode: Ka_r,n = Ka_max * K_a0
            For Lupi: Not used (None)
        ka_reduction_factor : float, optional
            For Lupi: K_a0,modal (turbulence reduction factor)
            For Vickery-Basu/Eurocode: Not used (None)
        """
        m_eq = structure.m_eq
        delta_s = structure.delta_s
        sc_over_4pi = Sc / (4.0 * np.pi)

        print(f"\n{'='*70}")
        print(f"RESPONSE CALCULATION")
        print(f"{'='*70}")
        print(f"Method: {config.damping_formulation.get_name()}")

        # Vickery & Basu formulation
        if isinstance(config.damping_formulation, VickeryBasuDamping):

            # Case 1: Outside lock-in: Ka_r,n ≤ 0
            if Ka_r_n < 0:
                print("ℹ️  Ka_r,n ≤ 0 → aerodynamic damping stabilizes the system.")
                print("ℹ️  Using background (non-resonant) response without aero damping.")

                # Closed-form buffeting RMS at resonance
                sigma_y_squared = C_an / sc_over_4pi
                sigma_y = np.sqrt(sigma_y_squared)
                sigma_y_over_d = sigma_y / d_ref

                Ka_final = Ka_r_n
                converged = True

                print(f"  σ_y/d_ref = {sigma_y_over_d:.6f}")
                print(f"  σ_y = {sigma_y:.6f} m (background-only)")
            
            # Case 2: Inside lock-in: Ka_r,n > 0
            else:
                print("Using analytical closed-form solution")
            
                c1 = (a_L**2 / 2.0) * (1.0 - Sc / (4.0 * np.pi * Ka_r_n))
                c2 = (a_L**2 * C_an) / (Ka_r_n * d_ref**2)

                discriminant = c1**2 + c2
                if discriminant < 0:
                    raise ValueError(
                        f"No real solution exists (discriminant={discriminant:.6e} < 0).\n"
                        f"Parameters: Sc={Sc:.4f}, Ka_r,n={Ka_r_n:.4f}, C_a,n={C_an:.6e}"
                    )
            
                sigma_y_over_d = np.sqrt(c1 + np.sqrt(discriminant))
                sigma_y = sigma_y_over_d * d_ref
                sigma_y_squared = sigma_y**2

                # Sanity check
                if sigma_y > a_L * d_ref:
                    raise ValueError(
                        f"Analytical solution σ_y={sigma_y:.6f} m exceeds "
                        f"valid range a_L*d_ref={a_L*d_ref:.6f} m.\n"
                        f"This indicates insufficient structural damping."
                    )

                # Convert damping parameters to log. damping decrements
                Ka_final = config.damping_formulation.calculate_ka(sigma_y, Ka_r_n, d_ref)
                converged = True

                print(f"  c1 = {c1:.6e}")
                print(f"  c2 = {c2:.6e}")
                print(f"  σ_y/d_ref = {sigma_y_over_d:.6f}")
                print(f"  σ_y = {sigma_y:.6f} m")
        
        # Lupi damping
        elif isinstance(config.damping_formulation, LupiDamping):
            print("Finding minimum stable amplitude for Lupi damping")
            
            # For Lupi: ka_reduction_factor is K_a0,modal
            # It gets passed to calculate_ka() which multiplies it with the envelope curve

            def damping_difference(sy):
                """
                Find where Ka(σ_y) = Sc/(4π)
                
                For Lupi: Ka(σy/d) = envelope_curve(σy/d) * K_a0,modal
                where ka_reduction_factor = K_a0,modal from ka_integrator
                """
                Ka = config.damping_formulation.calculate_ka(sy, ka_reduction_factor, d_ref)
                return sc_over_4pi - Ka
            
            x0 = C_an / sc_over_4pi     # initial value = static response
            sigma_y_min = fsolve(damping_difference, x0=x0, full_output=False)[0]
            print(f"  Minimum stable amplitude: σ_y_min = {sigma_y_min:.6f} m")
                       
            # Solve for equilibrium: σ_y² = C_a,n / (Sc/(4π) - Ka(σ_y))
            def residual(sy):
                Ka = config.damping_formulation.calculate_ka(sy, ka_reduction_factor, d_ref)
                damping_diff = sc_over_4pi - Ka
                if damping_diff <= 0:
                    return 1e10
                return (sy / d_ref)**2 - C_an / (d_ref**2 * damping_diff)
            
            sigma_y = fsolve(residual, x0=sigma_y_min * 1.01,
                            maxfev=100, full_output=False)[0]
            
            Ka_final = config.damping_formulation.calculate_ka(sigma_y, ka_reduction_factor, d_ref)
            damping_diff_final = sc_over_4pi - Ka_final
            sigma_y_squared = sigma_y**2

            converged = True

            print(f"  Final σ_y = {sigma_y:.6f} m")
            print(f"  Final Ka = {Ka_final:.4f}")

        # Eurocode formulation
        elif isinstance(config.damping_formulation, EurocodeDamping):
            print("Solving Eurocode damping formulation")

            if sc_over_4pi >= Ka_r_n:
                # Case 1: Sufficient structural damping (total damping positive for all values of sigma_y)
                print(f"  Sc/(4π)={sc_over_4pi:.4f} ≥ Ka_r,n={Ka_r_n:.4f}")
                print("  Total damping always positive → using static response as initial guess")

                def residual(sy):
                    Ka = config.damping_formulation.calculate_ka(sy, Ka_r_n, d_ref)
                    damping_diff = sc_over_4pi - Ka
                    return (sy/d_ref)**2 - C_an / (d_ref**2 * damping_diff)
            
                x0 = np.sqrt(C_an / sc_over_4pi)    # initial value = static response
                sigma_y = fsolve(residual, x0=x0, maxfev=100, full_output=False)[0]
            
                Ka_final = config.damping_formulation.calculate_ka(sigma_y, Ka_r_n, d_ref)
                converged = True

            else:
                # Case 2: total damping can be negative for small values of sigma_y
                print(f"  Sc/(4π)={sc_over_4pi:.4f} < Ka_r,n={Ka_r_n:.4f}")
                print("  Finding minimum stable amplitude")

                def damping_difference(sy):
                    """Find where Ka(σ_y) = Sc/(4π)"""
                    Ka = config.damping_formulation.calculate_ka(sy, Ka_r_n, d_ref)
                    return sc_over_4pi - Ka
                
                x0 = np.sqrt(C_an / sc_over_4pi)    # initial value = static response
                sigma_y_min = fsolve(damping_difference, x0=x0, full_output=False)[0]
                print(f"  Minimum stable amplitude: σ_y_min = {sigma_y_min:.6f} m")

                # Solve for equilibrium: σ_y² = C_a,n / (Sc/(4π) - Ka(σ_y))
                def residual(sy):
                    Ka = config.damping_formulation.calculate_ka(sy, Ka_r_n, d_ref)
                    damping_diff = sc_over_4pi - Ka
                    if damping_diff <= 0:
                        return 1e10
                    return (sy/d_ref)**2 - C_an / (d_ref**2 * damping_diff)

                sigma_y = fsolve(residual, x0=sigma_y_min*1.01, full_output=False)[0]
                
                Ka_final = config.damping_formulation.calculate_ka(sigma_y, Ka_r_n, d_ref)
                sigma_y_squared = sigma_y**2
                converged = True

            print(f"  Final σ_y = {sigma_y:.6f} m")

        else:
            raise NotImplementedError(
                f"Damping formulation {type(config.damping_formulation)} not supported"
            )
        
        # Calculate final results
        damping_diff_final = sc_over_4pi - Ka_final

        delta_a = Ka_final * 2 * np.pi * (self.rho_air * d_ref**2) / m_eq
        delta_total = delta_s - delta_a

        print(f"\n✅ Solution found:")
        print(f"  σ_y = {sigma_y:.6f} m")
        print(f"  Ka = {Ka_final:.4f}")
        print(f"  Sc/(4π) - Ka = {damping_diff_final:.4f}")
        print(f"  Converged: {converged}")

        return {
            'sigma_y': sigma_y,
            'sigma_y_squared': sigma_y_squared,
            'delta_a': delta_a,
            'delta_total': delta_total,
            'converged': converged,
            'Ka_final': Ka_final
        }