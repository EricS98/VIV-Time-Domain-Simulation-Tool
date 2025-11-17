# common/time_domain_damping.py
"""
Generic Time-Domain Damping Wrapper
===================================

Universal wrapper for amplitude-dependent aerodynamic damping in time-domain VIV simulations.

Works with any damping formulation:
- VickeryBasuDamping
- LupiDamping
- EurocodeDamping
- Any custom formulation inheriting from AmplitudeDependentDamping

Key concept:
- All formulations use RMS displacement: Ka = f(σ_y)
- Time-domain needs instantaneous damping: C(t) for each time step
- Solution: Calculate "local RMS" from running window of recent displacements

The wrapper:
1. Maintains circular buffer of recent displacements (2-3 natural periods)
2. Calculates running RMS: σ_y(t) from window
3. Calls formulation's calculate_ka(σ_y) method
4. Converts Ka to damping coefficient C_a(t)
"""

import numpy as np
from collections import deque
from typing import Optional, Dict, Any

from common.structures import StructureProperties
from config.analysis_config import VIVAnalysisConfig
from config.aerodynamic_parameters.amplitude_dependent_damping import AmplitudeDependentDamping


class TimeDomainDamping:
    """
    Generic time-domain wrapper for amplitude-dependent aerodynamic damping.

    This class provides the bridge between:
    - Frequency-domain: Ka(σ_y) based on global RMS displacement
    - Time-domain: C(y,ẏ) based on instantaneous displacement

    ethod:
    -------
    1. Maintain running history of y(t) over recent cycles
    2. Calculate local RMS: σ_y(t) from this window
    3. Call formulation's calculate_ka(σ_y(t))
    4. Convert Ka to damping coefficient C_a(t)
    """

    def __init__(
        self,
        structure: StructureProperties,
        config: VIVAnalysisConfig,
        damping_formulation: AmplitudeDependentDamping,
        Ka_r_n: float,
        d_ref: float,
        f_n: float,
        m_eq: float,
        M_n: float,
        dt: float,
        delta_s: float,
        window_cycles: float = 2.0,
        rho: float = 1.25
    ):
        """
        Initialize generic time-domain damping calculator.

        Parameters:
        -----------
        damping_formulation : AmplitudeDependentDamping
            Any damping formulation object (VickeryBasuDamping, LupiDamping, etc.)
            Must have calculate_ka(sigma_y, Ka_r_n, d_ref) method
        Ka_r_n : float
            Height-integrated aerodynamic parameter [-]
            Interpretation depends on damping formulation:
            - Vickery-Basu/Eurocode: Ka_r,n = Ka_max(Re) * K_a0
            - Lupi DMSM: K_a0,modal = K_a0 (turbulence factor only, NO Ka_max)
        d_ref : float
            Reference diameter [m]
        f_n : float
            Natural frequency [Hz]
        m_eq : float
            Equivalent mass per unit length [kg/m]
        M_n : float
            Modal mass [kg]
        dt : float
            Time step [s]
        delta_s : float
            Structural logarithmic decrement [-]
        window_cycles : float
            Number of natural periods for RMS window (default: 2.0)
            Recommended: 2.0 - 3.0
        rho : float
            Air density [kg/m³] (default: 1.25)
        """
        # Store references
        self.structure = structure
        self.config = config

        # Store damping formulation
        self.damping_formulation = damping_formulation

        # Store parameters
        self.Ka_r_n = Ka_r_n
        self.d_ref = d_ref
        self.f_n = f_n
        self.m_eq = m_eq
        self.M_n = M_n
        self.dt = dt
        self.rho = rho
        self.window_cycles = window_cycles
        self.delta_s = delta_s
        self.K_a0 = self._calculate_initial_K_a0(config)

        # Calculate window length in time steps
        T_n = 1.0 / f_n     # Natural period
        window_time = window_cycles * T_n   # Window duration
        self.window_length = int(np.ceil(window_time / dt))

        if self.window_length < 4:
            import warnings
            warnings.warn(
                f"Window length is very short ({self.window_length} steps). "
                f"Consider increasing dt or window_cycles."
            )

        # Initialize circular buffer for displacement history
        # Using deque with maxlen automatically drops oldest values
        self.displacement_history = deque(maxlen=self.window_length)

        # Current state 
        self.current_rms = 0.0
        self.current_time = 0.0
        self._aero_enabled = False
        self.time_step = 0

        # For tracking
        self.Ka_current = self.Ka_r_n    # Start with maximum Ka

        print(f"\n  Generic TimeDomainDamping initialized:")
        print(f"    Formulation: {self.damping_formulation.get_name()}")
        print(f"    Ka_r,n = {Ka_r_n:.4f}")
        print(f"    d_ref = {d_ref:.3f} m")
        print(f"    Natural frequency: {f_n:.4f} Hz (T_n = {T_n:.3f} s)")
        print(f"    RMS window: {window_cycles:.1f} cycles = {window_time:.3f} s")
        print(f"    Window length: {self.window_length} time steps")

    def _calculate_initial_K_a0(self, config) -> float:
        """
        Calculate initial K_a0 turbulence reduction factor.

        For time-domain simulations with constant wind profile.
        """
        h = self.structure.height
        config = self.config

        # Wind conditions at reference height
        u_ref = config.wind_profile.get_velocity([h], h=h)[-1]
        Iv_ref = config.wind_profile.get_turbulence_intensity([h], h)[-1]

        # Critical velocity
        u_crit = self.f_n * self.d_ref / config.St
        u_ucr = u_ref / u_crit

        # Apply turbulence modification
        K_a0 = config.damping_modification.apply(
            Ka_max=1.0,
            Iv=Iv_ref,
            u_ucr=u_ucr
        )
        return K_a0

    def get_damping_coefficient(self, y: float, ydot: float) -> float:
        """
        Calculate aerodynamic damping coefficient at current time step.

        This is called by the Newmark integrator at each time step.

        Parameters:
        -----------
        y : float
            Current displacement [m]
        ydot : float
            Current velocity [m/s] (not used, but kept for interface consistency)
            
        Returns:
        --------
        float
            Aerodynamic damping coefficient C_a [Ns/m]
        """

        # Add current displacement to history
        self.displacement_history.append(y)

        if (not self._aero_enabled) and (len(self.displacement_history) >= self.window_length):
            # first window just filled - enable aerodynamic damping from now on
            self._aero_enabled = True

        # No aerodynamic damping in the first window
        if not self._aero_enabled:
            return 0.0

        # Calculate running RMS from history window
        if len(self.displacement_history) > 0:
            # RMS of recent displacements
            y_squared = np.array([d**2 for d in self.displacement_history])
            self.current_rms = np.sqrt(np.mean(y_squared))
        else:
            self.current_rms = 0.0

        # Calculate Ka based on current RMS using formulation's method
        self.Ka_current = self.damping_formulation.calculate_ka(
            sigma_y=self.current_rms,
            Ka_r_n=self.Ka_r_n,
            d_ref = self.d_ref
        )

        # Aerodynamic damping ratio from Ka
        zeta_a = (self.rho * self.d_ref**2 / self.m_eq) * self.Ka_current

        # Convert Ka to damping coefficient
        # C_a = (ρ d² / m_eq) × 2π f_n × M_n × Ka
        omega_n = 2 * np.pi * self.f_n
        C_a = 2.0 * self.M_n * omega_n * zeta_a
        #C_a = (self.rho * self.d_ref**2 / self.m_eq) * \
             # 2 * np.pi * self.f_n * self.M_n * self.Ka_current
        
        return C_a
    
    def get_damping_derivative(self, y: float, ydot: float) -> float:
        """
        Calculate derivative of damping coefficient w.r.t. displacement.

        For non-linear Newmark integration, we need ∂C/∂y for the tangent stiffness.

        Uses finite differences to work with any formulation automatically.
        If the formulation provides an analytical derivative method
        (calculate_ka_derivative), it will be used instead for better accuracy.

        Parameters:
        -----------
        y : float
            Current displacement [m]
        ydot : float
            Current velocity [m/s]
            
        Returns:
        --------
        float
            Damping derivative ∂C/∂y [N/m]
        """
        if not self._aero_enabled:
            return 0.0

        # Check if formulation has analytical derivative
        if hasattr(self.damping_formulation, 'calculate_ka_derivative'):
            # Use analytical derivative (more accurate)
            if self.current_rms > 1e-10 and len(self.displacement_history) > 0:
                # ∂Ka/∂σ_y from formulation
                dKa_dsigma = self.damping_formulation.calculate_ka_derivative(
                    sigma_y=self.current_rms,
                    Ka_r_n = self.Ka_r_n,
                    d_ref = self.d_ref
                )

                # ∂σ_y/∂y ≈ y / (σ_y × N_window)
                sigma_eff = max(self.current_rms, 1e-6) # Avoid division by zero
                dsigma_dy = y / (sigma_eff * len(self.displacement_history))
                
                # ∂C/∂Ka (same for all formulations)
                dC_dKa = (self.rho * self.d_ref**2 / self.m_eq) * \
                         2 * np.pi * self.f_n * self.M_n
                
                # Chain rule: ∂C/∂y = (∂C/∂Ka) × (∂Ka/∂σ_y) × (∂σ_y/∂y)
                dC_dy = dC_dKa * dKa_dsigma * dsigma_dy
            else:
                dC_dy = 0.0

        else:
            # Use numerical derivative (works for any formulation)
            if self.current_rms > 1e-10 and len(self.displacement_history) > 0:
                # Finite difference step
                h = max(1e-6, 1e-4 * self.current_rms)

                # Ka at σ_y + h
                Ka_plus = self.damping_formulation.calculate_ka(
                    sigma_y=self.current_rms + h,
                    Ka_r_n=self.Ka_r_n,
                    d_ref=self.d_ref
                )

                # Ka at σ_y - h
                Ka_minus = self.damping_formulation.calculate_ka(
                    sigma_y=self.current_rms - h,
                    Ka_r_n=self.Ka_r_n,
                    d_ref=self.d_ref
                )

                # ∂Ka/∂σ_y ≈ (Ka+ - Ka-) / 2h
                dKa_dsigma = (Ka_plus - Ka_minus) / (2 * h)
                
                # ∂σ_y/∂y ≈ y / (σ_y × N_window)
                dsigma_dy = y / (self.current_rms * len(self.displacement_history))
                
                # ∂C/∂Ka
                dC_dKa = (self.rho * self.d_ref**2 / self.m_eq) * \
                         2 * np.pi * self.f_n * self.M_n
                
                # Chain rule
                dC_dy = dC_dKa * dKa_dsigma * dsigma_dy
            else:
                dC_dy = 0.0
        
        return dC_dy
    
    def set_current_time(self, t: float):
        """
        Update current simulation time (for tracking/debugging).

        Parameters:
        -----------
        t : float
            Current time [s]
        """
        self.current_time = t
        self.time_step += 1

    def get_current_rms(self) -> float:
        """
        Get current RMS displacement from running window.

        Returns:
        --------
        float
            Current RMS [m]
        """
        return self.current_rms
    
    def get_current_ka(self) -> float:
        """
        Get current Ka value.
        
        Returns:
        --------
        float
            Current aerodynamic damping parameter Ka [-]
        """
        return self.Ka_current
    
    def get_damping_info(self, y: float, t: float) -> Dict[str, float]:
        """
        Get detailed damping information at current state.
        
        Useful for debugging, visualization, and analysis.
        
        Parameters:
        -----------
        y : float
            Current displacement [m]
        t : float
            Current time [s]
            
        Returns:
        --------
        dict
            Dictionary with damping details:
            - time: Current time [s]
            - displacement: Current displacement [m]
            - rms: Current RMS [m]
            - Ka: Current Ka value [-]
            - C_a: Aerodynamic damping coefficient [Ns/m]
            - zeta_a: Aerodynamic damping ratio [-]
            - zeta_s: Structural damping ratio [-]
            - zeta_total: Total damping ratio [-]
            - window_fill: Fraction of window filled [0-1]
            - y_over_d: Normalized displacement [-]
            - rms_over_d: Normalized RMS [-]
        """

        # Aerodynamic damping coefficient
        C_a = (self.rho * self.d_ref**2 / self.m_eq) * \
              2 * np.pi * self.f_n * self.M_n * self.Ka_current
        
        # Aerodynamic damping ratio
        zeta_a = (self.rho * self.d_ref**2 / self.m_eq) * self.Ka_current
        zeta_s = self.delta_s / (2 * np.pi)

        zeta_total = zeta_s - zeta_a

        return {
            'time': t,
            'displacement': y,
            'rms': self.current_rms,
            'Ka': self.Ka_current,
            'C_a': C_a,
            'zeta_a': zeta_a,
            'zeta_s': zeta_s,
            'zeta_total': zeta_total,
            'window_fill': len(self.displacement_history) / self.window_length,
            'y_over_d': y / self.d_ref,
            'rms_over_d': self.current_rms / self.d_ref,
            'formulation': self.damping_formulation.get_name()
        }
    
    def reset(self):
        """
        Reset the damping calculator (clear history).

        Useful for running multiple simulations with same structure.
        """
        self.displacement_history.clear()
        self.current_rms = 0.0
        self.current_time = 0.0
        self.time_step = 0
        self.Ka_current = self.Ka_r_n


def create_time_domain_damping(
    structure: StructureProperties,
    config: VIVAnalysisConfig,
    d_ref: float,
    dt: float,
    window_cycles: float = 2.0
) -> TimeDomainDamping:
    """
    Factory function to create time-domain damping wrapper.

    This calculates all necessary parameters from structure and config,
    and creates the generic damping object.
    
    Parameters:
    -----------
    structure : StructureProperties
        Structure properties
    config : VIVAnalysisConfig
        Analysis configuration (must include damping_formulation)
    d_ref : float
        Reference diameter [m]
    dt : float
        Time step [s]
    window_cycles : float
        Number of natural periods for RMS window (default: 2.0)
        
    Returns:
    --------
    TimeDomainDamping
        Configured generic damping calculator
    """

    from calculators.ka_integrator import KaHeightIntegrator
    from calculators.base_calculator import BaseCalculator

    # Get basic parameters
    h = structure.height
    f_n = structure.f_n
    m_eq = structure.m_eq

    # Calculate modal mass
    mode_shape_integral = config.mode_shape.compute_modal_integral(h, mode_number=1)
    M_n = m_eq * mode_shape_integral

    # Calculate height-integrated Ka_r,n
    ka_integrator = KaHeightIntegrator(St=config.St)      
    base_calc = BaseCalculator(St=config.St)              

    # Discretize height
    n_points = 100
    z_points = np.linspace(0, h, n_points)

    # Get height-dependent properties
    d_z = config.cross_section.get_diameter(z_points, h, structure.diameter)
    u_z = config.wind_profile.get_velocity(z_points, h=h)
    phi_z = config.mode_shape.evaluate(z_points, h, mode_number=1)
    Re_z = base_calc.calculate_reynolds_number(u_z, d_z)
    Ka_max_z = config.damping_parameter.get_ka_max(Re_z)

    # Calculate integrated Ka_max (without turbulence modification)
    Ka_max_modal = ka_integrator.calculate_integrated_ka(
        structure, config, z_points, d_z, u_z, phi_z,
        Ka_max_z, d_ref
    )

    # Get damping formulation from config
    damping_formulation = config.damping_formulation

    # Create and return generic damping object
    return TimeDomainDamping(
        structure=structure,
        config=config,
        damping_formulation=damping_formulation,
        Ka_r_n=Ka_max_modal, 
        d_ref=d_ref,
        f_n=f_n,
        m_eq=m_eq,
        M_n=M_n,
        dt=dt,
        delta_s=structure.delta_s,
        window_cycles=window_cycles
    )

# Analytical derivatives for common formulations

def vickery_basu_derivative(sigma_y: float, Ka_r_n: float, a_L: float, d_ref: float) -> float:
    """
    Analytical derivative for Vickery-Basu damping.

    Ka(σ_y) = Ka_r,n × [1 - (σ_y/(a_L×d₁))²]
    ∂Ka/∂σ_y = -2 × Ka_r,n × σ_y / (a_L×d₁)²
    """
    return -2 * Ka_r_n * sigma_y / (a_L * d_ref)**2

def eurocode_derivative(sigma_y: float, Ka_r_n: float, a_L: float, d_ref: float, theta: float) -> float:
    """
    Analytical derivative for Eurocode damping.

    Ka(σ_y) = Ka_r,n × [1 - (σ_y/(a_L×d₁))^θ]
    ∂Ka/∂σ_y = -θ × Ka_r,n × (σ_y/(a_L×d₁))^(θ-1) / (a_L×d₁)
    """
    ratio = sigma_y / (a_L * d_ref)
    if ratio > 0:
        return -theta * Ka_r_n * ratio**(theta - 1) / (a_L * d_ref)
    else:
        return 0.0
    
def lupi_derivative(sigma_y: float, a: float, b: float, c: float, d_ref: float) -> float:
    """
    Analytical derivative for Lupi damping.

    Ka(σ_y) = a × exp(-b×σ_y/d₁) / (σ_y/d₁)^c
    ∂Ka/∂σ_y = -a × exp(-b×σ/d) × [b/d + c/(σ×d)] / (σ/d)^c
    """
    if sigma_y < 1e-10:
        return 0.0  # Avoid division by zero
    
    ratio = sigma_y / d_ref
    exp_term = np.exp(-b * ratio)

    # ∂Ka/∂σ_y = Ka × [-b/d - c/(σ×d)]
    return -a * exp_term * (b / d_ref + c / (sigma_y * d_ref)) / ratio**c