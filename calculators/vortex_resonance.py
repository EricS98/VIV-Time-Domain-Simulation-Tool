# calculators/vortex_resonance.py
"""
Vortex-Resonance Model Analysis
===============================

Calculation of vortex-induced vibrations using the vortex-resonance model.

References:
- H. Ruscheweyh, "Vortex Excited Vibrations" in Wind-Excited Vibrations of Structures, pages 51-84, 1994.
- H. Ruscheweyh, "Ein verfeinertes, praxisnahes Berechnungsverfahren wirbelerregter Schwingungen von schlanken Baukonstruktionen im Wind" in Beiträge zur Anwendung der Aeroelastik im Bauwesen, 1987.
- DIN EN 1991-1-4:2010-12, Eurocode 1: Einwirkungen auf Tragwerke - Teil 1-4: Allgemeine Einwirkungen - Windlasten.
- M. Clobes, A. Willecke, and U. Peil, "Wirbelerregung von Stahlschornsteinen: Zwei Grenzzustände der Tragfähigkeit und Vorschlag für die Bemessung" in Bauingenieur 87, 2012.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.integrate import quad

from common.structures import StructureProperties
from common.math_utils import loglin
from calculators.base_calculator import BaseCalculator
from config.analysis_config import VIVAnalysisConfig

@dataclass
class VortexResonanceResults:
    """Results from vortex-resonance calculation."""

    # Basic parameters
    structure_name: str
    height: float
    d_ref: float  # Reference diameter
    natural_frequency: float
    equivalent_mass: float
    delta_s: float
    v_crit: float   # Critical velocity at reference diameter
    Re: float  # Reynolds number at critical velocity
    Scruton_number: float
    zeta_s: float   # Structural damping ratio

    # Physical parameters
    rho_air: float
    nu_air: float

    # Model-specific coefficients
    c_lat: float
    Kw: float
    K_xi: float
    Le_d: float

    # Response results
    y_max: float
    y_max_over_d: float

    # Configuration details
    strouhal_number: float
    mode_exponent: Optional[float] = None

    def print_summary(self):
        """Print calculation results summary."""
        print(f"\nVortex-Resonance Model Results - {self.structure_name}")
        print(f"  Reference diameter: d_ref = {self.d_ref:.3f} m")
        print(f"  Strouhal number: St = {self.strouhal_number:.3f}")
        print(f"  Reynolds number: Re = {self.Re:.2e}")
        print(f"  Critical velocity: v_crit = {self.v_crit:.2f} m/s")
        print(f"  Scruton number: Sc = {self.Scruton_number:.2f}")
        print(f"  Lateral force coefficient: c_lat = {self.c_lat:.4f}")
        print(f"  Correlation factor: Kw = {self.Kw:.3f}")
        print(f"  Mode shape factor: K_xi = {self.K_xi:.3f}")
        if self.mode_exponent is not None:
            print(f"  Mode shape exponent: n = {self.mode_exponent:.1f}")
        print(f"  Effective correlation length: Le/d_ref = {self.Le_d:.2f}")
        print(f"  Maximum response: y_max = {self.y_max*1000:.2f} mm")
        print(f"  Response ratio: y_max/d_ref = {self.y_max_over_d:.4f}")
    
    def to_dict(self) -> Dict:
        """Convert results to dictionary."""
        return {
            'structure_name': self.structure_name,
            'height': self.height,
            'd_ref': self.d_ref,
            'natural_frequency': self.natural_frequency,
            'equivalent_mass': self.equivalent_mass,
            'strouhal_number': self.strouhal_number,
            'delta_s': self.delta_s,
            'v_crit': self.v_crit,
            'Re': self.Re,
            'Scruton_number': self.Scruton_number,
            'zeta_s': self.zeta_s,
            'c_lat': self.c_lat,
            'Kw': self.Kw,
            'K_xi': self.K_xi,
            'Le_d': self.Le_d,
            'y_max': self.y_max,
            'y_max_over_d': self.y_max_over_d,
            'mode_exponent': self.mode_exponent
        }
    
class VortexResonanceCalculator(BaseCalculator):
    """
    Calculator for vortex-resonance model VIV analysis.
    """

    def __init__(self, St: float=0.2, rho_air: float = 1.25, nu_air: float = 1.5e-5,
                 use_willecke_peil: Optional[bool] = False, manual_kw: float = None):
        """
        Initialize the vortex-resonance calculator.
        
        Parameters:
        -----------
        St : float
            Strouhal number [-]
        rho_air : float
            Air density [kg/m³] (can be manually set for stable atmospheric conditions)
        nu_air : float
            Kinematic viscosity [m²/s] (can be manually set for stable atmospheric conditions)
        use_willecke_peil : bool
            If True, use Willecke-Peil extension for stable atmospheric conditions
        manual_kw : float, optional
            Manual override for Kw factor (e.g., 0.95 for extreme events)
            If None, uses the calculated Kw with 0.6 limit
        """
        super().__init__(St=St, rho_air=rho_air, nu_air=nu_air)
        # Clobes et al. (2012) extension for stable atmospheric conditions
        self.use_willecke_peil = use_willecke_peil
        self.manual_kw = manual_kw

    def calculate_clat_re(self, Re: float):
        """
        Calculate lateral force coefficient c_lat based on the Reynolds number.
        
        Uses Eurocode/CICIND interpolation scheme.
        
        Parameters:
        -----------
        Re: float
            Reynolds number
        
        Returns:
        --------
        float: Lateral force coefficient c_lat
        """
        if Re <= 0:
            raise ValueError("Reynolds number must be positive.")
        
        # Eurocode curve (Ruscheweyh (1994), Fig. 2.4; Eurocode 1-4 (2010), Fig. E.2)
        if Re <= 3e5:
            return 0.70
        elif Re <= 5e5:
            # Log-linear interpolation
            return loglin(Re, 3e5, 0.70, 5e5, 0.20)
        elif Re <= 5e6:
            return 0.20
        elif Re <= 1e7:
            # Log-linear interpolation
            return loglin(Re, 5e6, 0.20, 1e7, 0.30)
        else:
            return 0.30
        
    def calculate_kw(self, Le_d: float, aspect_ratio: float) -> float:
        """
        Calculate correlation factor K_w for a cantilever in 1st mode.

        Parameters:
        -----------
        Le_d : float
            Effective correlation length Le/d_ref
        aspect_ratio : float
            Height-to-diameter ratio h/d_ref
            
        Returns:
        --------
        float
            Correlation factor Kw, limited to 0.6
        """
        # If manual Kw is set, use it directly without limit
        if self.manual_kw is not None:
            return self.manual_kw
        
        # Otherwise calculate normally
        le_ar_ratio = Le_d / aspect_ratio
        if le_ar_ratio >= 1.0:
            return 1.0

        # Ruscheweyh (1994), Eq. 2.21
        # Applicable for cantilever in 1st mode
        kw = 3 * le_ar_ratio * (1 - le_ar_ratio + (1/3) * le_ar_ratio**2)

        # Apply upper theshold of 0.6
        return min(kw, 0.6)
    
    def get_effective_correlation_length(self, y_d: float) -> float:
        """
        Calculate L_e/d_ref based on the amplitude ratio y/d_ref.
        
        Parameters:
        -----------
        y_d : float
            Amplitude ratio y/d_ref
            
        Returns:
        --------
        float
            Effective correlation length Le/d_ref
        """
        if y_d <= 0:
            raise ValueError("Amplitude ratio y/d_ref must be positive.")
        
        if self.use_willecke_peil:
            # Willecke-Peil extension for stable atmospheric conditions
            # Clobes et al. (2012), Tab. 1
            if y_d <= 0.05:
                return 12.0
            elif y_d >= 0.20:
                return 30.0
            else:
                # Linear interpolation between 0.05 and 0.20
                return 12.0 + (30.0 - 12.0) * (y_d - 0.05) / (0.20 - 0.05)
        
        else:
            # Standard Ruscheweyh correlation
            # Ruscheweyh (1994), Tab. 2.1, Eq. 2.16
            if y_d <= 0.1:
                return 6.0
            elif y_d < 0.6:
                return 4.8 + 12 * y_d
            else: 
                return 12.0
        
    def calculate_amplitude(
        self,
        structure: StructureProperties,
        config: VIVAnalysisConfig,
        d_ref: float,
        max_iter: int = 10,
        tol: float = 1e-4
    ) -> VortexResonanceResults:
        """
        Calculate vibration amplitude using iterative procedure.

        Parameters:
        -----------
        structure: StructureProperties
            Structure properties
        config: VIVAnalysisConfig
            Analysis configuration
        d_ref: float
            Reference diameter [m]
        max_iter: int
            Maximum number of iterations
        tol: float
            Tolerance for convergence
        
        Returns:
        --------
        VortexResonanceResults: Complete calculation results
        """
        # Scruton number based on reference diameter
        from calculators.structure_properties_calculator import StructurePropertiesCalculator
        calc = StructurePropertiesCalculator(rho_air=self.rho_air)
        Sc = calc.calculate_scruton_number(structure.m_eq, structure.delta_s, d_ref)

        # Use configuration St
        St = self.St
        print(f"ℹ️  Strouhal number: St = {St:.3f}")

        # Critical velocity at reference diameter
        v_crit = structure.f_n * d_ref / St
        print(f"ℹ️  Critical velocity: v_crit = {v_crit:.2f} m/s")

        # Calculate Reynolds number at critical velocity
        Re = v_crit * d_ref / self.nu_air
        print(f"ℹ️  Reynolds number: Re = {Re:.2e}")
        
        # Calculate lateral force coefficient
        c_lat = self.calculate_clat_re(Re)
        print(f"ℹ️  Lateral force coefficient: c_lat = {c_lat:.3f}")

        # Calculate K_xi from mode shape
        K_xi = config.mode_shape.compute_k_xi(structure.height, mode_number=1)
        print(f"ℹ️  Mode shape factor: K_xi = {K_xi:.4f}")

        # Aspect ratio
        aspect_ratio = structure.height / d_ref

        if self.manual_kw is not None:
            # No iteration needed - direct calculation
            Kw = self.manual_kw
            # Ruscheweyh (1994), Eq. 2.17
            y_d = (K_xi * Kw * c_lat) / (Sc * St**2)
            Le_d = self.get_effective_correlation_length(y_d)
            print(f"ℹ️  Using manual Kw = {Kw:.3f} (no iteration needed)")
            print(f"✅ Direct calculation completed.")
        else:
            # Step 1: Initial guess for L_e/d_ref
            Le_d = 6.0  
        # Iterative calculation
        for i in range(max_iter):
            # Step 2: Calculate K_w based on current L_e/d_ref
            Kw = self.calculate_kw(Le_d, aspect_ratio)
            
            # Step 3: Calculate the resulting amplitude ratio y/d_ref
            y_d = (K_xi * Kw * c_lat) / (Sc * St**2)
            
            # Step 4: Update L_e/d_ref based on new y/d_ref
            Le_d_new = self.get_effective_correlation_length(y_d)
            
            # Step 5: Check for convergence
            if abs(Le_d_new - Le_d) < tol:
                Le_d = Le_d_new  # Update for final iteration
                print(f"✅ Converged in {i+1} iterations.")
                break
            Le_d = Le_d_new
        else:
            print(f"⚠️  Warning: No convergence after {max_iter} iterations.")

        # Final amplitude
        if self.manual_kw is None:
            Kw_final = self.calculate_kw(Le_d, aspect_ratio)
        else:
            Kw_final = self.manual_kw

        # Print Willecke-Peil info if applicable
        if self.use_willecke_peil:
            print(f"ℹ️  Using Willecke-Peil extension for stable atmospheric conditions")
        if self.manual_kw is not None:
            print(f"ℹ️  Using manual Kw = {self.manual_kw:.3f} (no 0.6 limit applied)")

        y_d_final = (K_xi * Kw_final * c_lat) / (Sc * St**2)
        y_max = y_d_final * d_ref

        # Extract mode exponent if available
        mode_exponent = None
        if hasattr(config.mode_shape, 'exponent'):
            mode_exponent = config.mode_shape.exponent
        
        # Return results
        return VortexResonanceResults(
            # Basic parameters
            structure_name=structure.name,
            height=structure.height,
            d_ref=d_ref,
            natural_frequency=structure.f_n,
            equivalent_mass=structure.m_eq,
            strouhal_number=St,
            delta_s=structure.delta_s,
            v_crit=v_crit,
            Re=Re,
            Scruton_number=Sc,
            zeta_s=structure.zeta_s,
            # Physical parameters
            rho_air=self.rho_air,
            nu_air=self.nu_air,
            # Model-specific coefficients
            c_lat=c_lat,
            Kw=Kw_final,
            K_xi=K_xi,
            Le_d=Le_d,
            # Results
            y_max=y_max,
            y_max_over_d=y_d_final,
            # Configuration
            mode_exponent=mode_exponent
        )