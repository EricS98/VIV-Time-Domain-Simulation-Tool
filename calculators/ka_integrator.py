# calculators/ka_integrator.py
"""
Ka Height Integration
=====================

Handles height-integrated aerodynamic damping parameter.

IMPORTANT: Different handling for different damping models:
- Vickery-Basu/Eurocode: Integrates Ka_max(Re) * Ka_norm(U/Ucr, Iv) → Returns Ka_r,n
- Lupi DMSM: Integrates Ka_norm(U/Ucr, Iv) only → Returns K_a0,modal

References:
- E. Simon, "Development and Application of a Time-Domain Simulation Tool for Spectral Modeling of Vortex-Induced Vibrations", Master's Thesis, RWTH Aachen, 2025.
"""

import numpy as np

from common.structures import StructureProperties
from config import VIVAnalysisConfig
from config.aerodynamic_parameters.amplitude_dependent_damping import LupiDamping

class KaHeightIntegrator:
    """
    Calculate height-integrated generalized aerodynamic damping parameter.

    Formula depends on damping model:

    VICKERY-BASU / EUROCODE:
    -----------------------
    K_{a,n}(z) = K_{a,max}(Re(z)) * K_{a,norm}(U/Ucr(z), Iv(z))
    
    Ka_r,n = ∫₀ʰ K_{a,n}(z) d²(z) φ_n²(z) dz / (d_ref² ∫₀ʰ φ_n²(z) dz)
    
    Returns: Ka_r,n (used as input to amplitude-dependent Ka(σy))
    
    LUPI DMSM:
    ---------
    K_{a,norm}(z) = Ka_norm(U/Ucr(z), Iv(z))  [typically simplified: max(1-3Iv, 0.25)]
    
    K_{a0,modal} = ∫₀ʰ K_{a,norm}(z) d²(z) φ_n²(z) dz / (d_ref² ∫₀ʰ φ_n²(z) dz)
    
    Returns: K_{a0,modal} (turbulence reduction factor only, NO Ka_max!)
    
    Rationale for Lupi:
    - Lupi's envelope curve Ka(σy/d) = a·exp(-b·σy/d)/(σy/d)^c already captures 
      the Reynolds number effects and 3D behavior Ka(σy/d, V/Vcr)
    - Only turbulence effects need to be height-integrated
    - Using Ka_max(Re) would double-count Reynolds effects
    
    Parameters:
    -----------
    St : float
        Strouhal number (default: 0.2)
    """
    def __init__(self, St: float = 0.2):
        """Initialize integrator."""
        self.St = St  

    def calculate_integrated_ka(
        self,
        structure: StructureProperties,
        config: VIVAnalysisConfig,
        z_points: np.ndarray,
        d_z: np.ndarray,
        u_z: np.ndarray,
        phi_z: np.ndarray,
        Ka_max_z: np.ndarray,
        d_ref: float
    ) -> float:
        """
        Calculate height-integrated Ka parameter.

        Parameters:
        -----------
        structure : StructureProperties
            Structure properties
        config : VIVAnalysisConfig
            Analysis configuration
        z_points : np.ndarray
            Height discretization [m]
        d_z : np.ndarray
            Diameter at each height [m]
        u_z : np.ndarray
            Wind velocity at each height [m/s]
        phi_z : np.ndarray
            Mode shape at each height [-]
        Ka_max_z : np.ndarray
            Ka_max at each height point
        d_ref : float
            Reference diameter [m] (typically top diameter)
            
        Returns:
        --------
        float
            For Vickery-Basu/Eurocode: Ka_r,n
            For Lupi: K_a0,modal (turbulence reduction factor)
        """
        h = structure.height
        f_n = structure.f_n

        # Critical velocity and velocity ratios at each height
        u_crit_z = f_n * d_z / self.St
        u_ucr_z = u_z / u_crit_z

        # Turbulence intensity at each height
        Iv_z = config.wind_profile.get_turbulence_intensity(z_points, h)

        # --- Case 1: Constant properties (no height-integration) ---
        if config.wind_profile.is_constant() and config.cross_section.is_constant():
            u = u_z[-1]
            Iv = Iv_z[-1]
            u_cr = f_n * d_ref / self.St
            u_ucr = u / u_cr

            # Get turbulence reduction factor
            K_a0 = config.damping_modification.apply(
                Ka_max=1.0, Iv=Iv, u_ucr=u_ucr
            )

            if isinstance(config.damping_formulation, LupiDamping):
                # For Lupi: Return turbulence reduction factor only
                print(f"    Modal-averaged Ka reduction factor (constant profile):")
                print(f"    K_a0,modal = {K_a0:.4f} | U/Ucr = {u_ucr:.3f} | Iv = {Iv:.3f}")
                print(f"    (For Lupi: This is multiplied with envelope curve, NOT with Ka_max)")
                return K_a0
            else:
                # For Eurocode/VickeryBasu: multiply by Ka_max to get Ka_r_n
                Ka_r_n = Ka_max_z[-1] * K_a0
                print(f"    Height-integrated Ka (constant profile):")
                print(f"    Ka_max = {Ka_max_z[-1]:.3f} | K_a0 = {K_a0:.4f} | Ka_r,n = {Ka_r_n:.3f}")
                return Ka_r_n

        # --- Case 2: Varying properties (height integration required) ---

        # 1) Calculate K_a0(U/Ucr(z), Iv(z)) at each height
        K_a0_z = np.zeros_like(z_points, dtype=float)

        for i in range(len(z_points)):
            K_a0_z[i] = config.damping_modification.apply(
                Ka_max=1.0, Iv=Iv_z[i], u_ucr=u_ucr_z[i]
            )

        # 2) Compute integrals
        # Simon (2025), Eq. 3.45

        # Numerator: ∫₀ʰ K_a0(z) * d²(z) * φ_n²(z) dz
        integrand_Ka0 = K_a0_z * (d_z**2) * (phi_z**2)
        numerator = np.trapezoid(integrand_Ka0, z_points)

        # Denominator: d_ref² * ∫₀ʰ φ_n²(z) dz
        mode_shape_int = np.trapezoid(phi_z**2, z_points)
        denominator = (d_ref**2) * mode_shape_int

        # Modal-averaged Ka reduction factor
        K_a0_modal = numerator / denominator

        if isinstance(config.damping_formulation, LupiDamping):
            print(f"  Modal-averaged Ka reduction factor (height-integrated):")
            print(f"    K_a0,modal = {K_a0_modal:.4f}")
            print(f"    K_a0 range: [{K_a0_z.min():.4f}, {K_a0_z.max():.4f}]")
            print(f"    U/Ucr range: [{u_ucr_z.min():.3f}, {u_ucr_z.max():.3f}]")
            print(f"    Iv range: [{Iv_z.min():.3f}, {Iv_z.max():.3f}]")
            print(f"    (For Lupi: This is multiplied with envelope curve, NOT with Ka_max)")
            return K_a0_modal
    
        # For VickeryBasu/Eurocode: integrate Ka_max separately and multiply
        Ka_integrand = Ka_max_z * K_a0_z * d_z**2 * phi_z**2
        Ka_r_n = np.trapezoid(Ka_integrand, z_points) / denominator

        print(f"  Height-integrated Ka:")
        print(f"    Ka_max range = [{Ka_max_z.min():.3f}, {Ka_max_z.max():.3f}]")
        print(f"    K_a0 range = [{K_a0_z.min():.4f}, {K_a0_z.max():.4f}]")
        print(f"    Ka_r,n = {Ka_r_n:.3f}")
        print(f"    U/Ucr range: [{u_ucr_z.min():.3f}, {u_ucr_z.max():.3f}] | Iv range: [{Iv_z.min():.3f}, {Iv_z.max():.3f}]")

        return Ka_r_n