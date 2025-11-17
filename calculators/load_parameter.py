# calculators/load_parameter.py
"""
Load Parameter Calculator Module
================================

Calculates the generalized load parameter C_an. 
"""

import numpy as np
from common.structures import StructureProperties
from config import VIVAnalysisConfig
from calculators.base_calculator import BaseCalculator

class LoadParameterCalculator(BaseCalculator):
    """
    Calculator for the load parameter C_an. 
    """
    def __init__(self):
        """Initialize load parameter calculator."""
        super().__init__()

    def calculate_analytical(self, structure: StructureProperties,
                            config: VIVAnalysisConfig,
                            d: float, u: float,
                            lambda_corr: float, 
                            St: float,
                            sigma_CL: float, B: float) -> float:
        """
        Calculate load parameter C_an analytically (constant properties).
        
        Uses closed-form solution for constant diameter and wind profile.
        
        Formula (Vickery & Basu):
        C_an = (1/(2√π * 8 * (2π)³)) * (λ * σ²_CL) / (St⁴ * B * ∫φ²dz) * (ρ * d³ / m_eq)
        
        Parameters:
        -----------
        structure : StructureProperties
            Structure properties
        config : VIVAnalysisConfig
            Analysis configuration
        d : float
            Reference diameter [m]
        u : float
            Reference wind velocity [m/s] (not used in analytical form)
        lambda_corr : float
            Correlation length factor [-] (typically 1.0)
        St : float
            Strouhal number [-]
        sigma_CL : float
            RMS lift coefficient [-]
        B : float
            Bandwidth parameter [-]
            
        Returns:
        --------
        float
            Generalized load parameter C_an
        """
        # Validate inputs
        if d <= 0:
            raise ValueError("Diameter d must be positive.")
        if u <= 0:
            raise ValueError("Wind speed u must be positive.")
        if sigma_CL <= 0:
            raise ValueError("Lift coefficient sigma_CL must be positive.")
        if B <= 0:
            raise ValueError("Bandwidth B must be positive.")

        h = structure.height
        d = structure.diameter
        m_eq = structure.m_eq
        rho_air = self.rho_air
        
        # Mode shape integral
        mode_integral = config.mode_shape.compute_modal_integral(h, mode_number=1)
        
        # Generalized load parameter
        C_an = (1/(2*np.sqrt(np.pi)*8*(2*np.pi)**3) * 
                lambda_corr * sigma_CL**2 / (St**4 * B * mode_integral) * 
                rho_air * d**3 / m_eq)
        
        return C_an
    
    def calculate_integrated(self, config: VIVAnalysisConfig, 
                            z_points: np.ndarray,
                            d_z: np.ndarray, u_z: np.ndarray,
                            phi_z: np.ndarray, 
                            lambda_corr: float,
                            St: float,
                            sigma_cl_z: np.ndarray,
                            B_z: np.ndarray, 
                            structure: StructureProperties) -> float:
        """
        Calculate load parameter C_an with numerical integration.
        
        Integrates the generalized force spectrum over height for structures
        with varying diameter, wind profile, or aerodynamic properties.
        
        Formula:
        C_an = σ²_Qn / (ω_n² * M_n²)
        
        Where:
        σ²_Qn = 2λd_ref * ∫₀ʰ S_L(f_n, z) * φ²(z) dz
        S_L(f_n, z) = [0.5 * ρ * u(z)² * d(z)]² * [σ²_CL(z) / (√π * B(z) * f_s(z))]
        f_s(z) = St * u(z) / d(z)
        
        At resonance (u = u_crit): f_s = f_n, so the spectrum evaluates at natural frequency
        
        Parameters:
        -----------
        z_points : np.ndarray
            Height discretization points [m]
        d_z : np.ndarray
            Diameter at each height [m]
        u_z : np.ndarray
            Wind velocity at each height [m/s]
        phi_z : np.ndarray
            Mode shape at each height [-]
        lambda_corr : float
            Correlation length factor [-]
        St : float
            Strouhal number [-]
        sigma_cl_z : np.ndarray
            RMS lift coefficient at each height [-]
        B_z : np.ndarray
            Bandwidth parameter at each height [-]
        structure : StructureProperties
            Structure properties
            
        Returns:
        --------
        float
            Generalized load parameter C_an
        """
        
        h = structure.height
        f_n = structure.f_n
        m_eq = structure.m_eq
        rho_air = self.rho_air
        omega_n = 2 * np.pi * f_n

        # Shedding frequency at each height
        f_s_z = St * u_z / d_z

        # Lift coefficient spectrum at each height
        S_CL_z = sigma_cl_z**2 / (np.sqrt(np.pi) * B_z * f_s_z)
        
        # Sectional lift spectrum at each height
        S_L_z = (0.5 * rho_air * u_z**2 * d_z)**2 * S_CL_z
        
        # Generalized force spectrum (integrate over height)
        d_ref = d_z[-1]
        integrand = S_L_z * phi_z**2
        sigma_Qn_squared = 2 * lambda_corr * d_ref * np.trapezoid(integrand, z_points)

        # Modal mass
        mode_integral = config.mode_shape.compute_modal_integral(h, mode_number=1)

        # Generalized load parameter
        C_an = sigma_Qn_squared / (8 * omega_n**3 * m_eq * mode_integral**2 * rho_air * d_ref)
        
        return C_an