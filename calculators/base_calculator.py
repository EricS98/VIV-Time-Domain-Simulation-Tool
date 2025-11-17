# calculators/base_calculator.py

import numpy as np

from common.structures import StructureProperties

class BaseCalculator:
    """Base class for VIV calculators with shared constants and methods."""

    def __init__(self, St: float = 0.2, rho_air: float = 1.25, nu_air: float = 1.5e-5):
        """Initialize base calculator."""
        # Physical constants
        self.St = St            # Strouhal number [-]
        self.rho_air = rho_air  # Air density [kg/m³]
        self.nu_air = nu_air    # Kinematic viscosity [m²/s]

    def calculate_reynolds_number(self, u: float, d: float) -> float:
        """Calculate Reynolds number: Re = u*d/nu"""
        return u * d / self.nu_air

    def calculate_bandwidth(self, Iv: float) -> float:
        """
        Calculate spectrum bandwidth parameter B.
        CICIND Model Code formula: B = 0.1 + Iv, capped at 0.35

        Parameters:
        -----------
        Iv : float
            Turbulence intensity [-]
        
        Returns:
        --------
        float
            Bandwidth parameter B [-]
        """
        return min(0.1 + Iv, 0.35)

    def calculate_peak_factor(
        self, sc_over_4pi_ka: float,
        formula: str = "cicind"
    ) -> float:
        """
        Calculate peak factor for converting RMS to peak response.

        CICIND/Eurocode formula:
        k_p = sqrt(2) * [1 + 1.2 * arctan(0.75 * (Sc/(4πKa))^4)]
        
        Parameters:
        -----------
        sc_over_4pi_ka : float
            Ratio Sc/(4πKa), typically c1 from quadratic equation
        formula : str, optional
            Formula to use: 'cicind' (default)
            
        Returns:
        --------
        float
            Peak factor k_p
        """
        if formula == "cicind":
            # CICIND/Eurocode formula
            kp = np.sqrt(2) * (1 + 1.2 * np.arctan(0.75 * sc_over_4pi_ka**4))

        else:
            raise ValueError(f"Unknown peak factor formula: {formula}")
    
        return kp