# calculators/structure_properties_calculator.py
"""
Structure Properties Calculator
===============================

Calculates derived properties that depend on configuration parameters
like reference diameter and Strouhal number.

References:
- E. Simon, "Development and Application of a Time-Domain Simulation Tool for Spectral Modeling of Vortex-Induced Vibrations", Master's Thesis, RWTH Aachen, 2025.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from common.structures import StructureProperties
from config import VIVAnalysisConfig


@dataclass
class ComputedStructureProperties:
    """Container for computed structure properties."""
    # Input parameters
    d_nominal: float
    d_ref: float
    St: float

    # Geometric properties
    aspect_ratio: float

    # Dynamic properties
    scruton_number: float
    u_crit: float
    modal_mass: float

class StructurePropertiesCalculator:
    """
    Calculator for derived structure properties that depend on
    configuration parameters.
    """  

    def __init__(self, rho_air: float = 1.25):
        """Initialize calculator."""
        self.rho_air = rho_air

    def calculate_scruton_number(self, m_eq: float, delta_s: float, d: float) -> float:
        """
        Calculate Scruton number.

        Sc = 2 * m_eq * delta_s / (rho * d²)

        Parameters:
        -----------
        m_eq : float
            Equivalent mass [kg/m]
        delta_s : float
            Logarithmic damping decrement [-]
        d : float
            Diameter [m]
        
        Returns:
        --------
        float
            Scruton number [-]
        """
        # Simon (2025), Eq. 3.23
        return 2 * m_eq * delta_s / (self.rho_air * d**2)
    
    def calculate_critical_velocity(self, f_n: float, d: float, St: float) -> float:
        """
        Calculate critical wind velocity.

        u_crit = f_n * d / St
        
        Parameters:
        -----------
        f_n : float
            Natural frequency [Hz]
        d : float
            Diameter [m]
        St : float
            Strouhal number [-]
        
        Returns:
        --------
        float
            Critical velocity [m/s]
        """
        # Simon (2025), Eq. 2.9
        return f_n * d / St
    
    def calculate_modal_mass(
        self,
        mode_shape_integral: float,
        m_eq: float
    ) -> float:
        """
        Calculate modal mass.

        Parameters:
        -----------
        m_eq: float
            Equivalent mass [kg/m]
        """
        return m_eq * mode_shape_integral
    
    def compute_all_properties(
        self,
        structure: StructureProperties,
        d_ref: float,
        St: float,
        m_eq: float,
        mode_shape_integral: float
    ) -> ComputedStructureProperties:
        """
        Compute all derived properties for a structure.

        Parameters:
        -----------
        structure : StructureProperties
            Structure with basic properties
        d_ref : float
            Reference diameter [m]
        St : float
            Strouhal number [-]
        
        Returns:
        --------
        ComputedStructureProperties
            All computed properties
        """
        # Geometric properties
        aspect_ratio = structure.height / d_ref

        # Scruton number
        Sc = self.calculate_scruton_number(
            structure.m_eq, structure.delta_s, d_ref
        )

        # Critical velocities
        u_crit_ref = self.calculate_critical_velocity(
            structure.f_n, d_ref, St
        )

        # Modal mass
        modal_mass = self.calculate_modal_mass(
            mode_shape_integral, m_eq
        )

        return ComputedStructureProperties(
            d_nominal=structure.diameter,
            d_ref=d_ref,
            St=St,
            aspect_ratio=aspect_ratio,
            scruton_number=Sc,
            u_crit=u_crit_ref,
            modal_mass=modal_mass
        )
    
def print_structure_summary(
    structure: StructureProperties,
    computed: ComputedStructureProperties
) -> None:
    """
    Print formatted summary of structure properties.
    """
    print(f"\nSTRUCTURE SUMMARY: {structure.name}")
    print("=" * (len(structure.name) + 18))

    print("GEOMETRIC PROPERTIES:")
    print(f"  Height:                       {structure.height:8.1f} [m]")
    print(f"  Nominal diameter (d_nom):     {computed.d_nominal:8.3f} [m]")
    print(f"  Reference diameter (d_ref):   {computed.d_ref:8.3f} [m]")
    print(f"  Aspect ratio (h/d_ref):       {computed.aspect_ratio:8.1f} [-]")

    print("\nDYNAMIC PROPERTIES:")
    print(f"  Natural frequency:            {structure.f_n:8.3f} [Hz]")
    print(f"  Equivalent mass:              {structure.m_eq:8.1f} [kg/m]")
    print(f"  Modal mass:                   {computed.modal_mass:8.1f} [kg]")
    print(f"  Damping decrement:            {structure.delta_s:8.4f} [-]")
    print(f"  Damping ratio:                {structure.zeta_s:8.4f} [-]")
    print(f"  Scruton number (d_ref):       {computed.scruton_number:8.2f} [-]")
    print(f"  Critical velocity (d_ref):    {computed.u_crit:8.2f} [m/s]")

    # Measured properties if available
    if structure.measured_y_d or structure.measured_y_d_rare is not None:
        print("\nMEASURED RESPONSE:")
        if structure.measured_y_d is not None:
            print(f"  Measured normalized response (y/d): {structure.measured_y_d:8.4f} [-]")
        if structure.measured_y_d_rare is not None:
            print(f"  Measured normalized response (rare event) (y/d): {structure.measured_y_d_rare:8.4f} [-]")

    print("\nPHYSICAL CONSTANTS:")
    print(f"  Air density:                  {structure.rho_air:8.3f} [kg/m³]")
    print(f"  Kinematic viscosity:          {structure.nu_air:8.2e} [m²/s]")