# config/damping_model.py
"""
Damping Model Configuration
============================

Configuration for aerodynamic damping models (Vickery & Basu, Lupi, etc.).
Defines which damping formulation to use and model-specific parameters.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


# ============================================================================
# BASE CLASS
# ============================================================================

class DampingModel(ABC):
    """
    Abstract base class for aerodynamic damping models.
    
    Defines the interface for damping model configuration.
    """

    @abstractmethod
    def get_model_name(self) -> str:
        """
        Get the name of the damping model.

        Returns:
        --------
        str
            Model name ('vickery_basu', 'lupi', etc.)
        """
        pass
    
    @abstractmethod
    def requires_height_integration(self) -> bool:
        """
        Check if model requires numerical integration over height.

        Returns:
        --------
        bool
            True if height integration needed (Lupi), False otherwise (V&B)
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
    
# ============================================================================
# CONCRETE IMPLEMENTATIONS
# ============================================================================

@dataclass
class VickeryBasuDamping(DampingModel):
    """
    Vickery & Basu (1983) aerodynamic damping model.
    
    Time-domain formulation: F_a(t) = Ka * y_dot * [1 - (y/y_L)²]
    Frequency-domain: Ka_eq = Ka_0 * [1 - ε/2 * (σ_y/D)²]
    
    Parameters based on CICIND Model Code implementation.
    """
    
    # RMS calculation parameters (for time-domain)
    rms_window_cycles: float = 2.0  # Number of periods for RMS window

    # Reference wind speed location
    reference_location: str = 'top'  # 'top', 'z_ref', 'average', 'custom'
    custom_reference_height: Optional[float] = None  # [m], if 'custom'

    # Convergence parameters for nonlinear time integration
    max_iterations: int = 12
    convergence_tolerance: float = 1e-8

    def get_model_name(self) -> str:
        return 'vickery_basu'
    
    def requires_height_integration(self) -> bool:
        return False  # V&B uses properties at reference height
    
    def get_reference_height(self, structure_height: float, 
                            z_ref: float) -> float:
        """
        Determine reference height for wind speed evaluation.
        
        Parameters:
        -----------
        structure_height : float
            Total structure height [m]
        z_ref : float
            Reference height from wind profile [m]
            
        Returns:
        --------
        float
            Reference height to use [m]
        """
        if self.reference_location == 'top':
            return structure_height
        elif self.reference_location == 'z_ref':
            return z_ref
        elif self.reference_location == 'average':
            return structure_height / 2.0
        elif self.reference_location == 'custom':
            if self.custom_reference_height is None:
                raise ValueError("custom_reference_height must be set")
            return self.custom_reference_height
        else:
            raise ValueError(f"Unknown reference_location: {self.reference_location}")
    
    def __repr__(self) -> str:
        return (f"VickeryBasuDamping(ref={self.reference_location}, "
                f"rms_window={self.rms_window_cycles:.1f} cycles)")

@dataclass
class LupiDamping(DampingModel):
    """
    Lupi et al. (2018, 2019) aerodynamic damping model.
    
    Height-integrated formulation with variable diameter and wind profile.
    Requires numerical integration of Ka over structure height.
    
    Based on CICIND Commentaries implementation.
    """

    # Integration parameters
    num_integration_points: int = 1000  # Number of height points

    # Reference wind speed location
    reference_location: str = 'top'  # 'top', 'z_ref', 'average', 'custom'
    custom_reference_height: Optional[float] = None  # [m], if 'custom'
    
    # Optional: Lupi-specific parameters
    use_modified_ka_formula: bool = False  # If True, use Lupi's modified Ka
    
    def get_model_name(self) -> str:
        return 'lupi'
    
    def requires_height_integration(self) -> bool:
        return True  # Lupi integrates Ka over height
    
    def get_reference_height(self, structure_height: float,
                            z_ref: float) -> float:
        """Determine reference height (same logic as V&B)."""
        if self.reference_location == 'top':
            return structure_height
        elif self.reference_location == 'z_ref':
            return z_ref
        elif self.reference_location == 'average':
            return structure_height / 2.0
        elif self.reference_location == 'custom':
            if self.custom_reference_height is None:
                raise ValueError("custom_reference_height must be set")
            return self.custom_reference_height
        else:
            raise ValueError(f"Unknown reference_location: {self.reference_location}")
    
    def __repr__(self) -> str:
        return (f"LupiDamping(ref={self.reference_location}, "
                f"n_points={self.num_integration_points})")
    
@dataclass
class CustomDamping(DampingModel):
    """
    Custom damping model for research/development.
    
    Allows user to implement their own damping formulation.
    """
    
    model_name: str = 'custom'
    requires_integration: bool = False
    custom_parameters: dict = None
    
    def __post_init__(self):
        if self.custom_parameters is None:
            self.custom_parameters = {}
    
    def get_model_name(self) -> str:
        return self.model_name
    
    def requires_height_integration(self) -> bool:
        return self.requires_integration
    
    def __repr__(self) -> str:
        return f"CustomDamping(name={self.model_name})"


# ============================================================================
# FACTORY FUNCTIONS (for convenience)
# ============================================================================

def create_vickery_basu(rms_window_cycles: float = 2.0,
                        reference_location: str = 'top') -> VickeryBasuDamping:
    """
    Create Vickery & Basu damping configuration.
    
    Parameters:
    -----------
    rms_window_cycles : float
        Number of natural periods for RMS calculation window
    reference_location : str
        Where to evaluate wind speed: 'top', 'z_ref', 'average'
        
    Returns:
    --------
    VickeryBasuDamping
        Configured damping model
    """
    return VickeryBasuDamping(
        rms_window_cycles=rms_window_cycles,
        reference_location=reference_location
    )


def create_lupi(num_integration_points: int = 1000,
                reference_location: str = 'top') -> LupiDamping:
    """
    Create Lupi damping configuration.
    
    Parameters:
    -----------
    num_integration_points : int
        Number of height points for numerical integration
    reference_location : str
        Where to evaluate wind speed: 'top', 'z_ref', 'average'
        
    Returns:
    --------
    LupiDamping
        Configured damping model
    """
    return LupiDamping(
        num_integration_points=num_integration_points,
        reference_location=reference_location
    )

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("Damping Model Configuration - Demo")
    print("=" * 60)
    
    # Example 1: Vickery & Basu (default)
    print("\n1. Vickery & Basu Damping (default):")
    damping1 = VickeryBasuDamping()
    print(f"   {damping1}")
    print(f"   Model name: {damping1.get_model_name()}")
    print(f"   Requires height integration: {damping1.requires_height_integration()}")
    print(f"   Reference height (h=60m): {damping1.get_reference_height(60, 10):.1f} m")
    
    # Example 2: Vickery & Basu (custom settings)
    print("\n2. Vickery & Basu Damping (custom):")
    damping2 = VickeryBasuDamping(
        rms_window_cycles=3.0,
        reference_location='average',
        max_iterations=20
    )
    print(f"   {damping2}")
    print(f"   RMS window: {damping2.rms_window_cycles} cycles")
    print(f"   Reference height (h=60m): {damping2.get_reference_height(60, 10):.1f} m")
    
    # Example 3: Lupi damping
    print("\n3. Lupi Damping:")
    damping3 = LupiDamping(
        num_integration_points=500,
        reference_location='top'
    )
    print(f"   {damping3}")
    print(f"   Model name: {damping3.get_model_name()}")
    print(f"   Requires height integration: {damping3.requires_height_integration()}")
    print(f"   Integration points: {damping3.num_integration_points}")
    
    # Example 4: Factory functions
    print("\n4. Using Factory Functions:")
    damping4 = create_vickery_basu(rms_window_cycles=2.5)
    damping5 = create_lupi(num_integration_points=1000)
    print(f"   {damping4}")
    print(f"   {damping5}")
    
    # Example 5: Custom damping
    print("\n5. Custom Damping Model:")
    damping6 = CustomDamping(
        model_name='experimental',
        requires_integration=True,
        custom_parameters={'param1': 1.5, 'param2': 'test'}
    )
    print(f"   {damping6}")
    print(f"   Parameters: {damping6.custom_parameters}")
    
    print("\n" + "=" * 60)
    print("Damping models define HOW aerodynamic damping is calculated.")
    print("Compatible with both constant and height-varying properties.")