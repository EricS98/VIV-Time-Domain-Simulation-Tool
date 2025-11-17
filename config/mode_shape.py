# config/mode_shape.py
"""
Mode Shape Configuration
=========================

Abstract base classes for structural mode shapes.
Supports fundamental mode, multiple modes, and custom definitions.

Design principle: Inheritance-based, easily extensible.
"""

from abc import ABC, abstractmethod
from typing import Union, List, Optional
import numpy as np
from scipy.integrate import quad

# ============================================================================
# BASE CLASS
# ============================================================================

class ModeShape(ABC):
    """
    Abstract base class for mode shapes.
    
    Defines the interface for mode shape functions and related calculations.
    """

    @abstractmethod
    def evaluate(self, z: Union[float, np.ndarray], h: float,
                 mode_number: int = 1) -> Union[float, np.ndarray]:
        """
        Evaluate mode shape at height z.
        
        Parameters:
        -----------
        z : float or np.ndarray
            Height(s) above ground [m]
        h : float
            Total structure height [m]
        mode_number : int
            Mode number (1-indexed)
            
        Returns:
        --------
        float or np.ndarray
            Mode shape value at height z (normalized)
        """
        pass

    @abstractmethod
    def get_number_of_modes(self) -> int:
        """
        Get number of modes defined.

        Returns:
        --------
        int
            Number of modes (1 for single mode)
        """
        pass

    def compute_modal_integral(self, h: float, mode_number: int = 1) -> float:
        """
        Compute integral of squared mode shape.

        Parameters:
        -----------
        h : float
            Structure height [m]
        mode_number : int
            Mode number (1-indexed)
            
        Returns:
        --------
        float
            Integral value
        """
        # Default: numerical integration
        def integrand(z):
            phi = self.evaluate(z, h, mode_number)
            return phi * phi
        
        result, _ = quad(integrand, 0, h)
        return result
    
    def compute_k_xi(self, h: float, mode_number: int = 1) -> float:
        """
        Compute K_xi parameter for vortex-resonance model.

        K_xi = ∫₀ʰ Φₙ(z) dz / (4π ∫₀ʰ (Φₙ(z))² dz)

        This parameter appears in the vortex-resonance amplitude equation.
        
        Parameters:
        -----------
        h : float
            Structure height [m]
        mode_number : int
            Mode number (1-indexed)
            
        Returns:
        --------
        float
            K_xi parameter
        """
        def integrand_phi(z):
            return self.evaluate(z, h, mode_number)
        
        numerator, _ = quad(integrand_phi, 0, h)
        denominator = 4 * np.pi * self.compute_modal_integral(h, mode_number)
        return numerator / denominator
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
    
# ============================================================================
# CONCRETE IMPLEMENTATIONS
# ============================================================================

class FundamentalModeShape(ModeShape):
    """
    Fundamental mode shape: φ(z) = (z/h)^n
    
    Default for cantilever structures: n=2 (parabolic)
    """
    def __init__(self, exponent: float = 2.0):
        """
        Initialize fundamental mode shape.
        
        Parameters:
        -----------
        exponent : float
            Mode shape exponent (default: 2.0 for cantilever)
            - n=1: Linear
            - n=2: Parabolic (typical for cantilever)
            - n=3: Cubic
        """
        if exponent <= 0:
            raise ValueError("Mode shape exponent must be positive")
        
        self.exponent = exponent

    def evaluate(self, z: Union[float, np.ndarray], h: float,
                 mode_number: int = 1) -> Union[float, np.ndarray]:
            """Evaluate mode shape."""
            if mode_number != 1:
                raise ValueError("FundamentalModeShape only supports mode_number=1")
            
            z_array = np.asarray(z)
            return (z_array / h)**self.exponent
    
    def get_number_of_modes(self) -> int:
        return 1
    
    def compute_modal_integral(self, h: float, mode_number: int = 1) -> float:
        """Analytical solution: ∫[0,h] (z/h)^(2n) dz = h/(2n+1)"""
        if mode_number != 1:
            raise ValueError("FundamentalModeShape only supports mode_number=1")
        
        return h / (2 * self.exponent + 1)
    
    def __repr__(self) -> str:
        return f"FundamentalModeShape(exponent={self.exponent:.1f})"
    
class MultiModeModeShape(ModeShape):
    """
    Multiple mode shapes for multi-modal analysis.
    
    Stores a list of mode shape functions.
    """
    
    def __init__(self, mode_functions: List[callable], 
                 integral_functions: Optional[List[callable]] = None):
        """
        Initialize multi-mode shape.
        
        Parameters:
        -----------
        mode_functions : list of callable
            List of mode shape functions with signature: φ(z, h) -> value
        integral_functions : list of callable, optional
            List of integral functions for each mode: I(h) -> integral value
        """
        if not mode_functions:
            raise ValueError("At least one mode function required")
        
        if integral_functions is not None and len(integral_functions) != len(mode_functions):
            raise ValueError("Number of integral functions must match mode functions")
        
        self.mode_functions = mode_functions
        self.integral_functions = integral_functions
        self.n_modes = len(mode_functions)
    
    def evaluate(self, z: Union[float, np.ndarray], h: float, 
                mode_number: int = 1) -> Union[float, np.ndarray]:
        """Evaluate specified mode shape."""
        if mode_number < 1 or mode_number > self.n_modes:
            raise ValueError(f"mode_number must be in [1, {self.n_modes}]")
        
        return self.mode_functions[mode_number - 1](z, h)
    
    def get_number_of_modes(self) -> int:
        return self.n_modes
    
    def compute_modal_integral(self, h: float, mode_number: int = 1) -> float:
        """Compute modal integral for specified mode."""
        if mode_number < 1 or mode_number > self.n_modes:
            raise ValueError(f"mode_number must be in [1, {self.n_modes}]")
        
        # Use analytical if available
        if self.integral_functions is not None:
            return self.integral_functions[mode_number - 1](h)
        else:
            return super().compute_modal_integral(h, mode_number)
    
    def __repr__(self) -> str:
        return f"MultiModeModeShape(n_modes={self.n_modes})"
    
# ============================================================================
# COMMON MODE SHAPE DEFINITIONS
# ============================================================================

class CantileverBeamModes(MultiModeModeShape):
    """
    Analytical mode shapes for cantilever beam.
    
    Uses exact eigenfunctions of cantilever beam vibration.
    """
    
    # Modal constants for cantilever beam (first 5 modes)
    BETA_N = [1.875, 4.694, 7.855, 10.996, 14.137]
    
    def __init__(self, n_modes: int = 1):
        """
        Initialize cantilever beam mode shapes.
        
        Parameters:
        -----------
        n_modes : int
            Number of modes (1-5 supported)
        """
        if n_modes < 1 or n_modes > 5:
            raise ValueError("n_modes must be in [1, 5]")
        
        # Create mode functions
        mode_funcs = []
        for i in range(n_modes):
            beta = self.BETA_N[i]
            
            def mode_func(z, h, beta_val=beta):
                xi = z / h
                return (np.cosh(beta_val * xi) - np.cos(beta_val * xi) - 
                       ((np.cosh(beta_val) + np.cos(beta_val)) / 
                        (np.sinh(beta_val) + np.sin(beta_val))) * 
                       (np.sinh(beta_val * xi) - np.sin(beta_val * xi)))
            
            mode_funcs.append(mode_func)
        
        # Initialize parent (no analytical integrals provided - use numerical)
        super().__init__(mode_functions=mode_funcs, integral_functions=None)
        self.n_modes_requested = n_modes
    
    def __repr__(self) -> str:
        return f"CantileverBeamModes(n_modes={self.n_modes_requested})"


# ============================================================================
# FACTORY FUNCTIONS (for convenience)
# ============================================================================

def create_fundamental_mode(exponent: float = 2.0) -> FundamentalModeShape:
    """Create fundamental mode shape with given exponent."""
    return FundamentalModeShape(exponent)

def create_cantilever_modes(n_modes: int = 1) -> CantileverBeamModes:
    """Create analytical cantilever beam mode shapes."""
    return CantileverBeamModes(n_modes)

# ============================================================================
# EXAMPLE USAGE & TESTING
# ============================================================================

if __name__ == "__main__":
    print("Mode Shape Classes - Demo")
    print("=" * 60)
    
    # Example 1: Fundamental mode (default parabolic)
    print("\n1. Fundamental Mode Shape (parabolic):")
    mode1 = FundamentalModeShape(exponent=2.0)
    print(f"   {mode1}")
    h = 60.0
    heights = np.array([0, 15, 30, 45, 60])
    values = mode1.evaluate(heights, h)
    for z, phi in zip(heights, values):
        print(f"   φ({z:2.0f}m) = {phi:.4f}")
    integral = mode1.compute_modal_integral(h)
    print(f"   Modal integral: ∫φ² dz = {integral:.4f} m")
    print(f"   Analytical: h/(2n+1) = {h / (2*2+1):.4f} m")
    
    # Example 2: Fundamental mode (linear)
    print("\n2. Fundamental Mode Shape (linear):")
    mode2 = FundamentalModeShape(exponent=1.0)
    print(f"   {mode2}")
    values = mode2.evaluate(heights, h)
    for z, phi in zip(heights, values):
        print(f"   φ({z:2.0f}m) = {phi:.4f}")
    integral = mode2.compute_modal_integral(h)
    print(f"   Modal integral: ∫φ² dz = {integral:.4f} m")