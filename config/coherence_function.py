# config/coherence_function.py
"""
Coherence Function Configuration
================================

Defines spatial coherence/correlation functions for lift forces.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Union

class CoherenceFunction(ABC):
    """Abstract base for coeherence/correlation functions."""
    
    @abstractmethod
    def get_coherence(self, z1: float, z2: float, f: float, d1: float, d2: float) -> float:
        """
        Calculate coherence between heights z1 and z2.
        
        Parameters:
        -----------
        z1, z2 : float
            Heights [m]
        f : float
            Frequency [Hz]
        d1, d2 : float
            Diameters at z1 and z2 [m]
            
        Returns:
        --------
        float
            Coherence value [0, 1]
        """
        pass

    @abstractmethod
    def is_constant_correlation_length(self) -> bool:
        """
        Return true if this can be simplified to single integral.
        
        Returns:
        --------
        bool
            True for constant correlation length models
        """
        pass

    @abstractmethod
    def is_frequency_independent(self) -> bool:
        """
        Return true if coherence function does not depend on frequency.
        
        This allows optimization: when True, the coherence matrix can be
        calculated once and reused for all frequencies.
        
        Returns:
        --------
        bool
            True if coherence is frequency-independent
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Get descriptive name of coherence function.

        Returns:
        --------
        str
            Name of the coherence function
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

class ConstantCorrelationLength(CoherenceFunction):
    """Simplified model: λ = constant × D"""
    def __init__(self, lambda_factor: float = 1.0):
        """
        Initialize constant correlation length.
        
        Parameters:
        -----------
        lambda_factor : float
            Correlation length factor (default: 1.0)
            Correlation length = lambda_factor × diameter
        """
        if lambda_factor <= 0:
            raise ValueError("lambda_factor must be positive")
        self.lambda_factor = lambda_factor

    def get_coherence(self, z1: float, z2: float, f: float, d1: float, d2: float) -> float:
        """
        Not used for constant correlation length model.
        """
        raise NotImplementedError(
            "Constant correlation length uses simplified single integral."
        )

    def is_constant_correlation_length(self) -> bool:
        return True
    
    def is_frequency_independent(self) -> bool:
        return True 
    
    def get_name(self) -> str:
        return f"Constant correlation length (λ = {self.lambda_factor:.1f})"
    
    def __repr__(self) -> str:
        return f"ConstantCorrelationLength(lambda_factor={self.lambda_factor})"
    
class VickeryClarkCoherence(CoherenceFunction):
    """Vickery & Clark (1972) correlation function."""
    def __init__(self):
        """Initialize Vickery-Clark correlation function."""
        pass

    def get_coherence(self, z1: float, z2: float, f: float, d1: float, d2: float) -> float:
        """
        Calculate Vickery-Clark coherence.
        
        Parameters:
        -----------
        z1, z2 : float
            Heights [m]
        f : float
            Frequency [Hz] (not used in this model)
        d1, d2 : float
            Diameters at z1 and z2 [m]
            
        Returns:
        --------
        float
            Coherence value
        """
        # Avoid division by zero
        if d1 + d2 == 0:
            return 1.0 if z1 == z2 else 0.0
        
        # Normalized separation distance
        r = 2 * abs(z1 - z2) / (d1 + d2)

        return np.cos(2*r/3) * np.exp(-(r/3)**2)
    
    def is_constant_correlation_length(self) -> bool:
        return False
    
    def is_frequency_independent(self) -> bool:
        return True  # Vickery-Clark does not depend on frequency
    
    def get_name(self) -> str:
        return "Vickery-Clark (1972)"
    
    def __repr__(self) -> str:
        return "VickeryClarkCoherence()"
    
# Factory functions
def create_constant_correlation(lambda_factor: float = 1.0) -> ConstantCorrelationLength:
    """Create constant correlation length model."""
    return ConstantCorrelationLength(lambda_factor=lambda_factor)


def create_vickery_clark() -> VickeryClarkCoherence:
    """Create Vickery-Clark coherence function."""
    return VickeryClarkCoherence()