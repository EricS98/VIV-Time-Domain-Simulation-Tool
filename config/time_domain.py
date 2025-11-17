# config/time_domain.py
"""
Time-Domain Configuration
"""

import numpy as np
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Optional, Tuple

class NewmarkMethod(ABC):
    """Abstract base for Newmark methods."""

    @abstractmethod
    def get_parameters(self) -> tuple[float, float]:
        """Get Newmark parameters beta and gamma."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get descriptive name of method"""
        pass

    @abstractmethod
    def is_unconditionally_stable(self) -> bool:
        """Return True if unconditionally stable."""
        pass

    @abstractmethod
    def get_stability_limit(self) -> Optional[float]:
        """Get the maximum stable dt/T_n ratio."""
        pass

@dataclass
class ConstantAcceleration(NewmarkMethod):
    def get_parameters(self) -> Tuple[float, float]:
        # gamma=0.5, beta=0.25
        return (0.5, 0.25)

    def get_name(self) -> str:
        gamma, beta = self.get_parameters()
        return f"Average Acceleration Method (γ = {gamma:.2f}, β = {beta:.2f})"

    def is_unconditionally_stable(self) -> bool:
        return True
    
    def get_stability_limit(self) -> Optional[float]:
        return None
    
@dataclass
class LinearAcceleration(NewmarkMethod):
    def get_parameters(self) -> Tuple[float, float]:
        # gamma=0.5, beta=1/6
        return (0.5, 1.0/6.0)
    
    def get_name(self) -> str:
        gamma, beta = self.get_parameters()
        return f"Linear Acceleration Method (γ = {gamma:.2f}, β = {beta:.2f})"
    
    def is_unconditionally_stable(self) -> bool:
        return False
    
    def get_stability_limit(self) -> Optional[float]:
        # Stability limit: dt/T_n <= 1/(pi*sqrt(2)) * 1/sqrt(gamma - 2*beta)
        # For gamma=0.5, beta=1/6: gamma - 2*beta = 1/6. Limit is sqrt(3)/pi ≈ 0.5513
        stability_limit = np.sqrt(3.0) / np.pi
        return stability_limit

@dataclass
class RungeKuttaMethod:
    """Runge-Kutta time integration method."""

    def get_name(self) -> str:
        return "Runge-Kutta (RK45)"
    
    def is_unconditionally_stable(self) -> bool:
        # RK45 is an explicit method with adaptive stepping
        return False
    
    def get_stability_limit(self) -> Optional[float]:
        return 2.8

@dataclass
class TimeDomainConfig:
    """Configuration for time-domain analysis."""
    oversampling_factor: float = 2.0    # Nyquist margin above physical band
    nsigma_coverage: int = 4            # Gaussian coverage for force spectrum
    rms_window_cycles: float = 2.0      # Window for running RMS calculation
    duration: float = 600.0             # Default simulation duration [s]
    dt: float = 0.05                    # Default time step [s]
    min_natural_periods: int = 1000     # Minimum periods to simulate
    n_realizations: int = 1             # Default number of realizations
    integration_method: object = field(default_factory=ConstantAcceleration)
    random_seed: Optional[int] = None  # Seed for random number generation
    max_trim_fraction: float = 0.5      # Never discard more than half of the simulation

    def get_newmark_parameters(self) -> tuple[float, float]:
        """Returns (gamma, beta) according to selected variant."""
        if isinstance(self.integration_method, NewmarkMethod):
            return self.integration_method.get_parameters()
        
        raise TypeError(
            f"Cannot get Newmark parameters for method: {self.get_method_name()}. "
            "Integration method is not a Newmark variant. Check solver logic."
        )
        
    def get_method_name(self) -> str:
        """Get integration method name."""
        if hasattr(self.integration_method, 'get_name'):
            return self.integration_method.get_name()
        return "Unknown Method"
