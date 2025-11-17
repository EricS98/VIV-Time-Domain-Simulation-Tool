# config/aerodynamic_parameters/__init__.py
"""
Aerodynamic Parameters Configuration Package
=============================================

Provides modular configuration for aerodynamic parameters:
- RMS lift coefficient σ_CL(Re)
- Aerodynamic damping parameter Ka_max(Re)
- Turbulence reduction methods Ka_eff = f(Ka_max, Iv, V/Vcr)

Separation of concerns allows flexible mixing of different models.
"""

# Lift coefficient
from .lift_coefficient import (
    LiftCoefficient,
    CICINDLiftCoefficient,
    create_cicind_lift_coefficient,
)

# Damping parameter
from .damping_parameter import (
    DampingParameter,
    CICINDDampingParameter,
    EurocodeDampingParameter,
    create_cicind_damping_parameter,
    create_eurocode_damping_parameter,
)

# Turbulence reduction
from .aerodynamic_damping_reduction import (
    AeroDampingModification,
    AeroDampingSimplified,
    AeroDampingFullCICIND,
    NoAeroDampingModification,
    create_simplified_modification,
    create_full_modification,
    create_no_modification
)

# Amplitude dependent damping formulation
from .amplitude_dependent_damping import (
    AmplitudeDependentDamping,
    VickeryBasuDamping,
    LupiDamping,
    EurocodeDamping
)

__all__ = [
    # Lift coefficient
    'LiftCoefficient',
    'CICINDLiftCoefficient',
    'EurocodeLiftCoefficient',
    'ConstantLiftCoefficient',
    'create_cicind_lift_coefficient',
    'create_eurocode_lift_coefficient',
    'create_constant_lift_coefficient',
    
    # Damping parameter
    'DampingParameter',
    'CICINDDampingParameter',
    'EurocodeDampingParameter',
    'ConstantDampingParameter',
    'create_cicind_damping_parameter',
    'create_eurocode_damping_parameter',
    'create_constant_damping_parameter',
    
    # Turbulence reduction
    'AeroDampingModification',
    'AeroDampingSimplified',
    'AeroDampingFullCICIND',
    'NoAeroDampingModification',
    'create_simplified_modification',
    'create_full_modification',
    'create_no_modification',

    # Amplitude dependent aerodynamic damping
    'AmplitudeDependentDamping',
    'VickeryBasuDamping',
    'LupiDamping',
    'EurocodeDamping',
]