# config/__init__.py
"""
VIV Analysis Configuration Package
===================================

Flexible, extensible configuration system for VIV analysis.

This package provides:
- Wind profile configuration (constant, power law, logarithmic, terrain-based)
- Cross-section geometry (constant diameter, linear taper, custom)
- Mode shape configuration (fundamental, custom, cantilever beam modes)
- Damping model selection (Vickery & Basu, Lupi)
- Aerodynamic curve selection (CICIND, Eurocode, custom)
- Master configuration class combining all aspects

Advanced Usage:
--------------
    from config import (
        VIVAnalysisConfig,
        PowerLawWindProfile,
        LinearTaper,
        FundamentalModeShape,
        VickeryBasuDamping,
        CICINDCurves,
        ResponseType
    )
    
    config = VIVAnalysisConfig(
        structure_name="My_Chimney",
        wind_profile=ConstantWindProfile(u_ref=12.0),
        cross_section=ConstantDiameter(),
        mode_shape=FundamentalModeShape(exponent=2.0),
        damping_model=VickeryBasuDamping(),
        damping_parameter=CICINDDampingParameter(),
        lift_coefficient=CICINDLiftCoefficient(),
        damping_modification=SimplifiedTurbulenceReduction(),
        turbulence_intensity=0.08,
        response_type=ResponseType.FREQUENCY_DOMAIN
    )
"""

# Master configuration
from .analysis_config import (
    VIVAnalysisConfig,
    ResponseType,
    TimeDomainConfig,
)

# Wind profiles
from .wind_profile import (
    WindProfile,
    ConstantWindProfile,
    PowerLawWindProfile,
    TerrainBasedWindProfile,
    LogarithmicWindProfile,
    create_constant_profile,
    create_power_law_profile,
    create_terrain_profile,
    create_logarithmic_profile
)

# Geometry/Cross-section
from .geometry import (
    CrossSection,
    ConstantDiameter,
    LinearTaper,
    BaseTaper,
    create_constant_diameter,
    create_linear_taper,
    create_base_taper
)

# Reference diameter
from .reference_diameter import (
    ReferenceDiameterConfig,
    ReferenceDiameterMethod,
    ReferenceDiameterCalculator,
    create_effective_diameter_config,
    create_top_diameter_config,
    create_top_third_config
)

# Mode shapes
from .mode_shape import (
    ModeShape,
    FundamentalModeShape,
    MultiModeModeShape,
    CantileverBeamModes,
    create_fundamental_mode,
    create_cantilever_modes
)

# Damping models
from .damping_model import (
    DampingModel,
    VickeryBasuDamping,
    LupiDamping,
    CustomDamping,
    create_vickery_basu,
    create_lupi
)

# Aerodynamic parameters
from .aerodynamic_parameters import (
    DampingParameter,
    CICINDDampingParameter,
    EurocodeDampingParameter,
    LiftCoefficient,
    CICINDLiftCoefficient,
    AeroDampingModification,
    AeroDampingSimplified,
    AeroDampingFullCICIND,
    NoAeroDampingModification,
    AmplitudeDependentDamping,
    VickeryBasuDamping,
    LupiDamping,
    EurocodeDamping
)

# Coherence functions
from .coherence_function import (
    CoherenceFunction,
    ConstantCorrelationLength,
    VickeryClarkCoherence,
    create_constant_correlation,
    create_vickery_clark
)

# Time-domain
from .time_domain import (
    TimeDomainConfig,
    NewmarkMethod,
    ConstantAcceleration,
    LinearAcceleration
)

__all__ = [
    # Master configuration
    'VIVAnalysisConfig',
    'ResponseType',
    
    # Wind profiles
    'WindProfile',
    'ConstantWindProfile',
    'PowerLawWindProfile',
    'TerrainBasedWindProfile',
    'LogarithmicWindProfile',
    'create_constant_profile',
    'create_power_law_profile',
    'create_terrain_profile',
    'create_logarithmic_profile',
    
    # Geometry
    'CrossSection',
    'ConstantDiameter',
    'LinearTaper',
    'BaseTaper',
    'create_constant_diameter',
    'create_linear_taper',
    'create_base_taper',

    # Reference diameter
    'ReferenceDiameterConfig',
    'ReferenceDiameterMethod',
    'ReferenceDiameterCalculator',
    
    # Mode shapes
    'ModeShape',
    'FundamentalModeShape',
    'CustomModeShape',
    'MultiModeModeShape',
    'CantileverBeamModes',
    'create_fundamental_mode',
    'create_custom_mode',
    'create_cantilever_modes',
    
    # Damping models
    'DampingModel',
    'VickeryBasuDamping',
    'LupiDamping',
    'CustomDamping',
    'create_vickery_basu',
    'create_lupi',
    
    # Aerodynamic parameters
    'DampingParameter',
    'CICINDDampingParameter',
    'EurocodeDampingParameter',
    'LiftCoefficient',
    'CICINDLiftCoefficient',
    'AeroDampingModification',
    'AeroDampingSimplified',
    'AeroDampingFullCICIND',
    'NoAeroDampingModification',
    'AmplitudeDependentDamping',
    'VickeryBasuDamping',
    'LupiDamping',
    'EurocodeDamping',

    # Coherence function
    'CoherenceFunction',
    'ConstantCorrelationLength',
    'VickeryClarkCoherence',
    'create_constant_correlation',
    'create_vickery_clark',

    # Time-domain
    'TimeDomainConfig',
    'NewmarkMethod',
    'ConstantAcceleration',
    'LinearAcceleration',
]

__version__ = '1.0.0'