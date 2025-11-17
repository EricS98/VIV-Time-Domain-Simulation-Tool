# config/analysis_config.py
"""
Master VIV Analysis Configuration
==================================

Combines all configuration aspects into a single, cohesive configuration class.
This is the main interface users interact with to define their analysis.
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

# Import all configuration base classes
from .wind_profile import (
    WindProfile, 
    ConstantWindProfile, 
    TerrainBasedWindProfile,
    create_constant_profile,
    create_terrain_profile
)
from .geometry import (
    CrossSection,
    ConstantDiameter,
    LinearTaper,
    create_constant_diameter,
    create_linear_taper
)
from .mode_shape import (
    ModeShape,
    FundamentalModeShape,
    create_fundamental_mode
)
from .reference_diameter import (
    ReferenceDiameterConfig,
    ReferenceDiameterMethod,
    create_top_third_config
)
from .coherence_function import (
    CoherenceFunction,
    ConstantCorrelationLength,
    VickeryClarkCoherence
)
from .aerodynamic_parameters import (
    DampingParameter, CICINDDampingParameter,
    LiftCoefficient, CICINDLiftCoefficient,
    AeroDampingModification, AeroDampingSimplified, 
    AeroDampingFullCICIND, NoAeroDampingModification,
    AmplitudeDependentDamping, VickeryBasuDamping,
    LupiDamping, EurocodeDamping
)

from .time_domain import (
    NewmarkMethod, ConstantAcceleration, LinearAcceleration, TimeDomainConfig
)

# RESPONSE TYPE ENUM
class ResponseType(Enum):
    """Type of response calculation."""
    FREQUENCY_DOMAIN = "frequency_domain"
    TIME_DOMAIN = "time_domain"
    BOTH = "both"

@dataclass
class VIVAnalysisConfig:
    """
    Master configuration for VIV analysis.
    
    Combines all configuration aspects into a single, flexible setup.
    Users create instances of this class to define their analysis.
    """
    
    # Structure name (for reference)
    structure_name: str = "Structure"

    # CORE CONFIGURATION (required)
    St: float = 0.2     # Default Strouhal number
    
    wind_profile: WindProfile = field(
        default_factory=lambda: ConstantWindProfile(u_ref=10.0)
    )
    
    cross_section: CrossSection = field(
        default_factory=ConstantDiameter
    )

    d_ref_config: ReferenceDiameterConfig = field(
        default_factory=create_top_third_config
    )
    
    mode_shape: ModeShape = field(
        default_factory=lambda: FundamentalModeShape(exponent=2.0)
    )
    
    damping_parameter: DampingParameter = field(
        default_factory=CICINDDampingParameter
    )
    
    lift_coefficient: LiftCoefficient = field(
        default_factory=CICINDLiftCoefficient
    )
    
    damping_modification: AeroDampingModification = field(
        default_factory=AeroDampingSimplified
    )

    coherence_function: CoherenceFunction = field(
        default_factory=lambda: ConstantCorrelationLength(lambda_factor=1.0)
    )

    damping_formulation: AmplitudeDependentDamping = field(
        default_factory=VickeryBasuDamping
    )
    
    # ANALYSIS PARAMETERS        
    response_type: ResponseType = ResponseType.FREQUENCY_DOMAIN
    
    time_domain_config: TimeDomainConfig = field(default_factory=TimeDomainConfig)

    # WILLECKE-PEIL EXTENSION PARAMETERS (optional)
    use_willecke_peil: bool = False
    rho_air: Optional[float] = None
    nu_air: Optional[float] = None
    manual_kw: Optional[float] = None
    
    # METHODS    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Create time-domain config if needed and not provided
        if self.response_type in [ResponseType.TIME_DOMAIN, ResponseType.BOTH]:
            if self.time_domain_config is None:
                self.time_domain_config = TimeDomainConfig()
    
    def requires_height_integration(self) -> bool:
        """
        Check if configuration requires numerical height integration.
        
        Returns True if:
        - Wind profile is height-varying, OR
        - Cross-section is height-varying, OR
        - Damping model requires height integration (Lupi)
        
        Returns:
        --------
        bool
            True if height integration needed
        """
        return (not self.wind_profile.is_constant() or 
                not self.cross_section.is_constant())
    
    def get_analysis_description(self) -> str:
        """
        Get human-readable description of the analysis configuration.
        
        Returns:
        --------
        str
            Multi-line description of the configuration
        """
        lines = [
            f"VIV Analysis Configuration: {self.structure_name}",
            "=" * 70,
            "",
            "Configuration:",
            f"  Wind Profile:       {self.wind_profile}",
            f"  Cross-Section:      {self.cross_section}",
            f"  Mode Shape:         {self.mode_shape}",
            f"  Damping Model:      {self.damping_formulation.get_name()}",
            f"  Damping Parameter:  {self.damping_parameter}",
            f"  Lift Coefficient:   {self.lift_coefficient}",
            f"  Turb. Reduction:    {self.damping_modification}",

            "",
            "Analysis Parameters:",
            f"  Response Type:      {self.response_type.value}",
        ]
        
        if self.time_domain_config is not None:
            lines.extend([
                f"  Time Domain:        {self.time_domain_config}",
            ])
        
        lines.extend([
            "",
            "Computational Approach:",
            f"  Height Integration: {'Required' if self.requires_height_integration() else 'Not required (analytical)'}"
        ])
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        """Compact representation."""
        return (f"VIVAnalysisConfig(name='{self.structure_name}', "
                f"Iv={self.turbulence_intensity:.3f}, "
                f"response={self.response_type.value})")