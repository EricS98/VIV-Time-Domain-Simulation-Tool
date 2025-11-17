# common/analysis_config_helpers.py
"""
Analysis Configuration Helper Functions
========================================

Helper functions for building VIVAnalysisConfig interactively or from arguments.
Keeps the main script files clean and organized.
"""

from typing import Optional
from pathlib import Path
from config import (
    VIVAnalysisConfig,
    ConstantWindProfile, PowerLawWindProfile, TerrainBasedWindProfile,
    FundamentalModeShape,
    VickeryBasuDamping, LupiDamping, EurocodeDamping,
    CICINDDampingParameter, CICINDLiftCoefficient,
    AeroDampingSimplified, AeroDampingFullCICIND, NoAeroDampingModification,
    ConstantCorrelationLength, VickeryClarkCoherence
)
from config.geometry import (
    ConstantDiameter,
    LinearTaper,
    BaseTaper,
    TopTaper
)
from config.time_domain import (
    TimeDomainConfig,
    ConstantAcceleration,
    LinearAcceleration,
    RungeKuttaMethod,
    NewmarkMethod
)
from config.reference_diameter import (
    ReferenceDiameterConfig,
    ReferenceDiameterMethod,
    ReferenceDiameterCalculator,
    create_top_diameter_config,
    create_top_third_config,
    create_effective_diameter_config,
)
from common.structures import StructureProperties
from calculators.structure_properties_calculator import (
    StructurePropertiesCalculator,
    print_structure_summary
)


class ConfigBuilder:
    """Helper class for building VIVAnalysisConfig objects."""
    
    def __init__(self, interactive_helper=None):
        """
        Initialize config builder.
        
        Parameters:
        -----------
        interactive_helper : InteractiveInputHelpers, optional
            Helper object with methods like _get_float_input, _yesno, etc.
        """
        self.helper = interactive_helper
        self.structure_calc = StructurePropertiesCalculator()
    
    def build_interactive(self, structure: StructureProperties,
                          is_template_mode: bool = False) -> Optional[VIVAnalysisConfig]:
        """
        Build configuration interactively.
        
        Parameters:
        -----------
        structure : StructureProperties
            The structure being analyzed
        is_template_mode : bool
            If True, skips unnecessary prints/prompts for batch analysis.
            
        Returns:
        --------
        VIVAnalysisConfig or None
            Configuration object or None if failed
        """
        if self.helper is None:
            raise ValueError("Interactive helper required for interactive mode")
        
        try:
            if not is_template_mode:
                print("\n" + "=" * 70)
                print("ANALYSIS CONFIGURATION")
                print("=" * 70)

            # Step 1: Cross-section
            print("\n" + "—" * 70)
            print("STEP 1: CROSS-SECTION DEFINITION")
            print("—" * 70)
            cross_section = self._create_section_from_structure(structure)
            print(f"✅ Cross-Section: {cross_section}")

            # Step 2: Mode shape
            print("\n" + "—" * 70)
            print("STEP 2: MODE SHAPE")
            print("—" * 70)
            mode_exponent = self.helper._get_float_input(
                "Mode shape exponent a (z/h)**a (default 2.0 for cantilever): ",
                default=2.0, min_val=1.0, max_val=3.0
            )
            mode_shape = FundamentalModeShape(exponent=mode_exponent)
            mode_shape_integral = mode_shape.compute_modal_integral(
                structure.height, mode_number=1
            )

            # Step 3: Reference diameter
            print("\n" + "—" * 70)
            print("STEP 3: REFERENCE DIAMETER")
            print("—" * 70)
            d_ref_config = self._get_reference_diameter(cross_section)

            d_ref = ReferenceDiameterCalculator.calculate(
                config=d_ref_config,
                cross_section=cross_section,
                height=structure.height,
                d_nominal=structure.diameter,
                mode_shape=mode_shape,
                mode_number=1
            )
            # Print the calculated diameter for the user
            print(f"\n✅ Calculated Reference Diameter (d_ref): {d_ref:.3f} m (Method: {d_ref_config.method.value})")
            
            # Step 4: Strouhal number
            print("\n" + "—" * 70)
            print("STEP 4: STROUHAL NUMBER")
            print("—" * 70)
            St = self.helper._get_float_input(
                "Strouhal number (default: 0.2)",
                default=0.2, min_val=0.1, max_val=0.5
            )

            # Compute all properties and print summary
            if not is_template_mode:
                print("\n" + "=" * 70)
                computed_properties = self.structure_calc.compute_all_properties(
                    structure=structure,
                    d_ref=d_ref,
                    St=St,
                    m_eq=structure.m_eq,
                    mode_shape_integral=mode_shape_integral
                )
                print_structure_summary(structure, computed_properties)
                print("=" * 70)

                # Confirm before proceeding
                if not self.helper._yesno("\nProceed with these parameters?", default=True):
                    print("Configuration cancelled by user")
                    return None
            
            # Step 5: Wind Profile
            print("\n" + "—" * 70)
            if is_template_mode:
                print("STEP 5: WIND PROFILE (Selecting profile type only)")
            else:
                print("STEP 5: WIND PROFILE")
            print("—" * 70)
            self.helper.structure = structure
            wind_profile = self._get_wind_profile_interactive(St, d_ref=d_ref, is_template_mode=is_template_mode)
            
            # Step 6: Coherence function
            print("\n" + "—" * 70)
            print("STEP 6: COHERENCE FUNCTION")
            print("—" * 70)
            coherence = self._get_coherence_function_interactive()
            
            # Step 7: Damping formulation (amplitude-dependent damping)
            print("\n" + "—" * 70)
            print("STEP 7: DAMPING FORMULATION")
            print("—" * 70)
            damping_formulation = self._get_damping_formulation_interactive()

            # Step 8: Aerodynamic damping modification
            print("\n" + "—" * 70)
            print("STEP 8: AERODYNAMIC DAMPING MODIFICATION")
            print("—" * 70)
            damping_modification = self._get_damping_modification_interactive()

            # Fixed parameters (can be extended if needed)
            damping_parameter = CICINDDampingParameter()
            lift_coefficient = CICINDLiftCoefficient()
            
            # Build config
            config = VIVAnalysisConfig(
                structure_name=structure.name,
                St=St,
                wind_profile=wind_profile,
                cross_section=cross_section,
                mode_shape=mode_shape,
                damping_formulation=damping_formulation,
                damping_parameter=damping_parameter,
                lift_coefficient=lift_coefficient,
                damping_modification=damping_modification,
                coherence_function=coherence,
                d_ref_config=d_ref_config
            )
            
            print("\n✅ Configuration created")
            return config
            
        except Exception as e:
            print(f"\n❌ Configuration error: {e}")
            import traceback
            traceback.print_exc()
            return None
        
    def _create_section_from_structure(self, structure: StructureProperties):
        """
        Create cross-section object based on StructureProperties.
        """
        cs_type = structure.cross_section_variation.lower()
        d_top = structure.diameter
        d_base = structure.diameter_base
        h1 = structure.taper_height

        if cs_type == 'constant':
            return ConstantDiameter()
        elif cs_type == 'linear':
            return LinearTaper(d_base=d_base, d_top=d_top)
        elif cs_type == 'base_taper':
            return BaseTaper(d_base=d_base, d_top=d_top, h1=h1)
        elif cs_type == 'top_taper':
            return TopTaper(d_base=d_base, d_top=d_top, h1=h1)
        else:
            print(f"⚠️ Warning: Unknown cross-section type '{cs_type}'. Defaulting to ConstantDiameter.")
            return ConstantDiameter()
        
    def _get_wind_profile_interactive(self, St, d_ref=None, is_template_mode=False):
        """Interactively get wind profile configuration."""
        # Use d_ref for critical velocity if available, otherwise nominal
        d_ref = d_ref if d_ref is not None else self.helper.structure.diameter
        u_crit = self.helper.structure.f_n * d_ref / St

        # Skip asking for u_ref if we are in batch template mode
        if is_template_mode:
            print("ℹ️  Reference wind speed will be set to $u_{crit}$ for each structure later.")

        print("\n--- Wind Profile ---")
        print("1. Constant (uniform over height)")
        print("2. Terrain-based power law (CICIND/Eurocode)")
        print("3. Custom power law")
        
        choice = self.helper._get_user_choice("Select wind profile (1-3) (default: 1): ", [1, 2, 3], default=1)
        
        if choice == 1:
            if is_template_mode:
                u_ref = u_crit
            else:
                u_ref = self.helper._get_float_input(
                    f"Reference wind velocity [m/s] (default: {u_crit:.2f} m/s - critical wind velocity): ",
                    default=u_crit, min_val=0.1
                )   
            Iv = self.helper._get_float_input(
                "Turbulence intensity [0.0-0.3] (default: 0.0): ",
                default=0.0, min_val=0.0, max_val=0.3
            )
            return ConstantWindProfile(u_ref=u_ref, Iv=Iv)
        
        elif choice == 2:
            terrain = self.helper._get_terrain_input()
            z_ref = self.helper._get_float_input(
                f"Reference height [m] (default: {self.helper.structure.height:.1f} m - structure height): ",
                default=self.helper.structure.height, min_val=0.1
            )
            if is_template_mode:
                u_ref = u_crit
            else:
                u_ref = self.helper._get_float_input(
                    f"Wind velocity at reference height [m/s] (default: {u_crit:.2f} m/s - critical wind velocity): ",
                    default=u_crit, min_val=0.1
                )
            return TerrainBasedWindProfile(
                u_ref=u_ref,
                z_ref=z_ref,
                terrain_category=terrain
            )
        
        else:  # Custom power law
            z_ref = self.helper._get_float_input(
                f"Reference height [m] (default: {self.helper.structure.height:.1f} m - structure height): ",
                default=self.helper.structure.height, min_val=0.1
            )
            if is_template_mode:
                u_ref = u_crit
            else:
                u_ref = self.helper._get_float_input(
                    f"Wind velocity at reference height [m/s] (default: {u_crit:.2f} m/s - critical wind velocity): ",
                    default=u_crit, min_val=0.1
                )
            alpha = self.helper._get_float_input(
                "Power-law exponent alpha (default: 0.16): ",
                default=0.16, min_val=0.01, max_val=0.4
            )
            z_min = self.helper._get_float_input(
                "Min. height [m] (default: 4.0 m): ",
                default=4.0, min_val=0.01
            )
            return PowerLawWindProfile(u_ref=u_ref, z_ref=z_ref, alpha=alpha, z_min=z_min)
           
    def _get_reference_diameter(self, cross_section) -> ReferenceDiameterConfig:
        """Interactively configure reference diameter calculation method."""
        # For constant diameter, no need to select method
        if cross_section.is_constant():
            print("\n--- Reference Diameter ---")
            print("\nℹ️  Diameter is constant. Using nominal diameter as reference diameter.")
            return create_top_diameter_config() # Any method works
        
        # Only asks for method selection if diameter varies
        print("\n--- Reference Diameter ---")
        print("1. Top diameter")
        print("2. Top third average")
        print("3. Mode-shape weighted")

        while True:
            method_choice = input("Select method [1-3] (default: 3): ").strip()

            if not method_choice:   # Default
                method_choice = "3"

            if method_choice == "1":
                print("✓ Using TOP diameter method")
                return create_top_diameter_config()
            
            elif method_choice == "2":
                print("✓ Using TOP_THIRD_AVG method (CICIND recommendation)")
                return create_top_third_config(n_points=200)
            
            elif method_choice == "3":
                print("✓ Using EFFECTIVE diameter method (Livanos)")
                print("  Note: Requires mode shape to be configured")
                return create_effective_diameter_config(n_points=200)
        
            else:
                print(f"Invalid choice: {method_choice}. Please enter 1-3.")
    
    def _get_coherence_function_interactive(self):
        """Interactively get coherence function configuration."""
        print("\n--- Coherence Function ---")
        print("1. Constant correlation length")
        print("2. Vickery-Clark correlation function")
        
        choice = self.helper._get_user_choice("Select coherence (1-2) (default: 1): ", [1, 2], default=1)
        
        if choice == 1:
            lambda_factor = self.helper._get_float_input(
                "Correlation length factor (× diameter): ",
                default=1.0, min_val=0.5, max_val=3.0
            )
            return ConstantCorrelationLength(lambda_factor=lambda_factor)
        else:
            print("ℹ️  Using Vickery-Clark coherence function")
            return VickeryClarkCoherence()
        
    def _get_damping_modification_interactive(self):
        """Interactively get aerodynamic damping reduction configuration."""
        print("\n--- Aerodynamic Damping Modification ---")
        print("1. No modification (Ka_eff = Ka_max)")
        print("2. Simplified formulation (turbulence only)")
        print("3. Full (uses Ka_mean.csv curves) - considers wind velocity ratio")

        choice = self.helper._get_user_choice(
            "Select aerodynamic damping reduction method (1-3): ",
            [1, 2, 3],
            default=2
        )

        if choice == 1:
            return NoAeroDampingModification()
        elif choice == 2:
            return AeroDampingSimplified()
        else:   # choice == 3
            try:
                # Construct absolute path to Ka_mean.csv in project root
                project_root = Path(__file__).parent.parent
                ka_file = project_root / "data" / "aerodynamic_damping_modification" / "Ka_mean.csv"
                modification = AeroDampingFullCICIND(ka_data_file=str(ka_file))
                if modification.loaded:
                    return modification
                else:
                    print("⚠️  Failed to load Ka_mean.csv, falling back to simplified method")
                return AeroDampingSimplified()
            except Exception as e:
                print(f"⚠️  Error loading full aerodynamic damping modification: {e}")
                print("   Falling back to simplified method")
                return AeroDampingSimplified()
            
    def _get_damping_formulation_interactive(self):
        """Interactively get amplitude-dependent formulation."""
        print("\n--- Aerodynamic damping formulation ---")
        print("1. Vickery & Basu (1983) - Ka(σy) = Ka_r,n * (1 - (σy/(a_L*d_ref))²)")
        print("2. Lupi et al. (2018) - Ka(σy) = a * exp(-b*σy/d) / (σy/d)^c")
        print("3. Eurocode draft - Ka(σy) = Ka_r,n * (1 - (σy/(a_L*d_ref))^θ)")

        choice = self.helper._get_user_choice(
            "Select damping formulation (1-3): ",
            [1, 2, 3],
            default=1
        )

        if choice == 1:
            # Vickery & Basu formulation
            a_L = self.helper._get_float_input(
                "Limiting amplitude a_L (default: 0.4): ",
                default=0.4
            )
            return VickeryBasuDamping(a_L=a_L)
        
        elif choice == 2:
            # DMSM (Lupi) formulation
            # Default parameter sets from Lupi (2021)
            Lupi_defaults = {
                1: {"a": 0.3475, "b": 5.808, "c": 0.3582, "source": "Lupi (2018) - Original"},
                2: {"a": 0.2301, "b": 2.220, "c": 0.4416, "source": "Lupi (2021) - Proposal 1"},
                3: {"a": 0.220, "b": 2.000, "c": 0.5000, "source": "Lupi (2021) - Proposal 2"}
            }
            print("\n--- Lupi Damping Parameters ---")
            print("1. Lupi (2018) - Original: a=0.3475, b=5.808, c=0.3582")
            print("2. Lupi (2021) - Proposal 1: a=0.2301, b=2.220, c=0.4416")
            print("3. Lupi (2021) - Proposal 2: a=0.220, b=2.000, c=0.5000")
            print("4. Manual Entry (Custom)")

            param_choice = self.helper._get_user_choice(
                "Select parameter set (1-4) (default: 3): ",
                [1, 2, 3, 4],
                default=3
            )

            if param_choice in [1, 2, 3]:
                params = Lupi_defaults[param_choice]
                print(f"ℹ️  Using default parameters: a={params['a']}, b={params['b']}, c={params['c']}")
                return LupiDamping(a=params['a'], b=params['b'], c=params['c'])
            else: # Manual Entry
                print("ℹ️  Enter custom Lupi parameters:")
                a = self.helper._get_float_input("Parameter a: ", default=0.220)
                b = self.helper._get_float_input("Parameter b: ", default=2.000)
                c = self.helper._get_float_input("Parameter c: ", default=0.500)
                return LupiDamping(a=a, b=b, c=c)
            
        else:   # choice == 3
            # Eurocode formulation
            a_L = self.helper._get_float_input(
                "Limiting amplitude a_L (default: 0.4): ",
                default=0.4
            )
            theta = self.helper._get_float_input(
                "Exponent θ (default: 2.0): ",
                default=2.0
            )
            return EurocodeDamping(a_L=a_L, theta=theta)
    
    @staticmethod
    def build_from_args(args, structure: StructureProperties) -> VIVAnalysisConfig:
        """
        Build configuration from command line arguments.
        
        Parameters:
        -----------
        args : argparse.Namespace
            Parsed command line arguments
        structure : StructureProperties
            The structure being analyzed
            
        Returns:
        --------
        VIVAnalysisConfig
            Configuration object
        """
        # Calculate default wind speed (critical velocity) if not provided
        St = args.St if hasattr(args, 'St') else 0.2
        u_ref_default = structure.f_n * structure.diameter / St
    
        # Get wind speed (use provided value or default to critical velocity)
        u_ref = args.u_ref if args.u_ref is not None else u_ref_default

        # Wind profile
        if args.wind_type == 'constant':
            wind_profile = ConstantWindProfile(u_ref=args.wind_speed, Iv=args.Iv)
        elif args.wind_type == 'power-law':
            if args.alpha is None:
                raise ValueError("--alpha is required for power-law wind profile")
            wind_profile = PowerLawWindProfile(
                u_ref=args.wind_speed,
                z_ref=args.z_ref,
                alpha=args.alpha,
                z_min=0.01
            )
        elif args.wind_type == 'terrain':
            wind_profile = TerrainBasedWindProfile(
                u_ref=args.wind_speed,
                z_ref=args.z_ref,
                terrain_category=args.terrain
            )
        else:
            # Default
            wind_profile = ConstantWindProfile(u_ref=15.0, Iv=0.1)
        
        # Cross-section
        if args.taper:
            if not args.d_base or not args.d_top:
                raise ValueError("--d-base and --d-top are required when --taper is used")
            cross_section = LinearTaper(d_base=args.d_base, d_top=args.d_top)
        else:
            cross_section = ConstantDiameter()

        # Reference diameter
        if args.d_ref_method == 'top':
            d_ref_config = create_top_diameter_config()
        elif args.d_ref_method == 'top_third':
            d_ref_config = create_top_third_config(n_points=args.d_ref_n_points)
        elif args.d_ref_method == 'effective':
            d_ref_config = create_effective_diameter_config(n_points=args.d_ref_n_points)
        else:
            d_ref_config = create_top_third_config()    # Default
        
        # Mode shape
        mode_shape = FundamentalModeShape(exponent=2.0)
        
        # Damping model
        if hasattr(args, 'damping_model') and args.damping_model:
            if args.damping_model == 'Lupi':
                damping_model = LupiDamping()
            elif args.damping_model == 'Eurocode-draft':
                damping_model = EurocodeDamping()
            else:
                damping_model = VickeryBasuDamping()
        else:
            damping_model = VickeryBasuDamping()
    
        # Aerodynamic parameters
        damping_parameter = CICINDDampingParameter()
        lift_coefficient = CICINDLiftCoefficient()
              
        # Aerodynamic damping modification
        if hasattr(args, 'aerodynamic_damping_modification') and args.aerodynamic_damping_modification:
            if args.aerodynamic_damping_modification == 'full':
                damping_modification = AeroDampingFullCICIND()
            elif args.aerodynamic_damping_modification == 'none':
                damping_modification = NoAeroDampingModification()
            else:
                damping_modification = AeroDampingSimplified()
        else:
            damping_modification = AeroDampingSimplified()
        
        # Coherence function
        if hasattr(args, 'coherence') and args.coherence == 'vickery-clark':
            coherence_function = VickeryClarkCoherence()
        else:
            # Fix: lambda_factor vs correlation_length_factor
            lambda_factor = args.lambda_factor if hasattr(args, 'lambda_factor') else 1.0
            coherence_function = ConstantCorrelationLength(lambda_factor=lambda_factor)
    
        
        # Build config
        config = VIVAnalysisConfig(
            structure_name=structure.name,
            wind_profile=wind_profile,
            cross_section=cross_section,
            d_ref=d_ref_config,
            mode_shape=mode_shape,
            damping_model=damping_model,
            damping_parameter=damping_parameter,
            lift_coefficient=lift_coefficient,
            damping_modification=damping_modification,
            coherence_function=coherence_function,
            St=St
        )
        
        return config
    
    def build_from_template(self, template_config, structure):
        """
        Create a new config for a structure using template settings.

        Parameters:
        -----------
        template_config : VIVAnalysisConfig
            Template configuration with all settings
        structure : StructureProperties
            New structure to create config for
            
        Returns:
        --------
        VIVAnalysisConfig
            New configuration with template settings applied to new structure
        """
        from copy import deepcopy
        from config.reference_diameter import ReferenceDiameterMethod
        from calculators.structure_properties_calculator import StructurePropertiesCalculator

        # Create new config with same settings but different structure
        new_config = deepcopy(template_config)

        # 1. Create cross-section for this structure and store it
        new_section = self._create_section_from_structure(structure)
        new_config.cross_section = new_section
        new_config.structure_name = structure.name

        # 2. Calculate the correct reference diameter for the current structure
        d_ref = ReferenceDiameterCalculator.calculate(
            config=template_config.d_ref_config,
            cross_section=new_section,
            height=structure.height,
            d_nominal=structure.diameter,
            mode_shape=template_config.mode_shape,
            mode_number=1
        )

        # 3. Calculate the new critical velocity u_crit using the calculated d_ref
        f_n = structure.f_n
        St = template_config.St
        # Use the provided calculator to ensure correctness
        new_u_crit = StructurePropertiesCalculator().calculate_critical_velocity(
            f_n=f_n,
            d=d_ref,
            St=St
        )

        # Update wind profile
        if hasattr(new_config.wind_profile, 'u_ref'):
            new_config.wind_profile.u_ref = new_u_crit

        print(f"  → Reference Diameter $d_{{ref}}$: {d_ref:.3f} m")
        print(f"  → Critical Velocity $u_{{crit}}$ for {structure.name}: {new_u_crit:.2f} m/s")

        return new_config
    
    def build_vortex_resonance_config(self, structure: StructureProperties) -> Optional[VIVAnalysisConfig]:
        """
        Build configuration for vortex-resonance model interactively.
        """
        print("\n" + "=" * 70)
        print("VORTEX-RESONANCE MODEL CONFIGURATION")
        print("=" * 70)

        # Step 1: Cross-section
        cross_section = self._create_section_from_structure(structure)

        # Step 2: Mode shape
        print("\n" + "─" * 70)
        print("STEP 1: MODE SHAPE")
        print("─" * 70)
        mode_exponent = self.helper._get_float_input(
            "Mode shape exponent a (z/h)**a (default 2.0 for cantilever): ",
            default=2.0, min_val=1.0, max_val=3.0
        )
        mode_shape = FundamentalModeShape(exponent=mode_exponent)

        # Step 3: Reference diameter
        print("\n" + "─" * 70)
        print("STEP 2: REFERENCE DIAMETER")
        print("─" * 70)
        d_ref_config = self._get_reference_diameter(cross_section)
        d_ref = ReferenceDiameterCalculator.calculate(
            config=d_ref_config,
            cross_section=cross_section,
            height=structure.height,
            d_nominal=structure.diameter,
            mode_shape=mode_shape,
            mode_number=1
        )
        print(f"\n✅ Reference Diameter: {d_ref:.3f} m")

        # Step 4: Strouhal number
        print("\n" + "─" * 70)
        print("STEP 3: STROUHAL NUMBER")
        print("─" * 70)
        St = self.helper._get_float_input(
            "Strouhal number (default: 0.2)",
            default=0.2, min_val=0.1, max_val=0.5
        )

        # Step 5: Wind profile
        print("\n" + "─" * 70)
        print("STEP 4: WIND PROFILE")
        print("─" * 70)
        u_crit = structure.f_n * d_ref / St
        print(f"ℹ️  Wind speed automatically set to critical velocity: U = {u_crit:.2f} m/s")
        wind_profile = ConstantWindProfile(u_ref=u_crit, Iv=0)

        # Step 6: Willecke-Peil Extension
        print("\n" + "─" * 70)
        print("STEP 5: EXTENSION FOR STABLE ATMOSPHERIC CONDITIONS")
        print("─" * 70)
        print("This extension modifies the calculation for extreme events under stable conditions:")
        print("  • Modified L_e/d iteration (L_e/d=12 for y/d<0.05, L_e/d=30 for y/d>0.20)")
        print("  • Manual atmospheric parameters (rho, nu)")
        print("  • Manual K_w factor (e.g., 0.95 instead of 0.6 limit)")

        use_willecke = self.helper._yesno(
            "\nUse Willecke-Peil extension? (y/n): ",
            default=False
        )

        # Step 7: Manual physical constants
        rho_air = None
        nu_air = None

        if use_willecke:
            print("\n--- Manual Atmospheric Parameters ---")
            
            rho_air = self.helper._get_float_input(
                "Air density rho [kg/m³] (default: 1.35): ",
                default=1.35
            )
            nu_air = self.helper._get_float_input(
                "Kinematic viscosity nu [m²/s] (default: 1.25e-5)",
                default=1.25e-5
            )
            print(f"ℹ️  Using custom: rho={rho_air:.3f} kg/m³, nu={nu_air:.2e} m²/s")

        # Step 8: Manual K_w factor
        manual_kw = None

        if use_willecke:
            print("\n--- Manual K_w Factor ---")

            manual_kw = self.helper._get_float_input(
                "Select K_w value (suggested: 0.95): ",
                default=0.95
            )
            print(f"ℹ️  Using manual K_w = {manual_kw:.3f} (no 0.6 limit)")

        # Step 9: Build config (other parameters not needed for vortex-resonance)
        config = VIVAnalysisConfig(
            structure_name=structure.name,
            St=St,
            wind_profile=wind_profile,
            cross_section=cross_section,
            mode_shape=mode_shape,
            damping_formulation=VickeryBasuDamping(),  # Placeholder, not used
            damping_parameter=CICINDDampingParameter(),
            lift_coefficient=CICINDLiftCoefficient(),
            damping_modification=NoAeroDampingModification(),
            coherence_function=VickeryClarkCoherence(),
            d_ref_config=d_ref_config,
            use_willecke_peil=use_willecke,
            rho_air=rho_air,
            nu_air=nu_air,
            manual_kw=manual_kw
        )
    
        print("\n✅ Vortex-Resonance Configuration created")

        # Step 10: Summary
        print("\n" + "="*70)
        print("CONFIGURATION SUMMARY")
        print("="*70)
        print(f"Structure: {structure.name}")
        print(f"Strouhal number: St = {St:.3f}")
        print(f"Reference diameter method: {d_ref_config.method.value}")
        if use_willecke:
            print(f"Willecke-Peil extension: ENABLED")
            if rho_air is not None:
                print(f"  • Custom rho = {rho_air:.3f} kg/m³, nu = {nu_air:.2e} m²/s")
            if manual_kw is not None:
                print(f"  • Manual K_w = {manual_kw:.3f}")
        else:
            print(f"Willecke-Peil extension: DISABLED (standard calculation)")
        print("="*70)

        return config
    
    
def _get_integration_method_interactive(helper):
    """Interactively select the time integration method."""
    print("\n--- Time Integration Method ---")
    print("1. Newmark (Constant Acceleration) - Unconditionally Stable")
    print("2. Newmark (Linear Acceleration) - Conditionally Stable")
    print("3. Runge-Kutta (RK45) - Adaptive stepping")
    
    choice = helper._get_user_choice(
        "Select integration method (1-3) (default: 1): ",
        [1, 2, 3],
        default=1
    )
    if choice == 1:
        return ConstantAcceleration()
    elif choice == 2:
        return LinearAcceleration()
    else:
        return RungeKuttaMethod()

def get_analysis_params_interactive(helper) -> Optional[dict]:
    """
    Interactively gather analysis execution parameters.
    
    Parameters:
    -----------
    helper : InteractiveInputHelpers
        Helper object with interactive input methods
        
    Returns:
    --------
    dict or None
        Dictionary with keys: 'analysis_type', 'duration', 'dt', 'n_realizations'
    """
    print("\n" + "=" * 70)
    print("ANALYSIS PARAMETERS")
    print("=" * 70)
    
    # Analysis type
    print("\n1. Frequency-domain analysis")
    print("2. Time-domain analysis")
    print("3. Both frequency and time-domain analyses")
    
    analysis_choice = helper._get_user_choice(
        "Select analysis type (1-3) (default: 1): ",
        [1, 2, 3],
        default=1
    )
    
    analysis_type = {
        1: 'frequency_domain',  # Frequency-domain only
        2: 'time_domain',       # Time-domain only
        3: 'both'               # Both analyses
    }
    analysis_type = analysis_type[analysis_choice]

    # Initialize Newmark method to default
    newmark_method = ConstantAcceleration()
    time_domain_config = None
    
    # Time-domain parameters (if needed)
    if analysis_choice in [2, 3]:   # Time-domain or Both
        print("\n--- Time-Domain Parameters ---")

        # 1. Newmark Method Selection
        newmark_method = _get_integration_method_interactive(helper)

        # 2. Natural period for guidance
        try:
            f_n = helper.structure.f_n
            T_n = 1.0 / f_n
        except AttributeError:
            # Fallback if structure is not loaded/missing f_n
            f_n = 5.0
            T_n = 0.2
            accuracy_limit_dt = 0.02
            print("⚠️  Warning: Structure natural frequency (f_n) not available. Using T_n=0.2s fallback for constraints.")

        # Accuracy limit for time-integration methods
        accuracy_limit_dt = T_n / 20
        dt_default = 0.02
        print(f"   (Natural Period T_n = {T_n:.4f} s)")
        print(f"   (Accuracy constraint: dt <= {accuracy_limit_dt:.4f} s, for Δt/T_n=0.05)")

        # 3. Simulation duration
        duration = helper._get_float_input(
            "Simulation duration [s] (default: 600.0 s): ",
            default=600.0, min_val=60.0
        )

        # Time step
        valid_dt = False
        while not valid_dt:
            dt = helper._get_float_input(
                f"Time step [s] (default: {dt_default:.2f}): ",
                default=dt_default, min_val=0.001, max_val=1.0
            )
            current_dt_valid = True

            # A. Accuracy Constraint Check
            accuracy_passed = True
            if dt > accuracy_limit_dt:
                print(f"\n⚠️ WARNING: Accuracy constraint (Δt/T_n ≤ 0.1) exceeded!")
                print(f"   Recommended: dt <= {accuracy_limit_dt:.4f} s.")
                accuracy_passed = False

            # B. Stability Constraint Check
            stability_limit_ratio = newmark_method.get_stability_limit() if hasattr(newmark_method, "get_stability_limit") else None
            if stability_limit_ratio is not None: # Conditionally Stable
                stability_limit_dt = stability_limit_ratio * T_n
                if dt > stability_limit_dt:
                    print(f"\n❌ ERROR: Stability constraint violated for {newmark_method.get_name()}!")
                    print(f"   Required: dt <= {stability_limit_dt:.4f} s (Δt/T_n ≤ {stability_limit_ratio:.3f}).")
                    current_dt_valid = False

            if not current_dt_valid:
                print("\nOptions to fix the ERRORs (must be resolved):")
                print("   I. Reduce dt to meet the most restrictive requirement.")
                print("   II. Switch to an unconditionally stable method (Average Acceleration).")
                if helper._yesno("Adjust time step (dt)?", default=True):
                    continue
                else:
                    print("   Proceeding with current settings despite errors (may cause analysis failure)...")
                    valid_dt = True
            else:
                if accuracy_passed and stability_limit_ratio is None:
                    print("✅ Time step is unconditionally stable and satisfies accuracy guidance.")
                elif accuracy_passed:
                    print("✅ Time step is stable and satisfies accuracy guidance.")
                else:
                    print("✅ Time step is stable; accuracy warning remains.")
                valid_dt = True

        # Realization & seed
        n_realizations = helper._get_int_input(
            "Number of realizations (default: 1): ",
            default=1, min_val=1, max_val=10
        )

        use_fixed_seed = helper._yesno("Use a fixed random seed for reproducibility?", default=True)
        seed_value = helper._get_int_input("Enter integer seed value (default: 42): ", default=42) if use_fixed_seed else None

        # Create TimeDomainConfig object to pass back
        time_domain_config = TimeDomainConfig(
            duration=duration,
            dt=dt,
            n_realizations=n_realizations,
            integration_method=newmark_method,
            random_seed=seed_value
        )

    else:
        # Frequency-domain only
        duration = 600.0
        dt = 0.02
        n_realizations = 0  # No time series
    
    return {
        'analysis_type': analysis_type,
        'duration': duration,
        'dt': dt,
        'n_realizations': n_realizations,
        'time_domain_config': time_domain_config
    }


def print_results_summary(results: dict):
    """
    Print summary of analysis results.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from SpectralVIVAnalysis
    """
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    if 'spectrum' in results:
        spectrum = results['spectrum']
        print(f"\nGeneralized Force Spectrum:")
        print(f"  S_Qn(f_n) = {spectrum.spectrum_at_fn:.3e} N²·s")
        print(f"  Modal RMS = {spectrum.modal_rms:.3e} N")
        print(f"  Coherence method: {spectrum.coherence_method}")
        print(f"  Computation: {spectrum.method}")
    
    if 'time_series' in results and results['time_series']:
        ts = results['time_series']
        n_real = len(ts['realizations'])
        print(f"\nTime Series:")
        print(f"  Realizations: {n_real}")
        print(f"  Duration: {ts['grid'].T:.1f} s")
        print(f"  Time step: {ts['grid'].dt:.4f} s")
        print(f"  Samples: {ts['grid'].N}")
    
    if 'response' in results:
        print(f"\nStructural Response:")
        print(f"  (Response analysis results)")
    
    print("=" * 70)