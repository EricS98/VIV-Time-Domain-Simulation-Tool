# scripts/run_analysis.py
"""
VIV Spectral Analysis Runner
============================

Script for running VIV spectral analysis
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Setup project path directly
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent  # scripts/ -> project_root/
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from common.structures import StructureProperties, StructureDatabase
from common.interactive_utils import InteractiveInputHelpers
from common.analysis_config_helpers import(
    ConfigBuilder,
    get_analysis_params_interactive,
    print_results_summary
)

# Optional file picker
try:
    from common.file_picker import pick_directory
    HAS_PICKER = True
except ImportError:
    HAS_PICKER = False
    pick_directory = None

class RunAnalysis(InteractiveInputHelpers):
    """Main runner for VIV analysis."""

    def __init__(self):
        self.structure = None
        self.output_dir = None
        self.config_builder = ConfigBuilder(interactive_helper=self)

    def setup_output_directory(self, custom_output: Optional[str] = None,
                                use_picker: bool = False) -> Path:
        """Setup output directory."""
        default_out = (project_root / "analysis_results").resolve()

        if use_picker and HAS_PICKER:
            chosen = pick_directory(title="Choose output folder",
                                    initialdir=default_out.parent)
            output_dir = chosen if chosen else default_out
        elif custom_output:
            output_dir = Path(custom_output).resolve()
        else:
            if HAS_PICKER and self._yesno("Use file picker for output folder?", default=False):
                chosen = pick_directory(title="Choose output folder",
                                        initialdir=default_out.parent)
                output_dir = chosen if chosen else default_out
            else:
                output_dir = default_out

        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir
        return output_dir
    
    def interactive_mode(self):
        """Run complete interactive analysis session."""
        print("=" * 70)
        print(" VIV ANALYSIS")
        print("=" * 70)

        # Analysis mode selection
        print("\nAnalysis mode:")
        print("1. Single-structure analysis")
        print("2. Multi-structure analysis")

        mode = self._get_user_choice("Select mode (1-2) (default: 1): ", [1, 2], default=1)

        if mode == 1:
            self._run_single_structure()
        else:
            self._run_multi_structure()

    def command_line_mode(self, args):
        """Run analysis from command line arguments."""
        print("=" * 70)
        print(" VIV ANALYSIS")
        print("=" * 70)

        # Load structure
        try:
            self.structure = self._load_structure_from_args(args)
            print(f"✅ Structure loaded: {self.structure.name}")
        except Exception as e:
            print(f"❌ Failed to load structure: {e}")
            return
        
        # Create config
        try:
            config = ConfigBuilder.build_from_args(args, self.structure)
            print(f"✅ Configuration created")
        except Exception as e:
            print(f"❌ Failed to create configuration: {e}")
            return
        
        # Analysis parameters from args
        analysis_params = {
            'analysis_type': args.analysis_type,
            'duration': args.duration,
            'dt': args.dt,
            'n_realizations': args.n_realizations
        }

        # Run analysis
        self._run_analysis(config, analysis_params)
    
    def _handle_structure_input(self) -> bool:
        """Handle structure input selection."""
        print("\nStructure input:")
        print("1. Load from CSV file")
        print("2. Enter parameters manually")
        
        input_mode = self._get_user_choice("Select input mode (1-2): ", [1, 2], default=1)
        
        if input_mode == 1:
            return self._handle_csv_input()
        else:
            return self._handle_manual_input()
    
    def _handle_csv_input(self) -> bool:
        """Handle CSV structure input"""
        return self.handle_csv_structure_input(self, project_root)
    
    def _handle_manual_input(self) -> bool:
        """Handle manual structure input."""
        return self.handle_manual_structure_input(self)
    
    def _load_structure_from_args(self, args) -> StructureProperties:
        """Load structure from command line arguments."""
        if not args.csv:
            raise ValueError("--csv argument is required")
        
        csv_path = Path(args.csv)
        if not csv_path.is_absolute():
            csv_path = (project_root / csv_path).resolve()
        
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        db = StructureDatabase(str(csv_path))
        
        if args.structure:
            structure = db.get_structure(args.structure)
            if structure is None:
                available = db.get_structure_names()
                raise ValueError(
                    f"Structure '{args.structure}' not found. "
                    f"Available: {available}"
                )
        else:
            structures = db.get_all_structures()
            if not structures:
                raise ValueError("No structures found in CSV")
            structure = structures[0]
            print(f"ℹ️  No structure specified, using: {structure.name}")
        
        return structure
    
    def _run_analysis(self, config, analysis_params):
        """Execute the analysis."""
        print("\n" + "=" * 70)
        print("RUNNING ANALYSIS...")
        print("=" * 70)

        try:
            from applications.spectral_viv_analysis import SpectralVIVAnalysis

            tdc = analysis_params.get('time_domain_config')
            if tdc is not None:
                config.time_domain_config = tdc
                 
            analyzer = SpectralVIVAnalysis(
                self.structure,
                config,
                output_dir=str(self.output_dir)
            )

            results = analyzer.run_complete_analysis(
                duration=analysis_params['duration'],
                dt=analysis_params['dt'],
                n_realizations=analysis_params['n_realizations'],
                analysis_type=analysis_params['analysis_type']
            )

            analyzer.run_interactive_sensitivities(self)

            print("\n✅ Analysis completed successfully!")
            print(f"Results saved to: {self.output_dir}")

            # Show summary
            if hasattr(self, '_yesno'): # Interactive mode
                if self._yesno("\nShow results summary?", default=True):
                    print_results_summary(results)
            else:   # Command line mode - always show
                print_results_summary(results)

        except Exception as e:
            print(f"\n❌ Analysis failed: {e}")
            import traceback
            traceback.print_exc()

    def _run_single_structure(self):
        """Run analysis for a single structure."""
        # Step 1: Structure Input
        if not self._handle_structure_input():
            print("❌ Failed to load structure")
            return
        
        print(f"\n✅ Structure loaded: {self.structure.name}")

        # Step 2: Model selection
        print("\nSelect Analysis Model:")
        print("1. Spectral Model (Vickery & Basu)")
        print("2. Vortex-Resonance Model")

        model_choice = self._get_user_choice(
            "Select model (1-2) (default: 1): ", [1, 2], default=1
        )

        if model_choice == 1:
            config = self.config_builder.build_interactive(self.structure)
            if config is None:
                print("❌ Configuration failed")
                return

            analysis_params = get_analysis_params_interactive(self)
            if analysis_params is None:
                print("❌ Analysis parameters failed")
                return
            
            self._run_analysis(config, analysis_params)
            
        else:
            # Vortex-resonance path
            vr_config = self.config_builder.build_vortex_resonance_config(self.structure)
            if vr_config is None:
                print("❌ Configuration failed")
                return
            
            self._run_single_vortex_resonance_analysis(self.structure, vr_config)

    def _run_multi_structure(self):
        """Run batch analysis for multiple structures."""
        print("\n" + "=" * 70)
        print("MULTI-STRUCTURE BATCH ANALYSIS")
        print("=" * 70)

        # Step 1: Load CSV file
        csv_path = self._get_csv_file_path(project_root)
        if csv_path is None:
            print("❌ Failed to load CSV file")
            return
        
        # Load all structures
        db = StructureDatabase(str(csv_path))
        structures = db.get_all_structures()

        if not structures:
            print("❌ No structures found in CSV")
            return
        
        print(f"\n✅ Loaded {len(structures)} structures")
        for i, s in enumerate(structures, 1):
            print(f"  {i}. {s.name}")

        # Step 2: Model selection
        print("\nSelect Analysis Model:")
        print("1. Spectral Model (Vickery & Basu)")
        print("2. Vortex-Resonance Model")

        model_choice = self._get_user_choice(
            "Select model (1-2): ", [1, 2], default=1
        )

        if model_choice == 1:
            self._run_multi_structure_spectral(structures, db, csv_path)
        else:
            self._run_multi_structure_vortex_resonance(structures, db, csv_path)
        
    def _run_single_structure_spectral(self, structure, config, domain_type, time_params):
        """Run analysis for one structure in multi-structure mode."""
        try:
            from applications.spectral_viv_analysis import SpectralVIVAnalysis
            analyzer = SpectralVIVAnalysis(
                structure,
                config,
                output_dir=None
            )

            results = {}

            # Frequency-domain
            if domain_type in ['frequency', 'both']:
                freq_result = analyzer.run_complete_analysis(
                    analysis_type='frequency_domain', # Frequency-domain response
                    create_plots=False,
                    create_height_plots=False
                )
                results['frequency'] = freq_result

            # Time-domain
            if domain_type in ['time', 'both']:
                duration = None
                dt = None
                n_realizations = 1

                if time_params is not None:
                    duration = time_params.get('duration', None)
                    dt = time_params.get('dt', None)
                    n_realizations = time_params.get('n_realizations', 1)

                time_result = analyzer.run_complete_analysis(
                    duration=duration,
                    dt=dt,
                    n_realizations=n_realizations,
                    analysis_type='time_domain',    # Full time-domain response
                    create_plots=False,
                    create_height_plots=False
                )
                results['time'] = time_result

            return results
    
        except Exception as e:
            print(f"❌ Analysis failed for {structure.name}: {e}")
            import traceback
            traceback.print_exc()
            return None
        
    def _run_single_vortex_resonance_analysis(self, structure, config):
        """Run vortex-resonance model analysis."""
        try:
            from applications.vortex_resonance_analysis import VortexResonanceAnalysis
            vr_analysis = VortexResonanceAnalysis(
                structure=structure,
                config=config,
                output_dir=self.output_dir
            )
            result = vr_analysis.run_vortex_resonance_analysis(structure, config)
        
            if result:
                print("\n" + "="*70)
                print("ANALYSIS RESULTS")
                print("="*70)
                result.print_summary()

            return result
        
        except Exception as e:
            print(f"\n❌ Analysis failed: {e}")
            import traceback
            traceback.print_exc()
        
    def _run_multi_structure_spectral(self, structures, db, csv_path):
        """Run spectral analysis for multiple structures."""

        # Step 2: Select analysis domain
        print("\nAnalysis domain:")
        print("1. Frequency-domain only")
        print("2. Time-domain only")
        print("3. Both frequency and time domain")

        domain_choice = self._get_user_choice("Select domain (1-3): ", [1, 2, 3], default=1)
        domain_type = ['frequency', 'time', 'both'][domain_choice - 1]

        # Step 3: Wind velocity handling for multi-structure
        print("\n" + "=" * 70)
        print("WIND SPEED CONFIGURATION")
        print("=" * 70)
        wind_mode = 1
        print("✅ Wind speed set to calculate $u_{crit} = f_n \\cdot d_{ref} / St$ individually for each structure.")

        # Step 4: Get common configuration parameters
        print("\n" + "=" * 70)
        print("COMMON CONFIGURATION (applied to all structures)")
        print("=" * 70)

        # Use first structure as template for config
        template_config = self.config_builder.build_interactive(structures[0], is_template_mode=True)
        if template_config is None:
            print("❌ Configuration failed")
            return
        
        # Step 5: Get time-domain parameters (automatic in multi-structure mode)
        time_params = None
        if domain_type in ['time', 'both']:
            print("\nTime-domain parameters (multi-structure mode):")
            print("  → Duration T and time step Δt are chosen automatically for each structure:")
            print("     • T = max(600 s, 1000·T_n)")
            print("     • Δt from f_max and Gaussian bell coverage")
            print("  → Integration: Newmark (constant/average acceleration) from TimeDomainConfig default")

            time_params = {
                'duration': None,      # let TimeGridBuilder choose T
                'dt': None,           # let TimeGridBuilder choose Δt
                'n_realizations': 1 
            }

        # Step 6: Run analysis for all structures
        results = {
            'structures': [],
            'frequency_domain': [] if domain_type in ['frequency', 'both'] else None,
            'time_domain': [] if domain_type in ['time', 'both'] else None
        }

        for i, structure in enumerate(structures, 1):
            print(f"\n{'=' * 70}")
            print(f"ANALYZING STRUCTURE {i}/{len(structures)}: {structure.name}")
            print(f"{'=' * 70}")

            # Create config for this structure (reuse template settings)
            if wind_mode == 1:
                # Use u_crit for each structure (recalculated in build_from_template)
                config = self.config_builder.build_from_template(template_config, structure)
            else:
                # Ask for wind speed for each structure
                config = self.config_builder.build_from_template(template_config, structure)
                u_ref = self._get_float_input(f"Wind speed for {structure.name} [m/s]",
                                              default=config.wind_profile.u_ref)
                config.wind_profile.u_ref = u_ref
                print(f"  → Using u_ref = {u_ref:.2f} m/s")

            # Run analysis for this single structure
            structure_results = self._run_single_structure_spectral(
                structure, config, domain_type, time_params
            )

            if structure_results:
                results['structures'].append(structure.name)
                if results['frequency_domain'] is not None:
                    results['frequency_domain'].append(structure_results.get('frequency'))
                if results['time_domain'] is not None:
                    results['time_domain'].append(structure_results.get('time'))

        # Step 7: Generate comparison plots
        print("\n" + "=" * 70)
        print("GENERATING COMPARISON PLOTS")
        print("=" * 70)
        
        results['domain_type'] = domain_type
        results['config'] = template_config     # Store config for plotting

        if self._yesno("\nGenerate comparison plot?", default=True):
            self._generate_comparison_plots(structures, results, analysis_type='spectral')

        # Optional: generate LaTeX table (only if frequency-domain results exist)
        if domain_type in ['frequency', 'both'] and results['frequency_domain'] or \
            (domain_type in ['time', 'both'] and results['time_domain']):
            from applications.spectral_viv_analysis import SpectralVIVAnalysis

            if self._yesno("\nGenerate LaTeX table from spectral results?", default=False):
                # Combine frequency and time domain results for each structure
                combined_results = []
                structures_for_table = []
        
                for i, structure in enumerate(structures):
                    result_dict = {}

                    # Add frequency-domain result if available
                    if results['frequency_domain'] and i < len(results['frequency_domain']):
                        fd_res = results['frequency_domain'][i]
                        if fd_res and fd_res.get('response') is not None:
                            result_dict['frequency'] = fd_res
            
                    # Add time-domain result if available
                    if results['time_domain'] and i < len(results['time_domain']):
                        td_res = results['time_domain'][i]
                        if td_res and td_res.get('time_domain') is not None:
                            result_dict['time'] = td_res
            
                    # Only include structures that have at least one valid result
                    if result_dict:
                        combined_results.append(result_dict)
                        structures_for_table.append(structure)

                if structures_for_table:
                    SpectralVIVAnalysis.generate_latex_table(
                        structures_for_table,
                        combined_results,
                        self.output_dir
                    )
                else:
                    print("⚠️ No valid results for LaTeX table.")

        print("\n✅ Multi-structure analysis completed!")
        print(f"Results saved to: {self.output_dir}")

        
    def _run_multi_structure_vortex_resonance(self, structures, db, csv_path):
        """Run vortex-resonance analysis for multiple structures."""
        from applications.vortex_resonance_analysis import VortexResonanceAnalysis

        print("\n" + "=" * 70)
        print("COMMON CONFIGURATION (applied to all structures)")
        print("=" * 70)
        print("ℹ️  Wind speed automatically set to u_crit for each structure")

        # Use first structure as template for config
        template_config = self.config_builder.build_vortex_resonance_config(structures[0])
        if template_config is None:
            print("❌ Configuration failed")
            return
        
        # Run analysis for all structures
        results = {
            'structures': [],
            'vortex_resonance': []
        }

        for i, structure in enumerate(structures, 1):
            print(f"\n{'=' * 70}")
            print(f"ANALYZING STRUCTURE {i}/{len(structures)}: {structure.name}")
            print(f"{'=' * 70}")

            # Create config for this structure
            config = self.config_builder.build_from_template(template_config, structure)

            # Run vortex-resonance analysis
            vr_result = self._run_single_vortex_resonance_analysis(structure, config)

            if vr_result:
                results['structures'].append(structure.name)
                results['vortex_resonance'].append({
                    'response': vr_result
                })

        # Generate comparison plots
        print("\n" + "=" * 70)
        print("GENERATING COMPARISON PLOTS")
        print("=" * 70)
    
        if self._yesno("\nGenerate comparison plot?", default=True):
            self._generate_comparison_plots(structures, results, analysis_type='vortex_resonance')

        if self._yesno("\nGenerate LaTeX table?", default=False):
            vr_results_only = [r['response'] for r in results ['vortex_resonance'] if r and 'response' in r]
            VortexResonanceAnalysis.generate_latex_table(structures, vr_results_only, self.output_dir) 
    
        print("\n✅ Multi-structure vortex-resonance analysis completed!")
        print(f"Results saved to: {self.output_dir}")
                
    def _generate_comparison_plots(self, structures, results, analysis_type='spectral'):
        """
        Generate comparison plots for multi-structure analysis.
    
        Parameters:
        -----------
        structures : list
            List of StructureProperties objects
        results : list or dict
            For spectral: dict with 'frequency_domain' and 'time_domain' keys
            For vortex-resonance: list of VortexResonanceResults objects
        analysis_type : str
            'spectral' or 'vortex_resonance'
        """
        try:
            from visualization.multi_structure_plotter import MultiStructurePlotter
            from config.aerodynamic_parameters.amplitude_dependent_damping import (
                VickeryBasuDamping, EurocodeDamping
            )
        
            plotter = MultiStructurePlotter(self.output_dir)
            made_any = False
        
            if analysis_type == 'spectral':
                # Original spectral logic
                domain_type = results.get('domain_type', 'frequency') 

                freq_list = []
                time_list = []
                valid_freq = 0
                valid_time = 0
            
                if domain_type in ['frequency', 'both']:
                    freq_list = results['frequency_domain'] or []
                    valid_freq = sum(
                        1 for s, r in zip(structures, freq_list)
                        if r and r.get('response') and (getattr(s, 'measured_y_d', None) is not None or 
                                                        getattr(s, 'measured_y_d_rare', None) is not None)
                    )
                    if valid_freq > 0:
                        plotter.plot_comparison(structures, freq_list, domain_type='frequency')
                        made_any = True

                if domain_type in ['time', 'both']:
                    time_list = results['time_domain'] or []
                    valid_time = sum(
                        1 for s, r in zip(structures, time_list)
                        if r and r.get('time_domain') and (getattr(s, 'measured_y_d', None) is not None or 
                                                            getattr(s, 'measured_y_d_rare', None) is not None)
                    )
                    if valid_time > 0:
                        plotter.plot_comparison(structures, time_list, domain_type='time')
                        made_any = True

                if domain_type == 'both' and valid_freq > 0 and valid_time > 0:
                    plotter.plot_combined_comparison(structures, freq_list, time_list)
                    made_any = True

                if domain_type == 'both' and freq_list and time_list:
                    config = results.get('config')
                    if config is not None:
                        damping_formulation = config.damping_formulation
                        # Check if damping formulation is compatible
                        if isinstance(damping_formulation, (VickeryBasuDamping, EurocodeDamping)):
                            print("\n  Generating peak factor comparison plot...")
                            plotter.plot_peak_factor_comparison(structures, freq_list, time_list)
                            made_any = True
                        else:
                            print(f"\n  ℹ️  Peak factor plot not applicable for {damping_formulation.get_name()}")
                        
                
            elif analysis_type == 'vortex_resonance':
                # Convert vortex-resonance results to frequency-domain format
                vr_results_list = results.get('vortex_resonance', [])

                valid_results = sum(
                    1 for s, r in zip(structures, vr_results_list)
                    if r and r.get('response') and (getattr(s, 'measured_y_d', None) is not None or 
                                                    getattr(s, 'measured_y_d_rare', None) is not None)
                )

                if valid_results > 0:
                    plotter.plot_comparison(structures, vr_results_list, domain_type='frequency', analysis_type='vortex_resonance')
                    made_any = True

            if made_any:
                print("✅ Comparison plots generated")
            else:
                print("⚠️  No comparison plots generated (no valid data)")

        except Exception as e:
            print(f"⚠️  Could not generate comparison plots: {e}")
            import traceback
            traceback.print_exc()

def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Unified VIV Analysis Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python scripts/run_analysis.py
  
  # Command line mode
  python run_analysis.py --csv data/structures.csv --structure "Chimney_1"
        """
    )

    # Structure input
    input_group = parser.add_argument_group('Structure Input')
    input_group.add_argument('--csv', type=str, help='CSV file with structure data')
    input_group.add_argument('--structure', type=str, help='Structure name from CSV')
    input_group.add_argument('--manual', action='store_true', help='Use manual structure input')

    # Output options
    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--output-dir', type=str, help='Output directory')
    output_group.add_argument('--pick-output-dir', action='store_true',
                       help='Use file picker for output directory')
    output_group.add_argument('--save-plots', action='store_true', help='Save plots to files')
    output_group.add_argument('--show-plots', action='store_true', help='Show interactive plots')
    output_group.add_argument('--pick-csv', action='store_true', 
                       help='Use file picker for CSV selection')
    
    # Analysis Type
    analysis_group = parser.add_argument_group('Analysis Type')
    analysis_group.add_argument(
        '--analysis-type',
        type=str,
        choices=['spectrum', 'timeseries', 'response', 'all'],
        default='spectrum',
        help='Type of analysis to perform'
    )

    # Wind Profile
    wind_group = parser.add_argument_group('Wind Profile')
    wind_group.add_argument(
        '--wind-type',
        type=str,
        choices=['constant', 'power-law', 'terrain'],
        default='constant',
        help='Wind profile type'
    )
    wind_group.add_argument('--u-ref', type=float, default=None,
                            help='Reference wind speed [m/s]')
    wind_group.add_argument('--z-ref', type=float, default=10.0, 
                            help='Reference height [m]')
    wind_group.add_argument('--Iv', type=float, default=0.1, 
                            help='Turbulence intensity [-] (constant wind)')
    wind_group.add_argument('--alpha', type=float, 
                            help='Power law exponent')
    wind_group.add_argument('--terrain', type=str, choices=['I', 'II', 'III'],
                            default='II', help='Terrain category')

    # Cross-section
    section_group = parser.add_argument_group('Cross Section')
    section_group.add_argument('--taper', action='store_true',
                                help='Use tapered cross-section')
    section_group.add_argument('--d-base', type=float, 
                               help='Base diameter [m] (if tapered)')
    section_group.add_argument('--d-top', type=float, 
                               help='Top diameter [m] (if tapered)')
    
    # Reference diameter
    d_ref_group = parser.add_argument_group('Reference Diameter')
    d_ref_group.add_argument(
        '--d-ref-method',
        type=str,
        choices=['top', 'top_third', 'effective'],
        default='top_third',
        help='Reference diameter calculation method (default: top_third)'
    )
    d_ref_group.add_argument(
        '--d-ref-n-points',
        type=int,
        default=200,
        help='Integration points for top-third or effective methods (default: 100)'
    )

    # Coherence function
    coherence_group = parser.add_argument_group('Coherence Function')
    coherence_group.add_argument('--coherence', type=str, choices=['constant', 'vickery-clark'],
                                default='constant', help='Coherence function type')
    coherence_group.add_argument('--lambda-factor', type=float,
                                default=1.0, help='Correlation length factor')

    # Aerodynamic parameters
    aero_group = parser.add_argument_group('Aerodynamic Parameters')
    aero_group.add_argument('--lift-coefficient', type=str, choices=['CICIND'],
                            default='CICIND',help='Lift coefficient')
    aero_group.add_argument('--damping-coefficient', type=str, choices=['CICIND'],
        default='CICIND', help='Aerodynamic damping coefficient')
    aero_group.add_argument('--aerodynamic-damping-modification', type=str,
        choices=['simplified', 'full', 'none'], help='Aerodynamic damping modification method')
    aero_group.add_argument('--St', type=float, default=0.2, help='Strouhal number')

    # Aerodynamic damping model
    aeor_damping_group = parser.add_argument_group('Aerodynamic Damping Model')
    aeor_damping_group.add_argument('--damping-model', type=str, choices=['Vickery-Basu', 'Lupi', 'Eurocode-draft'], 
                                    help='Aerodynamic damping model')

    # Time series generation
    time_group = parser.add_argument_group('Time Series Generation')
    time_group.add_argument('--duration', type=float, default=600.0, 
                            help='Duration [s]')
    time_group.add_argument('--dt', type=float, default=0.05, 
                            help='Time step [s]')
    time_group.add_argument('--n-realizations', type=int, default=1, 
                            help='Number of realizations')
    time_group.add_argument('--seed', type=int, default=42, 
                            help='Random seed')

    return parser

    
def main():
    """Main execution function."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    runner = RunAnalysis()

    # Setup output directory
    runner.setup_output_directory(
        custom_output=args.output_dir,
        use_picker=args.pick_output_dir
    )
    
    print("=" * 70)
    print(" VIV SPECTRAL ANALYSIS")
    print("=" * 70)
    print(f"Output directory: {runner.output_dir}")
    
    # Check if interactive mode (no --csv argument)
    if not args.csv and not args.manual:
        print("\n🎯 Starting interactive mode...")
        runner.interactive_mode()
        return
    
    # Command line mode
    print("\n🎯 Command line mode")
    runner.command_line_mode(args)
    
if __name__ == "__main__":
    main()