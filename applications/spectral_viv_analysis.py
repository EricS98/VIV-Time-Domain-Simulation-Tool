# applications/spectral_viv_analysis.py
"""
Spectral VIV Analysis Module
============================
"""
import numpy as np
from typing import Optional
from pathlib import Path
from copy import deepcopy

from config.reference_diameter import ReferenceDiameterCalculator
from calculators.lift_spectrum import LiftSpectrumCalculator
from calculators.generalized_force_spectrum import GeneralizedForceSpectrumCalculator
from calculators.frequency_domain_response import FrequencyDomainResponseCalculator
from calculators.time_domain_response import TimeDomainResponseCalculator
from common.time_domain_base import TimeGridBuilder, TimeSeriesSynthesizer

# Import visualization
try:
    from visualization.height_profile_data import HeightProfileData
    from visualization.height_profile_plotter import HeightProfilePlotter
    HEIGHT_VIS_AVAILABLE = True
except ImportError:
    HEIGHT_VIS_AVAILABLE = False
    print("ℹ️  Height profile visualization not available")

# Import time-domain visualization
try:
    from visualization.time_domain_plotter import TimeDomainResponsePlotter
    TIME_DOMAIN_VIS_AVAILABLE = True
except ImportError:
    TIME_DOMAIN_VIS_AVAILABLE = False
    print("ℹ️  Time-domain visualization not available")

class SpectralVIVAnalysis:
    """
    Complete spectral VIV analysis workflow.
    """

    def __init__(self, structure, config, output_dir=None, verbose=True):
        self.structure = structure
        self.config = config
        self.output_dir = output_dir
        self.verbose = verbose

        # Initialize calculators
        self.lift_calc = LiftSpectrumCalculator(St=self.config.St)
        self.force_calc = GeneralizedForceSpectrumCalculator(St=self.config.St)
        self.response_calc = FrequencyDomainResponseCalculator(St=self.config.St)
        self.grid_builder = TimeGridBuilder()

        # Initialize plotters if available
        self.height_profile_plotter = HeightProfilePlotter() if HEIGHT_VIS_AVAILABLE else None
        self.time_domain_plotter = TimeDomainResponsePlotter() if TIME_DOMAIN_VIS_AVAILABLE else None

        # Storage for height-dependent data
        self.height_profile_data = None

    def run_complete_analysis(self, duration=600.0, dt=0.02, n_realizations=1, 
                              analysis_type='both', seed=42,
                              create_plots:bool = True,
                              create_height_plots:bool=True, n_points=100):
        """Run complete analysis: spectrum + time series + response."""

        print("\n" + "=" * 70)
        print("RUNNING COMPLETE SPECTRAL VIV ANALYSIS")
        print("=" * 70)

        # 1. Create simulation grid
        grid = self.force_calc.create_grid(
            self.structure, self.config, duration=duration, dt=dt
        )

        # 2. Calculate generalized force spectrum
        spectrum_result = self.force_calc.calculate_spectrum(
            self.structure, self.config, grid=grid
        )

        # Initialize result dictionary
        results = {
            'spectrum': spectrum_result, 
            'grid': grid,
            'analysis_type': analysis_type
        }

        # 3. Generate time series realizations
        if analysis_type in ['time_domain', 'both']:
            time_series = self._generate_time_series(
                spectrum_result, grid, n_realizations, seed
            )
            results['time_series'] = time_series

        # 4. Calculate response if requested
        if analysis_type in ['frequency_domain', 'both']:
            print("\n" + "="*70)
            print("CALCULATING FREQUENCY-DOMAIN RESPONSE")
            print("="*70)

            response_result = self._calculate_response(grid=grid)
            results['response'] = response_result

            # Print summary
            print(f"\nResponse Summary:")
            print(f"  RMS displacement:                 {response_result.sigma_y*1000:.2f} mm")
            print(f"  Peak response:                    {response_result.y_max*1000:.2f} mm")
            print(f"  Aerodynamic damping decrement:    {response_result.delta_a:.3f}")
            print(f"  Total damping:                    {response_result.delta_total:.3f}")
            print(f"  Converged:                        {response_result.converged}")

        # 5. Calculate reference diameter d_ref
        d_ref = ReferenceDiameterCalculator.calculate(
            config=self.config.d_ref_config,
            cross_section=self.config.cross_section,
            height=self.structure.height,
            d_nominal=self.structure.diameter,
            mode_shape=self.config.mode_shape,
            mode_number=1
        )
        print(f"\n📏 Reference diameter: {d_ref:.3f} m (method: {self.config.d_ref_config.method.value})")


        # 6. Calculate time-domain response
        if analysis_type in ['time_domain', 'both']:
            print("\n" + "="*70)
            print("CALCULATING TIME-DOMAIN RESPONSE")
            print("="*70)

            td_calculator = TimeDomainResponseCalculator(St=self.config.St)
            td_result = td_calculator.calculate_response(
                structure=self.structure,
                config=self.config,
                d_ref=d_ref,
                grid=grid,
                duration=duration,
                dt=dt,
                random_seed=self.config.time_domain_config.random_seed
            )
            results['time_domain'] = td_result

            # Optionally compare with frequency domain if both exist
            if 'response' in results:
                comparison = td_calculator.compare_with_frequency_domain(
                    td_result, results['response']
                )
                results['comparison'] = comparison

            # Time-domain plots (only if explicitly enabled)
            if (
                create_plots
                and TIME_DOMAIN_VIS_AVAILABLE
                and self.time_domain_plotter is not None
            ):
                self._create_time_domain_plots(
                    td_result,
                    fd_result=results.get("response"),
                    spectrum_result=results.get("spectrum"),
                    output_dir=Path(self.output_dir) if self.output_dir else None,
                )

        # 7. Create height profile data and visualizations if requested
        if create_height_plots and HEIGHT_VIS_AVAILABLE:
            print("\n" + "="*70)
            print("Creating height profile visualizations")
            print("="*70)

            # Extract height-dependent data
            self.height_profile_data = self._extract_height_profile_data(
                n_points=n_points
            )
            results['height_profile_data'] = self.height_profile_data

            # Create plots
            output_path = Path(self.output_dir) if self.output_dir else None
            self._create_height_plots(output_path)

        # 8. Save results
        if self.output_dir:
            self.save_results(results)

        print("\nSpectral VIV analysis completed.")
        return results
    
    def _generate_time_series(self, spectrum, grid, n_realizations, seed=42):
        """Generate time series from spectrum."""
        realizations = []
        for i in range(n_realizations):
            series = TimeSeriesSynthesizer.synthesize_from_psd(
                spectrum.spectrum, grid.df, grid.N, random_seed=seed+i
            )
            realizations.append(series)
        return {
            'time': grid.t,
            'realizations': realizations,
            'grid': grid
        }
    
    def _calculate_response(self, grid=None, tolerance=1e6, verbose=None):
        """
        Calculate frequency-domain VIV response.

        Parameters:
        -----------
        grid : SimulationGrid, optional
            Pre-computed frequency grid. If None, creates a new one.
        tolerance : float
            Convergence tolerance for sigma_y
        verbose : bool, optional
            If provided, overrides self.verbose for this calculation
        
        Returns:
        --------
        FrequencyDomainResults
            Complete response results with sigma_y, damping ratios, etc.
        """
        _verbose = verbose if verbose is not None else self.verbose

        # Calculate reference diameter
        d_ref = ReferenceDiameterCalculator.calculate(
            config=self.config.d_ref_config,
            cross_section=self.config.cross_section,
            height=self.structure.height,
            d_nominal=self.structure.diameter,
            mode_shape=self.config.mode_shape,
            mode_number=1
        )
        if _verbose:
            print(f"\n📏 Reference diameter: {d_ref:.3f} m (method: {self.config.d_ref_config.method.value})")
        
        a_L = 0.4
        if hasattr(self.config.damping_formulation, 'a_L'):
            a_L = self.config.damping_formulation.a_L

        # Calculate response using frequency-domain calculator
        response = self.response_calc.calculate_response(
            structure=self.structure,
            config=self.config,
            d_ref=d_ref,
            a_L=a_L,
            grid=grid,
            tolerance=tolerance,
            verbose=_verbose
        )

        return response
    
    def _extract_height_profile_data(self, n_points=100) -> HeightProfileData:
        """
        Extract height-dependent properties for visualization.

        This method evaluates all height-varying properties at discrete
        points along the structure height, capturing exactly what would 
        be used in the analysis.

        Parameters:
        -----------
        n_points : int
            Number of discretization points along height
            
        Returns:
        --------
        HeightProfileData
            Container with all height-dependent properties
        """
        h = self.structure.height
        d_nominal = self.structure.diameter
        f_n = self.structure.f_n
        St = self.lift_calc.St      

        # Create height discretization
        z_points = np.linspace(0, h, n_points)

        # Evaluate structural properties
        d_z = self.config.cross_section.get_diameter(z_points, h, d_nominal)
        phi_z = self.config.mode_shape.evaluate(z_points, h, mode_number=1)

        # Evaluate wind properties
        u_z = self.config.wind_profile.get_velocity(z_points, h)
        Iv_z = self.config.wind_profile.get_turbulence_intensity(z_points, h)
        
        # Calculate derived properties
        f_s_z = St * u_z / d_z      # Shedding frequency
        U_cr_z = f_n * d_z / St     # Critical velocity
        velocity_ratio_z = u_z / U_cr_z     # Velocity ratio

        # Get bandwidth parameter (constant or at reference height)
        B = np.minimum(0.1 + Iv_z, 0.35)

        try:
            # Calculate Reynolds number at each height
            nu_air = self.lift_calc.nu_air
            Re_z = u_z * d_z / nu_air
            
            # Get aerodynamic parameters at each height
            sigma_cl_z = np.array([self.config.lift_coefficient.get_sigma_cl(Re) 
                                   for Re in Re_z])
            
            if hasattr(self.config, 'damping_parameter') and hasattr(self.config, 'damping_modification'):
                # Step 1: Get Ka_max depending of the Reynold number
                Ka_max_z = np.array([self.config.damping_parameter.get_ka_max(Re) 
                                    for Re in Re_z])
                
                # Step 2: Apply turbulence and velocity ratio modification to get Ka_eff
                Ka_z = np.array([
                    self.config.damping_modification.apply(Ka_max, Iv, u_ucr)
                    for Ka_max, Iv, u_ucr in zip(Ka_max_z, Iv_z, velocity_ratio_z)
                ])
                # Debug: Show Ka range
                print(f"\n  Ka calculation debug:")
                print(f"    V/Vcr range: [{velocity_ratio_z.min():.3f}, {velocity_ratio_z.max():.3f}]")
                print(f"    Iv range: [{Iv_z.min():.3f}, {Iv_z.max():.3f}]")
                print(f"    Ka_max range: [{Ka_max_z.min():.3f}, {Ka_max_z.max():.3f}]")
                print(f"    Ka_eff range: [{Ka_z.min():.6f}, {Ka_z.max():.6f}]")
                if Ka_z.min() < 0:
                    print("Negative Ka values present (positive aerodynamic damping)")
                else:
                    print(f"⚠️  No negative Ka values (check if this is correct)")
            else:
                print("⚠️  Damping parameter or modification not configured")

        except Exception as e:
            print(f"ℹ️  Could not calculate aerodynamic parameters: {e}")
            import traceback
            traceback.print_exc()
        
        # Package into HeightProfileData
        return HeightProfileData(
            z=z_points,
            d=d_z,
            phi=phi_z,
            u=u_z,
            Iv=Iv_z,
            B=B,
            f_s=f_s_z,
            U_cr=U_cr_z,
            velocity_ratio=velocity_ratio_z,
            sigma_cl=sigma_cl_z,
            Ka=Ka_z
        )
    
    def _create_height_plots(self, output_dir: Optional[Path] = None):
        """
        Create height profile visualization plots.
        
        Parameters:
        -----------
        output_dir : Optional[Path]
            Directory to save plots. If None, plots are not saved.
        """
        if self.height_profile_plotter is None or self.height_profile_data is None:
            print("⚠️  Cannot create plots: plotter or data not available")
            return
        
        structure_name = self.structure.name.replace(' ', '_')

        try:
            # Create both structural/wind and spectral parameter plots
            path = self.height_profile_plotter.plot_height_profiles(
                data=self.height_profile_data,
                structure_name=structure_name,
                output_dir=output_dir,
                show_plot=False
            )
            print(f"✅ Height profile plot created: {path.name}")
                
        except Exception as e:
            print(f"⚠️  Error creating height profile plots: {e}")
            import traceback
            traceback.print_exc()

    def _create_spectrum_timeseries_plots(self, results):
        """
        Create spectrum and time-series visualization plots.

        Parameters:
        -----------
        results: dict
            Analysis results dictionary containing 'spectrum'
            and optionally 'time_series'
        """
        if self.spectrum_plotter is None:
            print("⚠️  Cannot create spectrum/time-series plots: plotter not available")
            return
    
        print("\n" + "="*70)
        print("Creating Spectrum and Time Series Plots")
        print("="*70)

        output_dir = Path(self.output_dir) if self.output_dir else None

        try:
            # Determine what to plot based on analysis type
            has_spectrum = 'spectrum' in results and results['spectrum'] is not None
            has_timeseries = 'time_series' in results and results['time_series'] is not None

            if has_spectrum and has_timeseries:
                # Plot both spectrum and time-series
                spectrum_path, timeseries_path = self.spectrum_plotter.plot_both(
                    results,
                    realization_index=0,
                    show_plot=False,
                    output_dir=output_dir,
                    max_time=None   # Show full duration
                )
                if spectrum_path and timeseries_path:
                    print(f"✅ Spectrum and time-series plots created::")
                    print(f"   📊 Spectrum: {spectrum_path.name}")
                    print(f"   📊 Time-series: {timeseries_path.name}")
                else:
                    print("ℹ️  Plots created but not saved (no output directory)")
            
            elif has_spectrum:
                # Plot only spectrum
                spectrum_path = output_dir/"generalized_force_spectrum.png" if output_dir else None
                self.spectrum_plotter.plot_spectrum(
                    results,
                    show_plot=False,
                    save_path=spectrum_path
                )
                if spectrum_path:
                    print(f"✅ Spectrum plot created: {spectrum_path.name}")
                else:
                    print("ℹ️  No spectrum or time-series data available to plot")
        
        except Exception as e:
            print(f"⚠️  Error creating spectrum/time-series plots: {e}")
            import traceback
            traceback.print_exc()

    def _create_time_domain_plots(self, td_result, fd_result=None, spectrum_result=None, output_dir=None):
        """
        Create time-domain visualization plots.

        Parameters:
        -----------
         td_result : TimeDomainResults
            Time-domain response results
        fd_result : FrequencyDomainResults, optional
            Frequency-domain results for comparison
        spectrum_result : GeneralizedForceSpectrumResult, optional
            Spectrum result for force spectrum plot
        output_dir : Path, optional
            Directory to save plots
        """
        if self.time_domain_plotter is None:
            print("⚠️  Cannot create time-domain plots: plotter not available")
            return
        
        print("\n" + "="*70)
        print("Creating Time-Domain Visualization Plots")
        print("="*70)

        try:
            # Create report with all plots
            saved_paths = self.time_domain_plotter.create_comprehensive_report(
                td_results=td_result,
                fd_results=fd_result,
                spectrum_result=spectrum_result,
                output_dir=output_dir,
                show_plots=False,
                max_time_detail=100.0
            )

            if saved_paths:
                print(f"\n✅ Time-domain plots created successfully")
            else:
                print("ℹ️  Plots created but not saved (no output directory)")

        except Exception as e:
            print(f"⚠️  Error creating time-domain plots: {e}")
            import traceback
            traceback.print_exc()

    def generate_latex_table(structures, fd_results_list, output_dir: Optional[Path] = None):
        """
        Generate LaTeX table for spectral VIV results (frequency domain).

        Parameters
        ----------
        structures : list[StructureProperties]
            List of structures in the same order as fd_results_list
        fd_results_list : list[dict]
            List of results dicts returned by run_complete_analysis
            with analysis_type='frequency_domain'
        output_dir : Path, optional
            Directory where the .tex file will be written
        """
        if output_dir is None:
            print("⚠️ No output directory provided for LaTeX table")
            return

        if not fd_results_list:
            print("⚠️ No frequency-domain results to generate LaTeX table")
            return

        latex_lines = []
        latex_lines.append(r"\begin{table}[htbp]")
        latex_lines.append(r"  \centering")
        latex_lines.append(r"  \caption{Spectral VIV analysis results}")
        latex_lines.append(r"  \label{tab:spectral_viv_results}")
        latex_lines.append(r"  \begin{tabular}{l c c c c}")
        latex_lines.append(r"    \toprule")
        latex_lines.append(
            r"    Structure & $\sigma_y/d$ & $y_{\max}/d$ & $(y/d)_\text{meas}$ & Rel.\ error [\%] \\"
        )
        latex_lines.append(r"    \midrule")

        for structure, fd_result in zip(structures, fd_results_list):
            if fd_result is None:
                continue

            response = fd_result['response']
            d = structure.diameter

            # Calculate normalized responses
            sigma_over_d = response.sigma_y / d
            ymax_over_d = response.y_max / d

            meas_freq = getattr(structure, "measured_y_d", None)
            meas_rare = getattr(structure, "measured_y_d_rare", None)

            # Format frequent event measurement
            meas_freq_str = f"{meas_freq:.4f}" if (meas_freq is not None and meas_freq > 0) else r"-"
        
            # Format rare event measurement
            meas_rare_str = f"{meas_rare:.4f}" if (meas_rare is not None and meas_rare > 0) else r"-"

            # Calculate error based on frequent event (if available)
            if meas_freq is not None and meas_freq > 0:
                rel_err = 100.0 * (sigma_over_d - meas_freq) / meas_freq
                err_str = f"{rel_err:.1f}"
            else:
                err_str = r"-"

            latex_lines.append(
                f"    {structure.name} & "
                f"{sigma_over_d:.4f} & "
                f"{ymax_over_d:.4f} & "
                f"{meas_freq_str} & "
                f"{meas_rare_str} & "
                f"{err_str} \\\\"
            )

        latex_lines.append(r"    \bottomrule")
        latex_lines.append(r"  \end{tabular}")
        latex_lines.append(r"\end{table}")

        output_file = Path(output_dir) / "spectral_viv_results_table.tex"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(latex_lines))

        print(f"\n✅ LaTeX table saved to: {output_file.name}")


    def run_structural_damping_sensitivity(self,
                                           delta_s_min: float = 0.01,
                                           delta_s_max: float = 0.06,
                                           step: float = 0.001,
                                           verbose=None):
        """
        Frequency-domain analysis over a range of structural damping values.
        """
        _verbose = verbose if verbose is not None else self.verbose

        try:
            from visualization.parameter_sensitivity_plotter import plot_structural_damping_sensitivity
        except ImportError:
            if verbose:
                print("⚠️  Parameter sensitivity plotting module not available")
            return None
        
        delta_s_values = np.arange(delta_s_min, delta_s_max + step, step)
        sigma_y_over_d = np.zeros_like(delta_s_values)
        ymax_over_d = np.zeros_like(delta_s_values)

        structure = self.structure

        if _verbose:
            print(f"\nRunning damping sensitivity sweep: {len(delta_s_values)} points...")

        for i, delta_s in enumerate(delta_s_values):
            s = deepcopy(structure)
            s.delta_s = float(delta_s)

            # Temporarily swap, compute, and restore
            self.structure = s
            try:
                # Pass verbose=False to suppress individual calculation prints
                fd = self._calculate_response(verbose=False)
            except Exception as e:
                if _verbose:
                    print(f"  ⚠️ Sweep step δ_s={delta_s:.4f} failed: {e}")
                sigma_y_over_d[i] = np.nan
                ymax_over_d[i] = np.nan
                continue

            # Normalize by reference diameter
            d = s.diameter
            sigma_y_over_d[i] = fd.sigma_y / d
            ymax_over_d[i] = fd.y_max / d

            # Progress indicator (every 10%)
            if _verbose and (i % max(1, len(delta_s_values) // 10) == 0):
                print(f"  Progress: {i}/{len(delta_s_values)} ({100*i/len(delta_s_values):.0f}%)")

        # Restore original structure
        self.structure = structure

        if _verbose:
            print(f"  ✅ Sweep complete: {np.sum(~np.isnan(sigma_y_over_d))}/{len(delta_s_values)} successful")

        # Plot and save
        title = f"{structure.name} - Structural Damping Sensitivity"
        output_dir = Path(self.output_dir) if self.output_dir else None
        plot_structural_damping_sensitivity(delta_s_values, sigma_y_over_d, ymax_over_d, title, output_dir)

        return {
            'delta_s': delta_s_values,
            'sigma_y_over_d': sigma_y_over_d,
            'y_max_over_d': ymax_over_d
        }
    
    def run_velocity_ratio_sensitivity(self,
                                       umin: float = 0.60,
                                       umax: float = 1.80,
                                       step: float = 0.02,
                                       verbose=None):
        """
        Frequency-domain analysis over a range of velocity ratios U/U_crit.
        """
        _verbose = verbose if verbose is not None else self.verbose

        try:
            from visualization.parameter_sensitivity_plotter import plot_velocity_ratio_sensitivity
        except ImportError:
            if _verbose:
                print("⚠️  Parameter sensitivity plotting module not available")
            return None
        
        from config.aerodynamic_parameters import AeroDampingFullCICIND
        if not isinstance(self.config.damping_modification, AeroDampingFullCICIND):
            if _verbose:
                print("⚠️  U/Ucrit sweep requires 'Full (Ka_mean.csv)' aerodynamic damping modification.")
            return None
        
        from config.reference_diameter import ReferenceDiameterCalculator
        d_ref = ReferenceDiameterCalculator.calculate(
            config=self.config.d_ref_config,
            cross_section=self.config.cross_section,
            height=self.structure.height,
            d_nominal=self.structure.diameter,
            mode_shape=self.config.mode_shape,
            mode_number=1
        )

        # Critical velocity
        u_crit = self.structure.f_n * d_ref / self.config.St

        u_ratios = np.arange(umin, umax + step, step)
        sigma_y_over_d = np.zeros_like(u_ratios)
        ymax_over_d = np.zeros_like(u_ratios)

        config0 = deepcopy(self.config)
        structure = self.structure

        if _verbose:
            print(f"\nRunning velocity ratio sweep: {len(u_ratios)} points...")

        for i, r in enumerate(u_ratios):
            cfg = deepcopy(config0)

            wp = cfg.wind_profile
            if hasattr(wp, 'u_ref'):
                wp.u_ref = float(r * u_crit)
            else:
                if _verbose:
                    print(f"  ⚠️ Wind profile type {type(wp).__name__} has no 'u_ref'; skipping step r={r:.2f}")
                sigma_y_over_d[i] = np.nan
                ymax_over_d[i] = np.nan
                continue

            # Compute frequency-domain response
            self.config = cfg
            try:
                fd = self._calculate_response(verbose=False)
            except Exception as e:
                print(f"  ⚠️ Sweep step U/Ucrit={r:.3f} failed: {e}")
                sigma_y_over_d[i] = np.nan
                ymax_over_d[i] = np.nan
            finally:
                self.config = config0

            # Normalize by nominal diameter
            d_plot = structure.diameter
            sigma_y_over_d[i] = fd.sigma_y / d_plot
            ymax_over_d[i] = fd.y_max / d_plot

            # Progress indicator (every 10%)
            if _verbose and (i % max(1, len(u_ratios) // 10) == 0):
                print(f"  Progress: {i}/{len(u_ratios)} ({100*i/len(u_ratios):.0f}%)")

        if _verbose:
            print(f"  ✅ Sweep complete: {np.sum(~np.isnan(sigma_y_over_d))}/{len(u_ratios)} successful")

        # Plot and save
        title = f"{structure.name} – Velocity Ratio Sensitivity"
        output_dir = Path(self.output_dir) if self.output_dir else None
        plot_velocity_ratio_sensitivity(u_ratios, sigma_y_over_d, ymax_over_d, title, output_dir)

        return {
            'u_over_ucrit': u_ratios,
            'sigma_y_over_d': sigma_y_over_d,
            'y_max_over_d': ymax_over_d
        }
    
    def save_results(self, results):
        """Save analysis results to files."""
        # Placeholder for saving functionality
        # Can be extended to save spectrum, time series, and response
        pass

    def run_interactive_sensitivities(self, interactive_helper):
        """
        Run optional sensitivity analyses in interactive mode.
    
        Parameters:
        -----------
        interactive_helper : object
            Object with _yesno method for user interaction
        """
        if interactive_helper._yesno("\nRun structural damping sensitivity analysis?", default=False):
            print("\nRunning δ_s sweep...")
            self.run_structural_damping_sensitivity(delta_s_min=0.01, delta_s_max=0.07, step=0.001)
    
        from config.aerodynamic_parameters import AeroDampingFullCICIND
        if isinstance(self.config.damping_modification, AeroDampingFullCICIND):
            if interactive_helper._yesno("\nRun U/Ucrit sensitivity analysis (requires Full Ka_mean.csv)?", default=False):
                print("\nRunning U/Ucrit sweep...")
                self.run_velocity_ratio_sensitivity(umin=0.60, umax=1.80, step=0.02)
        else:
            print("ℹ️  U/Ucrit sweep unavailable: requires 'Full (Ka_mean.csv)' aerodynamic damping modification.")