# applications/vortex_resonance_analysis.py
"""
Vortex-Resonance Method Analysis Module
=======================================

Complete workflow for vortex-resonance model VIV analysis.
"""

from pathlib import Path
from typing import Optional

from common.structures import StructureProperties
from config.analysis_config import VIVAnalysisConfig
from config.reference_diameter import ReferenceDiameterCalculator
from calculators.vortex_resonance import VortexResonanceCalculator, VortexResonanceResults


class VortexResonanceAnalysis:
    """Complete vortex-resonance VIV analysis workflow."""

    def __init__(self, structure: StructureProperties, config: VIVAnalysisConfig, output_dir: Optional[Path] = None):
        self.structure = structure
        self.config = config
        self.output_dir = output_dir

        # Initialize calculator
        self.vr_calc = VortexResonanceCalculator(St=config.St)

    def run_vortex_resonance_analysis(
            self,
            structure: StructureProperties,
            config: VIVAnalysisConfig,
            St: float = None,
            rho_air: float = None,
            nu_air: float = None,
            use_willecke_peil: Optional[bool] = None,
            manual_kw: float = None
        ) -> VortexResonanceResults: 
        """
        Run complete vortex-resonance analyis.
        Parameters:
        -----------
        structure : StructureProperties
            Structure properties
        config : VIVAnalysisConfig
            Analysis configuration
        St : float
            Strouhal number (default: 0.2)
        rho_air : float
            Air density [kg/m³] (default: 1.25)
        nu_air : float
            Kinematic viscosity [m²/s] (default: 1.5e-5)
        use_willecke_peil : bool
            Use Willecke-Peil extension for stable atmospheric conditions
        manual_kw : float, optional
            Manual Kw value (e.g., 0.95 for extreme events)
        
        Returns:
        --------
        VortexResonanceResults
            Complete analysis results
        """
        # Use config values if not overridden
        St = config.St if St is None else St
        rho_air = rho_air if rho_air is not None else (
            self.config.rho_air if self.config.rho_air else 1.25
        )
        nu_air = nu_air if nu_air is not None else (
            self.config.nu_air if self.config.nu_air else 1.5e-5
        )
        use_willecke_peil = use_willecke_peil if use_willecke_peil is not None else (
            self.config.use_willecke_peil if hasattr(self.config, 'use_willecke_peil') else False
        )
        manual_kw = manual_kw if manual_kw is not None else (
            self.config.manual_kw if hasattr(self.config, 'manual_kw') else None
        )

        print(f"\n{'='*70}")
        print(f" RUNNING VORTEX-RESONANCE ANALYSIS")
        print(f"{'='*70}\n")

        # Step 1: Calculate d_ref
        d_ref = ReferenceDiameterCalculator.calculate(
            config=self.config.d_ref_config,
            cross_section=self.config.cross_section,
            height=self.structure.height,
            d_nominal=self.structure.diameter,                
            mode_shape=self.config.mode_shape,
            mode_number=1
        )

        print(f"  → Reference Diameter $d_{{ref}}$: {d_ref:.3f} m")

        # Step 2: Create calculator with custom parameters
        calc = VortexResonanceCalculator(
            St=St,
            rho_air=rho_air,
            nu_air=nu_air,
            use_willecke_peil=use_willecke_peil,
            manual_kw=manual_kw
        )

        # Step 3: Calculate amplitude
        results = calc.calculate_amplitude(structure, config, d_ref)

        return results
        
    def print_results_summary(structures, vr_results_list):
        """Print console summary table for multiple structures."""
        print("\n" + "=" * 70)
        print("VORTEX-RESONANCE RESULTS SUMMARY")
        print("=" * 70)
        print(f"{'Structure':<25} {'y_max [mm]':>12} {'y/d_ref':>12} {'Sc':>12}")
        print("-" * 70)
    
        for structure, result_dict in zip(structures, vr_results_list):
            if result_dict and 'response' in result_dict:
                result = result_dict['response']
                print(f"{structure.name:<25} {result.y_max*1000:>12.2f} "
                      f"{result.y_max_over_d:>12.4f} {result.Scruton_number:>12.2f}")
    
    @staticmethod
    def generate_latex_table(structures, vr_results_list, output_dir=None):
        """
        Generate LaTex table for vortex-resonance results.

        Parameters:
        -----------
        structures : list of StructureProperties
            List of structure objects
        vr_results_list : list of VortexResonanceResults
            List of vortex-resonance results
        output_dir : Path, optional
            Directory to save the table file
        """
        if not vr_results_list:
            print("⚠️ No results to generate LaTex table")
            return
           
        # Start building LaTex table
        latex_lines = []
    
        # 1. Start landscape, small size, and longtable environments
        #latex_lines.append(r"\begin{landscape}")
        latex_lines.append(r"\begin{scriptsize}")
        latex_lines.append(r"  \begin{longtable}{r c c c c c c c c}")
    
        # 2. Caption and Label
        latex_lines.append(r"  \caption{Vortex-Resonance Analysis Results}%")
        latex_lines.append(r"  \label{tab:vortex_resonance_results}\\")
    
        # 3. First Page Header
        latex_lines.append(r"    \toprule")
        latex_lines.append(r"    No. & Sc$^{(1)}$ & $U_{\text{cr}}^{(2)}$ & Re$^{(3)}$ & $c_{\text{lat}}$ & $K_W$ & $\widebar{K}$ & $L_e/d$ & $y_{\text{max}}/d_{\text{top}}$ \\")
        latex_lines.append(r"    & [-] & [m/s] & [-] & [-] & [-] & [-] & [-] & [-] \\")
        latex_lines.append(r"    \midrule")
        latex_lines.append(r"    \endfirsthead")
    
        # 4. Subsequent Page Header (continuation)
        latex_lines.append(r"    ")
        latex_lines.append(r"    \toprule")
        latex_lines.append(r"    \multicolumn{9}{l}{\small\itshape Table \ref{tab:vortex_resonance_results} (continued)}\\")
        latex_lines.append(r"    \toprule")
        latex_lines.append(r"    No. & Sc$^{(1)}$ & $U_{\text{cr}}^{(2)}$ & Re$^{(3)}$ & $c_{\text{lat}}$ & $K_W$ & $\widebar{K}$ & $L_e/d$ & $y_{\text{max}}/d_{\text{top}}$ \\")
        latex_lines.append(r"    & [-] & [m/s] & [-] & [-] & [-] & [-] & [-] & [-] \\")
        latex_lines.append(r"    \midrule")
        latex_lines.append(r"    \endhead")
    
        # 5. Page Footer
        latex_lines.append(r"    ")
        latex_lines.append(r"    \bottomrule")
        latex_lines.append(r"    \endfoot")
        latex_lines.append(r"    ")

        # 6. Add data rows
        for idx, (structure, result) in enumerate(zip(structures, vr_results_list), start=1):
            if result is None:
                continue

            #meas = structure.measured_y_d
            #meas_str = f"{meas:.3f}" if meas is not None else r"\textemdash"
                
            latex_lines.append(
                f"{     {idx}} & "
                #f"    {structure.name} & "
                #f"{result.strouhal_number:.3f} & "
                f"{result.Scruton_number:.2f} &"
                f"{result.v_crit:.2f} & "
                f"{result.Re:.2e} & "
                f"{result.c_lat:.4f} & "
                f"{result.Kw:.3f} & "
                f"{result.K_xi:.3f} & "
                f"{result.Le_d:.2f} & "
                #f"{result.y_max:.2f} & "
                f"{result.y_max_over_d:.3f} \\\\ "
                #f"{meas_str} \\\\"
                )

        # 7. End longtable, smallsize, and landscape environments
        latex_lines.append(r"    ") # Add a final blank line before closing environment
        latex_lines.append(r"    \end{longtable}")
        latex_lines.append(r"\vspace{2mm}")
        latex_lines.append(
            rf"\textit{{(1) Scruton number computed using air density }} "
            rf"$\rho = {vr_results_list[0].rho_air:.2f}\,\mathrm{{kg/m^3}}$.\\"
        )
        latex_lines.append(
            rf"\textit{{(2) Critical velocity computed using Strouhal number }} "
            rf"St $= {vr_results_list[0].strouhal_number:.3f}\,\mathrm{{kg/m^3}}$.\\"
        )
        latex_lines.append(
            rf"\textit{{(3) Reynolds number computed using kinematic viscosity }} "
            rf"$\nu = {vr_results_list[0].nu_air:.2e}\,\mathrm{{m^2/s}}$."
        )
        latex_lines.append(r"  \end{scriptsize}")
        #latex_lines.append(r"\end{landscape}")

        # Save to file
        if output_dir:
            output_file = output_dir / "vortex_resonance_results_table.tex"
            with open(output_file, 'w') as f:
                f.write('\n'.join(latex_lines))
            print(f"\n✅ LaTeX table saved to: {output_file.name}")
        else:
            print("\n✅ LaTeX table generated (no output directory provided).")

    def save_results(self, results: VortexResonanceResults):
        """Save analysis results."""
        if not self.output_dir:
            return
        
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)