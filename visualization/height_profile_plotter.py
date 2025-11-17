# visualization/height_profile_plotter.py
"""
Height Profile Plotter
======================

Visualization module for height-dependent structural and wind properties.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Optional

from .height_profile_data import HeightProfileData
from common.file_picker import pick_save_file

class HeightProfilePlotter:
    """
    Plotter for height-dependent VIV analysis properties.

    Generates standardized visualization of structural geometry, 
    wind conditions, and spectral model parameters along structure height.
    """

    def __init__(self):
        """Initialize the plotter."""
        # Define consistent style
        self.color_structure = '#2C3E50'  # Dark blue-gray for structural properties
        self.color_wind = '#2E86C1'       # Blue for wind properties
        self.color_spectral = '#E67E22'   # Orange for spectral parameters
        self.color_critical = '#27AE60'   # Green for critical values
        self.color_ratio = '#C0392B'      # Red for velocity ratio
        self.linewidth = 1.5              # Thinner lines for thesis

    def plot_height_profiles(
        self,
        data: HeightProfileData,
        structure_name: str = "Structure",
        output_dir: Optional[Path] = None,
        show_plot: bool = False
    ) -> Optional[Path]:
        """
        Create a single 3x2 figure with all height-dependent properties.

        Returns path to the saved file (or None if not saved).
        """
        z = data.z

        # 3x2 layout (3 rows, 2 columns)
        fig, axes = plt.subplots(3, 2, figsize=(12, 14), sharey=True)
        fig.suptitle(
            f'Height-Dependent Properties - {structure_name}',
            fontsize=16,
            fontweight='bold'
        )

        for ax in axes.ravel():
            ax.tick_params(
                direction='in',
                top=True,
                right=True,
                labelsize=10
            )
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)

        # --- Row 1, Col 1: Diameter profile ---
        ax = axes[0, 0]
        ax.plot(data.d, z, color=self.color_structure,
                linewidth=self.linewidth)
        ax.set_xlabel('Diameter, d(z) [m]', color=self.color_structure)
        ax.set_ylabel('Height, z [m]')
        ax.set_title('Diameter Profile')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.0, data.height])
        ax.tick_params(axis='x', labelcolor=self.color_structure)

        # --- Row 1, Col 2: Mode shape ---
        ax = axes[0, 1]
        ax.plot(data.phi, z, color=self.color_structure,
                linewidth=self.linewidth)
        ax.set_xlabel('Mode Shape, φ(z) [-]', color=self.color_structure)
        ax.set_title('Mode Shape')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.0, data.height])
        ax.set_xlim([-2, 2])
        ax.axvline(x=0, color='gray', linewidth=0.8, alpha=0.5)
        ax.tick_params(axis='x', labelcolor=self.color_structure)

        # --- Row 2, Col 1: Wind velocity, critical velocity, and velocity ratio ---
        ax = axes[1, 0]
        # Wind velocities on main x-axis
        ax.plot(data.u, z, color=self.color_wind,
                linewidth=self.linewidth, label='U(z)')
        ax.plot(data.U_cr, z, color=self.color_critical,
                linewidth=self.linewidth, label='$U_{cr}(z)$')
        ax.set_xlabel('Velocity [m/s]', color=self.color_wind)
        ax.set_ylabel('Height, z [m]')
        ax.set_title('Wind Velocity, Critical Velocity and Ratio')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.0, data.height])
        ax.tick_params(axis='x', labelcolor=self.color_wind)

        # Nice x-limit for velocities
        u_max = max(np.max(data.u), np.max(data.U_cr))
        ax.set_xlim([0.0, 1.1 * u_max])

        # Velocity ratio on twin x-axis with color-coded labels
        ax_ratio = ax.twiny()
        ax_ratio.plot(data.velocity_ratio, z, color=self.color_ratio,
                     linewidth=self.linewidth, label='U/U$_{cr}$(z)')
        ax_ratio.set_xlabel('Velocity Ratio, U/U$_{cr}$ [-]', color=self.color_ratio)
        ax_ratio.tick_params(axis='x', labelcolor=self.color_ratio)
        
        # Reasonable limit for ratio
        ratio_max = max(1.0, np.max(data.velocity_ratio))
        ax_ratio.set_xlim([0.0, 1.1 * ratio_max])
        
        # Vertical line at lock-in (dashed is OK for reference lines)
        ax_ratio.axvline(x=1.0, color=self.color_ratio, linestyle='--',
                        linewidth=1.0, alpha=0.5)

        # Combined legend
        lines_main, labels_main = ax.get_legend_handles_labels()
        lines_ratio, labels_ratio = ax_ratio.get_legend_handles_labels()
        ax.legend(lines_main + lines_ratio, labels_main + labels_ratio,
                 loc='lower right', fontsize=9, framealpha=0.9)

        # --- Row 2, Col 2: RMS lift coefficient ---
        ax = axes[1, 1]
        if data.sigma_cl is not None:
            ax.plot(data.sigma_cl, z, color=self.color_spectral,
                    linewidth=self.linewidth)
            ax.set_xlabel('$\\sigma_{C_L}(z)$ [-]', color=self.color_spectral)
            ax.set_title('RMS Lift Coefficient Profile')
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0.0, data.height])
            ax.tick_params(axis='x', labelcolor=self.color_spectral)
            
            # Set reasonable x-limits
            sigma_max = np.max(data.sigma_cl)
            ax.set_xlim([0.0, 1.1 * sigma_max])
        else:
            ax.axis('off')
            ax.text(0.5, 0.5, 'No $\\sigma_{C_L}(z)$ data\nprovided',
                   ha='center', va='center', fontsize=11, style='italic',
                   color='gray', alpha=0.7, transform=ax.transAxes)

        # --- Row 3, Col 1: Turbulence intensity and bandwidth parameter ---
        ax = axes[2, 0]
        ax.plot(data.Iv, z, color=self.color_wind,
                linewidth=self.linewidth, label='$I_v(z)$')
        ax.set_xlabel('Turbulence Intensity, $I_v(z)$ [-]', color=self.color_wind)
        ax.set_ylabel('Height, z [m]')
        ax.set_title('Turbulence Intensity and Bandwidth Parameter')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.0, data.height])
        ax.set_xlim([0.0, 1.1 * np.max(data.Iv)])
        ax.tick_params(axis='x', labelcolor=self.color_wind)

        # Bandwidth parameter on twin x-axis with color-coded labels
        ax_B = ax.twiny()
        ax_B.plot(data.B, z, color=self.color_spectral,
                 linewidth=self.linewidth, label='B(z)')
        ax_B.set_xlabel('Bandwidth Parameter, B(z) [-]', color=self.color_spectral)
        ax_B.tick_params(axis='x', labelcolor=self.color_spectral)
        
        B_max = np.max(data.B)
        ax_B.set_xlim([0.0, 1.1 * B_max])

        # Combined legend
        lines_Iv, labels_Iv = ax.get_legend_handles_labels()
        lines_B, labels_B = ax_B.get_legend_handles_labels()
        ax.legend(lines_Iv + lines_B, labels_Iv + labels_B,
                 loc='lower right', fontsize=9, framealpha=0.9)

        # --- Row 3, Col 2: Aerodynamic damping parameter ---
        ax = axes[2, 1]
        if data.Ka is not None:
            ax.plot(data.Ka, z, color=self.color_spectral,
                    linewidth=self.linewidth)
            ax.set_xlabel('$K_{a,n}(z)$ [-]', color=self.color_spectral)
            ax.set_title('Aerodynamic Damping Parameter Profile')
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0.0, data.height])
            ax.tick_params(axis='x', labelcolor=self.color_spectral)
            
            # Add vertical line at Ka=0 for reference (dashed is OK)
            ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        else:
            ax.axis('off')
            ax.text(0.5, 0.5, 'No $K_{a,n}(z)$ data\nprovided',
                   ha='center', va='center', fontsize=11, style='italic',
                   color='gray', alpha=0.7, transform=ax.transAxes)

        plt.tight_layout(rect=[0, 0.0, 1, 0.97])

        filepath = self._save_or_show_plot(
            fig,
            f"height_profiles_all_{structure_name}",
            output_dir,
            show_plot
        )
        return filepath
    
    def _save_or_show_plot(
        self,
        fig: plt.Figure,
        base_filename: str,
        output_dir: Optional[Path],
        show_plot: bool
    ) -> Optional[Path]:
        """
        Helper to save or show a plot.
        
        Parameters:
        -----------
        fig : plt.Figure
            Figure to save/show
        base_filename : str
            Base filename (without extension)
        output_dir : Optional[Path]
            Output directory (if None, use file picker)
        show_plot : bool
            Whether to show plot interactively
            
        Returns:
        --------
        Optional[Path]
            Path to saved file, or None
        """
        
        # Sanitize filename
        safe_filename = base_filename.replace(' ', '_').replace('/', '_')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{safe_filename}_{timestamp}.png"
            filepath = output_dir / filename
        else:
            suggested = f"{safe_filename}_{timestamp}.png"
            filepath = pick_save_file(initialfile=suggested, defaultextension=".png")
        
        if filepath:
            fig.savefig(str(filepath), dpi=300, bbox_inches='tight')
            print(f"💾 Saved plot to: {filepath}")
            
            if not show_plot:
                plt.close(fig)
            else:
                plt.show()
            
            return filepath
        else:
            if show_plot:
                plt.show()
            else:
                plt.close(fig)
            return None