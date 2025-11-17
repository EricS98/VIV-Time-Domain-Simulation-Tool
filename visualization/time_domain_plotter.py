# visualization/time_domain_plotter.py
"""
Time-Domain Response Visualization Module
=========================================

Comprehensive visualization tools for time-domain VIV response analysis.

Features:
- Response time series (displacement, velocity, acceleration)
- Damping evolution plots
- Statistical analysis plots
- Comparison with frequency-domain results
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from scipy import stats
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple,  List
import warnings

from common.file_picker import pick_save_file
from calculators.statistics import calculate_pdf_statistics, calculate_normalized_data, get_gaussian_reference
from common.structures import StructureProperties

# Suppress matplotlib warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


class TimeDomainResponsePlotter:
    """Plotter for time-domain VIV response visualization."""

    def __init__(self, figsize_single=(12, 5), figsize_multi=(16, 10), dpi=300):
        """
        Initialize the time-domain plotter.

        Parameters:
        -----------
        figsize_single : tuple
            Figure size for single plots (width, height) in inches
        figsize_multi : tuple
            Figure size for multi-panel plots (width, height) in inches
        dpi : int
            Figure resolution
        """
        self.figsize_single = figsize_single
        self.figsize_multi = figsize_multi
        self.dpi = dpi

        # Color scheme
        self.colors = {
            'displacement': '#2E86AB',  # Blue
            'velocity': '#A23B72',      # Purple
            'acceleration': '#F18F01',  # Orange
            'force': '#C73E1D',         # Red
            'damping_struct': '#6A994E', # Green
            'damping_aero': '#BC4749',  # Dark red
            'damping_total': '#2F4858', # Dark blue
            'rms': '#8338EC',           # Purple
            'peak': '#FF006E'           # Pink
        }

    def _create_filename(self, base_name: str, output_dir: Optional[Path]) -> Optional[Path]:
        """Create timestamped filename."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = base_name.replace(' ', '_').replace('/', '_')

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            return output_dir / f"{safe_name}_{timestamp}.png"
        else:
            return None
        
    def plot_force_spectrum(
        self,
        spectrum_result,
        force_timeseries: np.ndarray,
        grid,
        max_time: float = 600.0,
        show_plot: bool = True,
        save_path: Optional[Path] = None
    ) -> Optional[Figure]:
        """
        Plot generalized force spectrum and first 600s of synthesized time series.
        Shows both real and imaginary parts of the time series (imaginary should be zero).
        
        Parameters:
        -----------
        spectrum_result : GeneralizedForceSpectrumResult
            Spectrum result object containing frequency and spectrum data
        force_timeseries : np.ndarray
            Force time series Q(t)
        grid : SimulationGrid
            Simulation grid with time array
        max_time : float
            Maximum time to display for time series (default: 600s)
        show_plot : bool
            If True, display the plot
        save_path : Path, optional
            If provided, save figure to this path
            
        Returns:
        --------
        Figure or None
        """
        if spectrum_result is None:
            print("⚠️  No spectrum data provided")
            return None
    
        if force_timeseries is None:
            print("⚠️  No force time series data provided")
            return None
        
        # Create figure with two subplots
        fig = plt.figure(figsize=(12, 8), dpi=600)
        gs = GridSpec(2, 1, figure=fig, hspace=0.3)

        ax1 = fig.add_subplot(gs[0, 0])

        # Extract data
        f = spectrum_result.frequency
        S_Qn = spectrum_result.spectrum
        f_n = spectrum_result.natural_frequency
        S_Qn_fn = spectrum_result.spectrum_at_fn

        # Plot spectrum
        ax1.plot(f, S_Qn, 'b-', linewidth=1.5, label='$S_{Qn}(f)$')

        # Mark natural frequency
        ax1.axvline(f_n, color='r', linestyle='--', linewidth=1.5, 
                   label=f'$f_n$ = {f_n:.4f} Hz')
        ax1.plot(f_n, S_Qn_fn, 'ro', markersize=6, 
                label=f'$S_{{Qn}}(f_n)$ = {S_Qn_fn:.2e} N²·s', zorder=3)
        
        # Labels and formatting
        ax1.set_xlabel('Frequency $f$ [Hz]', fontsize=12)
        ax1.set_ylabel('Generalized Force Spectrum $S_{Qn}(f)$ [N²·s]', fontsize=12)
        ax1.set_title('Generalized Force Spectrum', fontsize=13, fontweight='bold', pad=10)
        ax1.set_xlim(0, 2.5)
        ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax1.legend(loc='best', fontsize=10, framealpha=0.9)

        # Add statistics text box
        stats_text = (
            f"Modal RMS Force: {spectrum_result.modal_rms:.3e} N\n"
            f"Modal Variance: {spectrum_result.modal_variance:.3e} N²\n"
        )
        ax1.text(0.98, 0.97, stats_text, transform=ax1.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', 
                          alpha=0.8, edgecolor='gray', linewidth=0.5), 
                          fontsize=9, family='monospace')
        
        ax2 = fig.add_subplot(gs[1, 0])

        # Extract time series data
        t = grid.t
        Q_t = force_timeseries

        # Apply time limit
        mask = t <= max_time
        t_plot = t[mask]
        Q_t_plot = Q_t[mask]

        # The time series should be real-valued (imaginary part = 0)
        # Extract real and imaginary parts to verify
        Q_real = np.real(Q_t_plot)
        Q_imag = np.imag(Q_t_plot)

        # Plot real part
        ax2.plot(t_plot, Q_real, 'b-', linewidth=0.6, alpha=0.9, label='Real part')

        # Plot imaginary part
        ax2.plot(t_plot, Q_imag, 'r-', linewidth=0.6, alpha=0.7, label='Imaginary part')

        # Calculate statistics
        Q_rms = np.std(Q_t)
        Q_var = Q_rms**2
        Q_mean = np.mean(Q_t)

        # Labels and formatting
        ax2.set_xlabel('Time [s]', fontsize=12)
        ax2.set_ylabel('Generalized Force $Q_n(t)$ [N]', fontsize=12)
        ax2.set_title(f'Synthesized Force Time Series (First {max_time} s)', 
                    fontsize=14, fontweight='bold')
        ax2.set_xlim(0, max_time)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best', fontsize=10)

        # Add statistics text box
        stats_text_ts = (
            f"RMS: {Q_rms:.3e} N\n"
            f"Variance: {Q_var:.3e} N²\n"
            f"Mean: {Q_mean:.3e} N\n"
        )
        ax2.text(0.02, 0.97, stats_text_ts, transform=ax2.transAxes,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', 
                        alpha=0.8, edgecolor='gray', linewidth=0.5),
                fontsize=9, family='monospace')

        plt.tight_layout()

        # Save if requested
        if save_path:
            fig.savefig(save_path, dpi=600, bbox_inches='tight')
            print(f"✅ Spectrum plot saved to: {save_path}")

        # Show if requested
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_response_time_series(
        self,
        results,
        max_time: Optional[float] = None,
        show_statistics: bool = True,
        show_plot: bool = True,
        save_path: Optional[Path] = None,
    ) -> Optional[Figure]:
        """
        Plot displacement, velocity, and acceleration time series.

        Parameters:
        -----------
        results : TimeDomainResults
            Time-domain response results
        max_time : float, optional
            Maximum time to display [s] (useful for zooming into start)
        show_statistics : bool
            Whether to show statistics text box
        show_plot : bool
            If True, display the plot interactively
        save_path : Path, optional
            If provided, save figure to this path
            
        Returns:
        --------
        Figure or None
        """
        # Extrect data
        t = results.time
        y = results.displacement        # [m]
        ydot = results.velocity         # [m/s]
        ydotdot = results.acceleration  # [m/s²]
        Q = results.force               # [N]

        # Apply time limit if requested
        if max_time is not None:
            mask = t <= max_time
            t = t[mask]
            y = y[mask]
            ydot = ydot[mask]
            ydotdot = ydotdot[mask]
            Q = Q[mask]

        # Precompute statistics
        Q_rms = float(np.std(Q))
        Q_max = float(np.max(np.abs(Q)))

        # Create figure with 4 subplots (force + 4 response components)
        fig = plt.figure(figsize=self.figsize_multi, dpi=self.dpi)
        gs = GridSpec(4, 1, figure=fig, hspace=0.3)

        # 1. Generalized force
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.plot(t, Q, color=self.colors['force'], linewidth=0.8, alpha=0.8)
        ax0.axhline(Q_rms, color='r', linestyle='--', linewidth=1.0,
                    alpha=0.7, label=f'RMS = {Q_rms:.2f} N')
        ax0.axhline(-Q_rms, color='r', linestyle='--', linewidth=1.0,
                    alpha=0.7)
        ax0.set_ylabel('Force [N]', fontsize=11, fontweight='bold')
        ax0.set_title('Generalized Force and Time-Domain Response',
                    fontsize=13, fontweight='bold', pad=10)
        ax0.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        ax0.legend(loc='upper right', fontsize=9)

        # Optional small stats box for force
        stats_text_force = (
            f"RMS: {Q_rms:.2f} N\n"
            f"Max: {Q_max:.2f} N\n"
            f"Duration: {t[-1]:.1f} s"
        )
        ax0.text(0.02, 0.98, stats_text_force, transform=ax0.transAxes,
                verticalalignment='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        # 2. Displacement
        ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
        ax1.plot(t, y, color=self.colors['displacement'], linewidth=0.8, alpha=0.8)
        ax1.axhline(results.sigma_y, color='r', linestyle='--',
                    linewidth=1.2, alpha=0.7, label=f'RMS = {results.sigma_y:.2f} m')
        ax1.axhline(-results.sigma_y, color='r', linestyle='--',
                    linewidth=1.2, alpha=0.7)
        ax1.set_ylabel('Displacement [m]', fontsize=11, fontweight='bold')
        ax1.set_title('Time-Domain Response', fontsize=13, fontweight='bold', pad=10)
        ax1.set_xlim(0, results.time[-1])
        ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        ax1.legend(loc='upper right', fontsize=9)

        # Add statistics box if requested
        if show_statistics:
            stats_text = (
                f"σ_y = {results.sigma_y:.2f} m\n"
                f"y_max = {results.y_max:.2f} m\n"
                f"k_p = {results.peak_factor:.3f}"
            )
            ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                     verticalalignment='top', fontsize=9,
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
        # 3. Velocity
        ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)
        ax2.plot(t, ydot, color=self.colors['velocity'], linewidth=0.8, alpha=0.8)
        ax2.axhline(results.sigma_ydot, color='r', linestyle='--', 
                   linewidth=1.2, alpha=0.7, label=f'RMS = {results.sigma_ydot:.2f} m/s')
        ax2.axhline(-results.sigma_ydot, color='r', linestyle='--', 
                   linewidth=1.2, alpha=0.7)
        ax2.set_ylabel('Velocity [m/s]', fontsize=11, fontweight='bold')
        ax2.set_xlim(0, results.time[-1])
        ax2.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        ax2.legend(loc='upper right', fontsize=9)
        
        # 4. Acceleration
        ax3 = fig.add_subplot(gs[3, 0], sharex=ax0)
        ax3.plot(t, ydotdot, color=self.colors['acceleration'], linewidth=0.8, alpha=0.8)
        ax3.set_xlabel('Time [s]', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Acceleration [m/s²]', fontsize=11, fontweight='bold')
        ax3.set_xlim(0, results.time[-1])
        ax3.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

        ax0.set_xlim(t[0], t[-1])
        
        # Overall title
        title = 'Time-Domain VIV Response'
        if max_time:
            title += f' (First {max_time} s)'
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
        
        plt.tight_layout()

        # Save if requested
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✅ Response time series saved to: {save_path}")
        
        # Show if requested
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def plot_damping_evolution(
        self,
        results,
        max_time: Optional[float] = None,
        show_plot: bool = True,
        save_path: Optional[Path] = None,
        show_transient_detection: bool = True
    ) -> Optional[Figure]:
        """
        Plot damping ratio and RMS displacement evolution over time.
        
        Parameters:
        -----------
        results : TimeDomainResults
            Time-domain response results with damping_history
        max_time : float, optional
            Maximum time to display [s]
        show_plot : bool
            If True, display the plot interactively
        save_path : Path, optional
            If provided, save figure to this path
        show_transient_detection : bool
            If True, overlay transient detection analysis (default: True)
            
        Returns:
        --------
        Figure or None
        """
        if results.damping_history is None:
            print("⚠️  No damping history available. Run with store_damping_history=True")
            return None
        
        # Extract data
        history = results.damping_history
        t = history['time']
        rms = history['rms_history']
        zeta_s = history['zeta_struct']
        zeta_a = history['zeta_aero']
        zeta_total = history['zeta_total']

        # Apply time limit if requested
        if max_time is not None:
            mask = t <= max_time
            t = t[mask]
            rms = rms[mask]
            zeta_s = zeta_s[mask]
            zeta_a = zeta_a[mask]
            zeta_total = zeta_total[mask]

        # Steady-state / limit-cycle onset from time-domain statistics
        steady_start_index = getattr(results, "steady_start_index", 0)
        # Use either stored time or derive from index as fallback
        if hasattr(results, "steady_start_time") and results.steady_start_time > 0.0:
            steady_start_time = results.steady_start_time
        else:
            steady_start_time = t[min(steady_start_index, len(t) - 1)]

        # Number of natural periods trimmed
        trimmed_periods = steady_start_time * results.natural_frequency

        # Create figure with 2 subplots
        fig = plt.figure(figsize=self.figsize_multi, dpi=self.dpi)
        gs = GridSpec(2, 1, figure=fig, hspace=0.35)
        
        # 1. RMS displacement evolution
        ax1 = fig.add_subplot(gs[0, 0])

        # Plot instantaneous RMS
        ax1.plot(t, rms * 1000, color=self.colors['rms'], 
                 linewidth=1.5, label='Running RMS', zorder=3)

        # Shade transient region and mark limit-cycle onset (if requested)
        if show_transient_detection:
            # Transient region
            ax1.axvspan(t[0], steady_start_time, alpha=0.12, color='gray',
                        label='Transient region', zorder=1)

            # Vertical line at onset of limit-cycle / steady-state
            label_text = f'Steady-state onset\n({trimmed_periods:.0f} periods)'
            ax1.axvline(steady_start_time, color='#C73E1D', linestyle='--',
                    linewidth=1.5, alpha=0.85, label=label_text, zorder=2)
        
        # Final RMS reference line
        ax1.axhline(results.sigma_y * 1000, color='k', linestyle=':', 
                    linewidth=1.3, alpha=0.65, 
                    label=rf'$\sigma_y$ (steady) = {results.sigma_y*1000:.2f} mm',
                    zorder=2)
        
        ax1.set_ylabel(r'RMS Displacement $\sigma_y$ [mm]', fontsize=11)
        ax1.set_title('(a) RMS Displacement Evolution', fontsize=11, loc='left', pad=8)
        ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        ax1.minorticks_on()
        ax1.grid(True, which='minor', alpha=0.1, linestyle=':', linewidth=0.3)
        ax1.legend(loc='best', fontsize=8.5, framealpha=0.95, edgecolor='gray')
        
        # 2. Damping ratio evolution
        ax2 = fig.add_subplot(gs[1, 0])
        
        ax2.plot(t, zeta_s * 100, color=self.colors['damping_struct'], 
                linewidth=1.5, label='Structural $\zeta_s$', alpha=0.9)
        ax2.plot(t, -zeta_a * 100, color=self.colors['damping_aero'], 
                linewidth=1.5, label='Aerodynamic $-\zeta_a$', alpha=0.9)
        ax2.plot(t, zeta_total * 100, color=self.colors['damping_total'], 
                linewidth=2.0, label='Total $\zeta_{\mathrm{tot}}$', alpha=0.9)
        ax2.axhline(0, color='k', linestyle='-', linewidth=0.8, alpha=0.4)
        ax2.set_xlabel('Time [s]', fontsize=11)
        ax2.set_ylabel(r"Damping ratio $\zeta$ [%]", fontsize=11)
        ax2.set_title("(b) Damping ratio evolution", fontsize=11, loc="left", pad=8)
        ax2.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        ax2.minorticks_on()
        ax2.grid(True, which="minor", alpha=0.1, linestyle=":", linewidth=0.3)
        ax2.legend(loc='best', fontsize=8.5, framealpha=0.95, ncol=3, edgecolor="gray")
        
        # Add statistics box
        stats_text = (
            f"{r'$\zeta_s$'} = {zeta_s[-1]*100:.3f}% \n"
            f"{r'$\zeta_a$'} = {-zeta_a[-1]*100:.3f}% \n"
            f"{r'$\zeta_{tot}$'} = {zeta_total[-1]*100:.3f}%"
        )
               
        ax2.text(0.97, 0.97, stats_text, transform=ax2.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.6', facecolor='white', 
                          alpha=0.9, edgecolor='gray', linewidth=0.8), 
                          fontsize=8.5)
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✅ Damping evolution plot saved to: {save_path}")
        
        # Show if requested
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def plot_comparison_with_frequency_domain(
        self,
        td_results,  # TimeDomainResults
        fd_results,  # FrequencyDomainResults
        show_plot: bool = True,
        save_path: Optional[Path] = None
    ) -> Optional[Figure]:
        """
        Create comparison plot between time-domain and frequency-domain results.
        
        Parameters:
        -----------
        td_results : TimeDomainResults
            Time-domain response results
        fd_results : FrequencyDomainResults
            Frequency-domain response results
        show_plot : bool
            If True, display the plot interactively
        save_path : Path, optional
            If provided, save figure to this path
            
        Returns:
        --------
        Figure or None
        """
        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=self.dpi)
        
        # Calculate comparison metrics
        sigma_y_ratio = td_results.sigma_y / fd_results.sigma_y
        y_max_ratio = td_results.y_max / fd_results.y_max
        kp_diff = td_results.peak_factor - fd_results.peak_factor
        
        # 1. RMS Displacement Comparison
        ax1 = axes[0, 0]
        categories = ['Time\nDomain', 'Frequency\nDomain']
        values = [td_results.sigma_y * 1000, fd_results.sigma_y * 1000]
        bars = ax1.bar(categories, values, color=[self.colors['displacement'], self.colors['velocity']], 
                      alpha=0.7, edgecolor='black', linewidth=1.5)
        ax1.set_ylabel('RMS Displacement σ_y [mm]', fontsize=11, fontweight='bold')
        ax1.set_title('RMS Displacement Comparison', fontsize=12, fontweight='bold')
        ax1.grid(True, axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add ratio text
        ax1.text(0.5, 0.95, f'Ratio: {sigma_y_ratio:.3f}', 
                transform=ax1.transAxes, ha='center', va='top',
                fontsize=10, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        # 2. Peak Displacement Comparison
        ax2 = axes[0, 1]
        values = [td_results.y_max * 1000, fd_results.y_max * 1000]
        bars = ax2.bar(categories, values, color=[self.colors['displacement'], self.colors['velocity']], 
                      alpha=0.7, edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('Peak Displacement y_max [mm]', fontsize=11, fontweight='bold')
        ax2.set_title('Peak Displacement Comparison', fontsize=12, fontweight='bold')
        ax2.grid(True, axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add ratio text
        ax2.text(0.5, 0.95, f'Ratio: {y_max_ratio:.3f}', 
                transform=ax2.transAxes, ha='center', va='top',
                fontsize=10, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        # 3. Peak Factor Comparison
        ax3 = axes[1, 0]
        values = [td_results.peak_factor, fd_results.peak_factor]
        bars = ax3.bar(categories, values, color=[self.colors['displacement'], self.colors['velocity']], 
                      alpha=0.7, edgecolor='black', linewidth=1.5)
        ax3.set_ylabel('Peak Factor k_p [-]', fontsize=11, fontweight='bold')
        ax3.set_title('Peak Factor Comparison', fontsize=12, fontweight='bold')
        ax3.grid(True, axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add difference text
        ax3.text(0.5, 0.95, f'Difference: {kp_diff:+.3f}', 
                transform=ax3.transAxes, ha='center', va='top',
                fontsize=10, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        # 4. Summary Statistics Table
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Create table data
        table_data = [
            ['Parameter', 'Time Domain', 'Freq Domain', 'Ratio/Diff'],
            ['σ_y [mm]', f'{td_results.sigma_y*1000:.2f}', 
             f'{fd_results.sigma_y*1000:.2f}', f'{sigma_y_ratio:.3f}'],
            ['y_max [mm]', f'{td_results.y_max*1000:.2f}', 
             f'{fd_results.y_max*1000:.2f}', f'{y_max_ratio:.3f}'],
            ['k_p [-]', f'{td_results.peak_factor:.3f}', 
             f'{fd_results.peak_factor:.3f}', f'{kp_diff:+.3f}'],
            ['Converged', str(td_results.converged), 
             str(fd_results.converged), '-']
        ]
        
        # Create table
        table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                         colWidths=[0.3, 0.25, 0.25, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)
        
        # Style header row
        for i in range(4):
            cell = table[(0, i)]
            cell.set_facecolor('#4472C4')
            cell.set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(table_data)):
            for j in range(4):
                cell = table[(i, j)]
                if i % 2 == 0:
                    cell.set_facecolor('#E7E6E6')
                else:
                    cell.set_facecolor('#F2F2F2')
        
        ax4.set_title('Summary Comparison', fontsize=12, fontweight='bold', pad=20)
        
        # Overall title
        fig.suptitle('Time-Domain vs Frequency-Domain Comparison', 
                    fontsize=15, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✅ Comparison plot saved to: {save_path}")
        
        # Show if requested
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def plot_probability_distribution(
        self,
        results,
        trim_fraction: float = 0.2,
        show_plot: bool = True,
        save_path: Optional[Path] = None
    ) -> Optional[Figure]:
        """
        Plot probability distribution of displacement to check for non-Gaussian behavior.
    
        Parameters:
        -----------
        results : TimeDomainResults
            Time-domain response results
        trim_fraction : float
            Fraction of data to trim from beginning (default: 0.2)
        show_plot : bool
            If True, display the plot interactively
        save_path : Path, optional
            If provided, save figure to this path
    
        Returns:
        --------
        Figure or None
        """
        # Get displacement data
        y = results.displacement
        t = results.time

        # Trim transient
        if hasattr(results, 'steady_start_index') and results.steady_start_index > 0:
            n_skip = results.steady_start_index
            t_start = results.steady_start_time
        else:
            # Fallback to trim_fraction if steady_start not available
            n_skip = int(len(y) * trim_fraction)
            t_start = t[n_skip]

        y_steady = y[n_skip:]

        # Calculate statistics
        pdf_stats = calculate_pdf_statistics(y, trim_fraction=0.0)

        # Normalize data
        y_normalized = calculate_normalized_data(y_steady)

        # Create figure
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6), dpi=self.dpi)

        # Calculate histogram
        n_bins = 60 
    
        # Plot histogram
        ax1.hist(y_normalized, bins=n_bins, density=True, 
                alpha=0.7, 
                color='#4A90E2', 
                edgecolor='white', 
                linewidth=0.8,
                label='Time-Domain Data',
                zorder=5)

        # Overlay Gaussian reference
        x_gaussian = np.linspace(-4, 4, 300) 
        y_gaussian = get_gaussian_reference(x_gaussian)
        ax1.plot(x_gaussian, y_gaussian, 
                color='#E74C3C', 
                linestyle='--', 
                linewidth=2.5, 
                label='Gaussian Reference (κ = 3.0)',
                zorder=10)
    
        ax1.set_xlabel(r'Normalized Displacement $\frac{y - \mu}{\sigma}$ [–]', 
                        fontsize=13)
        ax1.set_ylabel('Probability Density [–]', fontsize=13)
        ax1.set_title('Probability Distribution Function', 
                    fontsize=14, pad=15)
    
        ax1.legend(loc='upper right', fontsize=11, framealpha=0.95,
              edgecolor='gray', fancybox=False)
    
        ax1.grid(True, alpha=0.25, linestyle='--', linewidth=0.8, color='gray')
        ax1.set_axisbelow(True)  # Grid behind plot elements
    
        ax1.set_xlim([-4, 4])
        ax1.set_ylim([0, None])  # Auto y-limit with 0 floor
    
        # Statistics box
        stats_text = (
            f"Steady-State Analysis\n"
            f"{'─'*24}\n"
            f"Data after t = {t_start:.0f} s\n"
            f"Sample size: N = {len(y_steady):,}\n"
            f"\n"
            f"Distribution Metrics\n"
            f"{'─'*24}\n"
            f"Kurtosis:  κ = {pdf_stats.kurtosis:.3f}\n"
            f"Skewness:  γ = {pdf_stats.skewness:.3f}"
        )

        # Place statistics box with better styling
        ax1.text(0.02, 0.98, stats_text, 
                transform=ax1.transAxes,
                verticalalignment='top', 
                horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.8', 
                        facecolor='white', 
                        edgecolor='gray',
                        alpha=0.95,
                        linewidth=1.2),
                fontsize=10, 
                family='monospace',
                linespacing=1.4)

        plt.tight_layout()

        # Save if requested
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight', 
                        facecolor='white', edgecolor='none')
            print(f"✅ PDF plot saved to: {save_path}")

        # Show if requested
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def create_comprehensive_report(
        self,
        td_results,  # TimeDomainResults
        fd_results=None,  # Optional FrequencyDomainResults
        structure=None,
        spectrum_result=None,
        output_dir: Optional[Path] = None,
        show_plots: bool = False,
        max_time_detail: Optional[float] = 100.0
    ) -> Dict[str, Path]:
        """
        Create comprehensive analysis report with all plots.
        
        Parameters:
        -----------
        td_results : TimeDomainResults
            Time-domain response results
        fd_results : FrequencyDomainResults, optional
            Frequency-domain results for comparison
        output_dir : Path, optional
            Directory to save all plots
        show_plots : bool
            Whether to display plots interactively
        max_time_detail : float, optional
            Maximum time for detailed plots [s]
            
        Returns:
        --------
        dict : Dictionary mapping plot names to file paths
        """
        if output_dir is None:
            print("⚠️  No output directory specified. Plots will not be saved.")
            save_plots = False
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            save_plots = True
        
        print("\n" + "="*70)
        print("CREATING COMPREHENSIVE TIME-DOMAIN ANALYSIS REPORT")
        print("="*70)
        
        saved_paths = {}

        # 0. Force spectrum and synthesized time series
        if spectrum_result is not None and td_results.force is not None and td_results.grid is not None:
            print("\n0. Creating force spectrum and synthesized time series...")
            path = self._create_filename("force_spectrum_and_timeseries", output_dir) if save_plots else None
            self.plot_force_spectrum(
                spectrum_result=spectrum_result,
                force_timeseries=td_results.force,
                grid=td_results.grid,
                max_time=600.0,
                show_plot=show_plots,
                save_path=path
            )
            if path:
                saved_paths['force_spectrum_timeseries'] = path
        else:
            print("\n0. Skipping force spectrum plot (spectrum data not available)")
        
        # 1. Response time series (full)
        print("\n1. Creating force and response time series (full duration)...")
        path = self._create_filename("td_response_full", output_dir) if save_plots else None
        self.plot_response_time_series(
            td_results,
            max_time=None,
            show_plot=show_plots,
            save_path=path
        )
        if path:
            saved_paths['response_full'] = path
        
        # 2. Response time series (detail)
        if max_time_detail:
            print(f"\n2. Creating response time series (first {max_time_detail}s)...")
            path = self._create_filename("td_response_detail", output_dir) if save_plots else None
            self.plot_response_time_series(
                td_results,
                max_time=max_time_detail,
                show_plot=show_plots,
                save_path=path
            )
            if path:
                saved_paths['response_detail'] = path
        
        # 3. Damping evolution
        if td_results.damping_history is not None:
            print("\n3. Creating damping evolution plot...")
            path = self._create_filename("td_damping_evolution.png", output_dir) if save_plots else None
            self.plot_damping_evolution(
                td_results,
                max_time=None,
                show_plot=show_plots,
                save_path=path
            )
            if path:
                saved_paths['damping_evolution'] = path
        else:
            print("\n3. Skipping damping evolution (no history available)")

        # 4. Probability Distribution Analysis
        print("\n4. Creating probability distribution analysis...")
        path = self._create_filename("td_probability_distribution.png", output_dir) if save_plots else None
        self.plot_probability_distribution(
            td_results,
            show_plot=show_plots,
            save_path=path
        )
        if path:
            saved_paths['pdf_analysis'] = path
        
        # 6. Comparison with frequency domain (if available)
        #if fd_results is not None:
        #    print("\n6. Creating comparison with frequency domain...")
        #    path = self._create_filename("td_vs_fd_comparison.png", output_dir) if save_plots else None
        #    self.plot_comparison_with_frequency_domain(
        #        td_results,
        #        fd_results,
        #        show_plot=show_plots,
        #        save_path=path
        #    )
        #    if path:
        #        saved_paths['comparison'] = path
        #else:
        #    print("\n6. Skipping frequency domain comparison (no FD results provided)")

        # 6. Comparison with frequency-domain (if available)
        
        print("\n" + "="*70)
        print("REPORT GENERATION COMPLETE")
        print("="*70)
        
        if saved_paths:
            print("\n📊 Generated plots:")
            for name, path in saved_paths.items():
                print(f"  • {name}: {path}")
        
        return saved_paths
    
# Convenience function for quick plotting
def plot_time_domain_results(
    td_results,
    fd_results=None,
    output_dir: Optional[Path] = None,
    show_plots: bool = True,
    create_full_report: bool = True
):
    """
    Convenience function to quickly plot time-domain results.
    
    Parameters:
    -----------
    td_results : TimeDomainResults
        Time-domain response results
    fd_results : FrequencyDomainResults, optional
        Frequency-domain results for comparison
    output_dir : Path, optional
        Directory to save plots
    show_plots : bool
        Whether to display plots interactively
    create_full_report : bool
        If True, create comprehensive report; else just basic plots
    """
    plotter = TimeDomainResponsePlotter()
    
    if create_full_report:
        plotter.create_comprehensive_report(
            td_results=td_results,
            fd_results=fd_results,
            output_dir=output_dir,
            show_plots=show_plots
        )
    else:
        # Just create basic plots
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Response time series
        path = output_dir / "response.png" if output_dir else None
        plotter.plot_response_time_series(td_results, show_plot=show_plots, save_path=path)
        
        # Comparison if available
        if fd_results is not None:
            path = output_dir / "comparison.png" if output_dir else None
            plotter.plot_comparison_with_frequency_domain(
                td_results, fd_results, show_plot=show_plots, save_path=path)