# visualization/multi_structure_plotter.py

"""
Multi-Structure Visualization Module
====================================

Simple plotting utilities for comparing results from multiple structures.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime


class MultiStructurePlotter:
    """Simple plotter for multi-structure comparison."""
    
    def __init__(self, output_dir, analysis_type='spectral', base_fontsize=20):
        """
        Initialize plotter.
        
        Parameters:
        -----------
        output_dir : str or Path
            Directory to save plots
        analysis_type : str
            Type of analysis: 'spectral' or 'vortex_resonance'
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_type = analysis_type
        self.base_fontsize = base_fontsize

        # GLOBAL FONT SETTINGS FOR THESIS QUALITY
        plt.rcParams.update({
            "font.size": base_fontsize,          # general font size
            "axes.titlesize": base_fontsize + 2,
            "axes.labelsize": base_fontsize,
            "xtick.labelsize": base_fontsize - 2,
            "ytick.labelsize": base_fontsize - 2,
            "legend.fontsize": base_fontsize - 2,
        })

    def _setup_prediction_axes(self, ax, limit=0.6):
        """
        Setup axes with fixed limits, 1:1 line, and ±20% lines.
        
        Parameters:
        -----------
        ax : matplotlib axes
            The axes object to configure
        limit : float
            Maximum value for both axes
        """
        # Set equal limits for both axes
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
              
        # Add 1:1 line
        ax.plot([0, limit], [0, limit], 'k-', linewidth=1.5, 
                label='1:1 line', zorder=1)
        
        # Add ±20% lines
        ax.plot([0, limit/1.2], [0, limit], 'k--', linewidth=1, 
                alpha=0.6, zorder=1)
        ax.plot([0, limit], [0, limit * 0.8], 'k--', linewidth=1, 
                alpha=0.6, label='±20%', zorder=1)
        
        # Labels and grid
        ax.set_xlabel('Measured y/d [-]')
        ax.set_ylabel('Predicted y/d [-]')
        ax.grid(True, alpha=0.3)
    
    def plot_comparison(self, structures, results_list, domain_type='frequency', analysis_type=None):
        """
        Plot predicted vs measured response for all structures.
        
        Parameters:
        -----------
        structures : list of StructureProperties
            List of structure objects
        results_list : list of dict
            List of results dictionaries from frequency-domain analysis
        domain_type : str
            'frequency' or 'time' - determines where to look for results
        analysis_type : str, optional
            'spectral' or 'vortex_resonance' - overrides self.analysis_type if provided
        """
        if not results_list or not structures:
            print("⚠️  No results to plot")
            return
        
        analysis_type = analysis_type or self.analysis_type
        
        # Extract data
        predicted = []
        measured_freq = []
        measured_rare = []
        structure_indices = []
        
        for struct, result in zip(structures, results_list):
            if result is None:
                continue
            
            # Get predicted response based on domain type
            pred_y = 0.0
            if domain_type == 'frequency' and 'response' in result and result['response'] is not None:
                response_obj = result['response']
                if hasattr(response_obj, 'y_max'):
                    pred_y = response_obj.y_max / struct.diameter
            elif domain_type == 'time' and 'time_domain' in result and result['time_domain'] is not None:
                td_object = result['time_domain']
                if hasattr(td_object, 'y_max'):
                    pred_y = td_object.y_max / struct.diameter
                
            if pred_y <= 0:
                continue
            
            # Get measured responses
            meas_y_freq = getattr(struct, 'measured_y_d', None)
            meas_y_rare = getattr(struct, 'measured_y_d_rare', None)
            structure_idx = structures.index(struct) + 1
            
            predicted.append(pred_y)
            measured_freq.append(meas_y_freq)
            measured_rare.append(meas_y_rare)
            structure_indices.append(structure_idx)
        
        if not predicted:
            print("⚠️  No valid data for plotting")
            return
        
        # Create plot
        fig, ax = plt.subplots(figsize=(18, 9))
        
        # Setup axes with fixed limits
        self._setup_prediction_axes(ax, limit=0.6)
        
        # Plot frequent measurements
        freq_valid = [(m, p, n) for m, p, n in zip(measured_freq, predicted, structure_indices) if m is not None]
        if freq_valid:
            freq_meas, freq_pred, freq_names = zip(*freq_valid)
            marker = 'o' if domain_type == 'frequency' else 's'
            ax.scatter(freq_meas, freq_pred, s=120, alpha=0.6, 
                      edgecolors='black', marker=marker, label='Frequent Event', zorder=3)
            
            for m, p, n in zip(freq_meas, freq_pred, freq_names):
                ax.annotate(str(n), (m, p), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=self.base_fontsize - 4, alpha=0.7)
        
        # Plot rare measurements
        rare_valid = [(m, p, n) for m, p, n in zip(measured_rare, predicted, structure_indices) if m is not None]
        if rare_valid:
            rare_meas, rare_pred, rare_names = zip(*rare_valid)
            marker_rare = 'D' if domain_type == 'frequency' else '^'
            ax.scatter(rare_meas, rare_pred, s=120, alpha=0.6, 
                      edgecolors='red', facecolors='red', marker=marker_rare, 
                      label='Rare Event', zorder=3)
            
            for m, p, n in zip(rare_meas, rare_pred, rare_names):
                ax.annotate(str(n), (m, p), 
                           xytext=(5, -12), textcoords='offset points',
                           fontsize=self.base_fontsize-4, alpha=0.7, color='red')
        
        # Title
        if analysis_type == 'vortex_resonance':
            title = 'Vortex-Resonance Method: Predicted vs Measured Response'
        else:
            domain_label = 'Frequency-Domain' if domain_type == 'frequency' else 'Time-Domain'
            title = f'Spectral Method - {domain_label}: Predicted vs Measured Response'
        ax.set_title(title)
        ax.legend()
        
        # Count and annotate outliers
        all_values = [(m, p) for m, p in zip(measured_freq + measured_rare, predicted * 2) 
                      if m is not None]
        n_outliers = sum(1 for m, p in all_values if m > 0.6 or p > 0.6)
        if n_outliers > 0:
            ax.text(0.95, 0.05, f'{n_outliers} point(s) outside range', 
                    transform=ax.transAxes, ha='right', fontsize=self.base_fontsize-4, 
                    style='italic', alpha=0.7)
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if analysis_type == 'vortex_resonance':
            filename = f'vortex_resonance_method_comparison_{timestamp}.png'
        else:
            filename = f'{domain_type}_domain_comparison_{timestamp}.png'
        save_path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  💾 Saved: {save_path}")
    
    def plot_combined_comparison(self, structures, freq_results, time_results):
        """
        Plot frequency and time-domain side by side.
        
        Parameters:
        -----------
        structures : list of StructureProperties
            List of structure objects
        freq_results : list of dict
            Frequency-domain results
        time_results : list of dict
            Time-domain results
        """
        if not freq_results and not time_results:
            print("⚠️  No data to plot in combined comparison")
            return
    
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14))
        
        # Plot both domains
        if freq_results:
            self._plot_on_axis(
                ax1, structures, 
                results_list=freq_results, 
                domain_type='frequency', 
                title='Spectral Method (Frequency-Domain): Predicted vs Measured Response', 
                marker='o'
            )

        if time_results:
            self._plot_on_axis(
                ax2, structures, 
                results_list=time_results, 
                domain_type='time', 
                title='Spectral Method (Time-Domain): Predicted vs Measured Response', 
                marker='s'
            )
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = self.output_dir / f'combined_comparison_{timestamp}.png'
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  💾 Saved: {save_path}")
    
    def _plot_on_axis(self, ax, structures, results_list, domain_type, title, marker):
        """Helper to plot on a specific axis."""
        predicted = []
        measured_freq = []
        measured_rare = []
        structure_indices = []
        
        for struct, result in zip(structures, results_list):
            if result is None:
                continue
            
            # Get predicted response
            pred_y = 0.0
            if domain_type == 'frequency' and 'response' in result and result['response'] is not None:
                response_obj = result['response']
                if hasattr(response_obj, 'y_max'):
                    pred_y = response_obj.y_max / struct.diameter
            elif domain_type == 'time' and 'time_domain' in result and result['time_domain'] is not None:
                td_object = result['time_domain']
                if hasattr(td_object, 'y_max'):
                    pred_y = td_object.y_max / struct.diameter

            if pred_y <= 0:
                continue
            
            meas_y_freq = getattr(struct, 'measured_y_d', None)
            meas_y_rare = getattr(struct, 'measured_y_d_rare', None)
            structure_idx = structures.index(struct) + 1
            
            predicted.append(pred_y)
            measured_freq.append(meas_y_freq)
            measured_rare.append(meas_y_rare)
            structure_indices.append(structure_idx)
        
        if not predicted:
            ax.text(0.5, 0.5, 'No data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_axis_off()
            return
        
        # Setup axes with fixed limits
        self._setup_prediction_axes(ax, limit=0.6)
        
        # Plot frequent measurements
        freq_valid = [(m, p, n) for m, p, n in zip(measured_freq, predicted, structure_indices) if m is not None]
        if freq_valid:
            freq_meas, freq_pred, freq_names = zip(*freq_valid)
            ax.scatter(freq_meas, freq_pred, s=120, alpha=0.6, 
                        marker=marker, edgecolors='black', 
                        label='Frequent Event', zorder=3)
            
            for m, p, n in freq_valid:
                ax.annotate(str(n), (m, p), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=self.base_fontsize-4, alpha=0.7)
        
        # Plot rare measurements
        rare_valid = [(m, p, n) for m, p, n in zip(measured_rare, predicted, structure_indices) if m is not None]
        if rare_valid:
            rare_meas, rare_pred, rare_names = zip(*rare_valid)
            marker_rare = 'D' if marker == 'o' else '^'
            ax.scatter(rare_meas, rare_pred, s=120, alpha=0.6, 
                        edgecolors='red', facecolors='red', 
                        marker=marker_rare, label='Rare Event', zorder=3)
            
            for m, p, n in rare_valid:
                ax.annotate(str(n), (m, p), 
                            xytext=(5, -12), textcoords='offset points',
                            fontsize=self.base_fontsize-4, alpha=0.7, color='red')
                
        ax.set_title(title)
        ax.legend()
        
        # Count and annotate outliers
        all_values = [(m, p) for m, p in zip(measured_freq + measured_rare, predicted * 2) 
                      if m is not None]
        n_outliers = sum(1 for m, p in all_values if m > 0.6 or p > 0.6)
        if n_outliers > 0:
            ax.text(0.95, 0.05, f'{n_outliers} point(s) outside range', 
                    transform=ax.transAxes, ha='right', fontsize=self.base_fontsize-4, 
                    style='italic', alpha=0.7)
            

    def plot_peak_factor_comparison(self, structures, freq_results, time_results):
        """
        Plot peak factor vs reduced damping parameter Sc/(4πKa,n).

        Shows theoretical CICIND peak factor curve and time-domain simulation results
        as scatter points to visualize deviations.

        Parameters:
        -----------
        structures : list of StructureProperties
            List of structure objects
        freq_results : list of dict
            Frequency-domain results
        time_results : list of dict
            Time-domain results
        """
        if not freq_results or not time_results:
            print("⚠️  Need both frequency and time-domain results for peak factor plot")
            return
        
        # Extract data for structures with valid results
        reduced_damping = []
        peak_factors_td = []
        structure_indices = []

        for idx, (struct, freq_res, time_res) in enumerate(zip(structures, freq_results, time_results)):
            if freq_res is None or time_res is None:
                continue
        
            # Get frequency-domain response object
            fd_response = freq_res.get('response')
            if fd_response is None or not hasattr(fd_response, 'Ka_r_n'):
                continue
        
            # Get time-domain response object
            td_response = time_res.get('time_domain')
            if td_response is None or not hasattr(td_response, 'peak_factor'):
                continue
        
            # Calculate Sc / (4π Ka,n)
            Sc = fd_response.Scruton_number
            Ka_n = fd_response.Ka_r_n  # Ka at zero amplitude
        
            if Ka_n <= 0:
                continue
        
            sc_over_4pi_ka = Sc / (4 * np.pi * Ka_n)
        
            # Get time-domain peak factor
            g_td = td_response.peak_factor
        
            reduced_damping.append(sc_over_4pi_ka)
            peak_factors_td.append(g_td)
            structure_indices.append(idx + 1)
    
        if not reduced_damping:
            print("⚠️  No valid data for peak factor comparison plot")
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 8))
    
        # Generate theoretical CICIND peak factor curve
        x_theory = np.linspace(0, 3, 300)
        g_theory = np.sqrt(2) * (1 + 1.2 * np.arctan(0.75 * x_theory**4))
    
        # Plot theoretical curve
        ax.plot(x_theory, g_theory, 'b-', linewidth=2, label='CICIND formula (frequency-domain)', zorder=2)
    
        # Plot time-domain simulation results as scatter
        ax.scatter(reduced_damping, peak_factors_td, s=150, alpha=0.7, 
                    edgecolors='darkred', facecolors='red', marker='o',
                    label='Time-domain simulations', zorder=3)
    
        # Annotate points with structure indices
        for x, y, idx in zip(reduced_damping, peak_factors_td, structure_indices):
            ax.annotate(str(idx), (x, y), 
                        xytext=(6, 6), textcoords='offset points',
                        fontsize=self.base_fontsize-4, alpha=0.8)
    
        # Formatting
        ax.set_xlabel('$S_c / (4\\pi K_{a,n})$ [-]')
        ax.set_ylabel('Peak Factor $k_p$ [-]')
        ax.set_title('Peak Factor Comparison: Frequency vs Time-Domain', pad=20)
    
        ax.set_xlim(0, 3)
        ax.set_ylim(1.3, 4.2)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.legend(loc='best', framealpha=0.95)
    
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = self.output_dir / f'peak_factor_comparison_{timestamp}.png'
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
        print(f"  💾 Saved: {save_path}")