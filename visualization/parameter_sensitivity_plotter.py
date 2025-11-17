# visualization/parameter_sensitivity_plotter.py
"""
Parameter Sensitivity Visualization Module
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

def plot_structural_damping_sensitivity(
    delta_s, sigma_y_over_d, y_max_over_d,
    title_prefix: str, output_dir: Path = None
):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    fig1, ax1 = plt.subplots(figsize=(11, 6), dpi=300)
    ax1.plot(delta_s, sigma_y_over_d, marker='o', linewidth=1.5, label=r'$\sigma_y/D$')
    ax1.set_xlabel(r'Damping Decrement $\delta_s$ [-]')
    ax1.set_ylabel(r'$\sigma_y/D$ [-]')
    ax1.set_title(f'{title_prefix} – RMS Response vs. Structural Damping')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    if output_dir:
        out = Path(output_dir) / f'damping_sweep_sigma_over_D_{ts}.png'
        fig1.savefig(out, bbox_inches='tight')
        print(f'  💾 Saved: {out}')
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(11, 6), dpi=300)
    ax2.plot(delta_s, y_max_over_d, 'o-', lw=1.5, label=r'$y_{max}/D$')
    ax2.set_xlabel(r'Damping Decrement $\delta_s$ [-]')
    ax2.set_ylabel(r'$y_{max}/D$ [-]')
    ax2.set_title(f'{title_prefix} – Peak Response vs. Structural Damping')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend()
    if output_dir:
        out = Path(output_dir) / f'damping_sweep_ymax_over_D_{ts}.png'
        fig2.savefig(out, bbox_inches='tight')
        print(f'  💾 Saved: {out}')
    plt.close(fig2)


def plot_velocity_ratio_sensitivity(
    u_over_ucrit, sigma_y_over_d, y_max_over_d,
    title_prefix: str, output_dir: Path = None
):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    fig1, ax1 = plt.subplots(figsize=(11, 6), dpi=300)
    ax1.plot(u_over_ucrit, sigma_y_over_d, marker='o', linewidth=1.5, label=r'$\sigma_y/D$')
    ax1.set_xlabel(r'Velocity Ratio $U/U_{crit}$ [-]')
    ax1.set_ylabel(r'$\sigma_y/D$ [-]')
    ax1.set_title(f'{title_prefix} – RMS Response vs. $U/U_{{crit}}$')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    if output_dir:
        out = Path(output_dir) / f'u_ratio_sweep_sigma_over_D_{ts}.png'
        fig1.savefig(out, bbox_inches='tight')
        print(f'  💾 Saved: {out}')
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(11, 6), dpi=300)
    ax2.plot(u_over_ucrit, y_max_over_d, 'o-', lw=1.5, label=r'$y_{max}/D$')
    ax2.set_xlabel(r'Velocity Ratio $U/U_{crit}$ [-]')
    ax2.set_ylabel(r'$y_{max}/D$ [-]')
    ax2.set_title(f'{title_prefix} – Peak Response vs. $U/U_{{crit}}$')
    ax2.grid(True, which='both', alpha=0.3)
    ax2.legend()
    if output_dir:
        out = Path(output_dir) / f'u_ratio_sweep_ymax_over_D_{ts}.png'
        fig2.savefig(out, bbox_inches='tight')
        print(f'  💾 Saved: {out}')
    plt.close(fig2)