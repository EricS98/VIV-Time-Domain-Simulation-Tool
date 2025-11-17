# common/time_domain_base.py
"""
Streamlined Time Domain Base Module
===================================

Simplified foundation module for time-domain simulations providing:
- Essential simulation grid construction with key constraints
- Nyquist frequency validation
- Optimal time step and duration calculation
- Power-of-two FFT optimization
- Time series synthesis utilities

Streamlined to focus on essential constraints only:
- Nyquist criterion
- Minimum natural periods (for structural applications)
- Power-of-2 efficiency for FFT

Includes essential VIV-specific logic:
- Bandwidth factor B(Iv) and nσ-coverage for force-spectrum support
- Physics-based f_max from shedding frequency
- Nyquist margin via oversampling factor
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
import warnings

from common.structures import StructureProperties
from config.analysis_config import VIVAnalysisConfig
from calculators.base_calculator import BaseCalculator

def _next_power_of_2(n: int) -> int:
    """Find the next power of 2 greater than or equal to n."""
    if n <= 1:
        return 1
    return 1 << (int(np.ceil(np.log2(n))))

@dataclass
class SimulationGrid:
    """
    Container for time-domain simulation parameters.
    
    Attributes:
    -----------
    dt: float
        Time step [s]
    T: float  
        Total simulation time [s]
    N: int
        Number of time samples (power of 2 for FFT efficiency)
    df: float
        Frequency resolution [Hz] = 1/T
    f_nyquist: float
        Nyquist frequency [Hz] = 1/(2*dt)
    t: np.ndarray
        Time array [s]
    f: np.ndarray
        One-sided frequency array [Hz] for PSD calculations
    """
    dt: float
    T: float
    N: int
    df: float
    f_nyquist: float
    t: np.ndarray
    f: np.ndarray
    
    def __post_init__(self):
        """Validate the simulation grid after initialization."""
        if self.dt <= 0:
            raise ValueError("Time step dt must be positive")
        if self.T <= 0:
            raise ValueError("Simulation time T must be positive")
        if self.N <= 0:
            raise ValueError("Number of samples N must be positive")
        if abs(self.T - self.N * self.dt) > 1e-10:
            raise ValueError("Inconsistent grid: T ≠ N × dt")
        if abs(self.df - 1.0/self.T) > 1e-10:
            raise ValueError("Inconsistent frequency resolution: df ≠ 1/T")
        if abs(self.f_nyquist - 1.0/(2*self.dt)) > 1e-10:
            raise ValueError("Inconsistent Nyquist frequency")

    def get_info(self) -> Dict[str, Any]:
        """Get summary information about the simulation grid."""
        return {
            'time_step': self.dt,
            'duration': self.T,
            'n_samples': self.N,
            'frequency_resolution': self.df,
            'nyquist_frequency': self.f_nyquist,
            'max_frequency': self.f[-1],
            'is_power_of_2': (self.N & (self.N - 1)) == 0,
            'sampling_rate': 1.0/self.dt
        }


class TimeGridBuilder:
    """
    Streamlined builder class for creating simulation grids with essential constraints only.
    """
    
    def __init__(self):
        """Initialize the grid builder."""
        self.warnings_enabled = True
        self.base_calc = BaseCalculator()
    
    def set_warnings(self, enabled: bool) -> None:
        """Enable or disable warning messages."""
        self.warnings_enabled = enabled

    def create_grid(
            self, 
            structure: StructureProperties, 
            config: VIVAnalysisConfig, 
            duration: Optional[float] = None, 
            dt: Optional[float] = None,
            min_natural_periods: int = 1000,
            force_power_of_2: bool = True,
            z_points: int = 200) -> SimulationGrid:
        """
        Build simulation grid for time-series synthesis.
        """
        h = structure.height
        St = config.St
        z = np.linspace(0, h, z_points)

        d_z = config.cross_section.get_diameter(z, h, structure.diameter)
        u_z = config.wind_profile.get_velocity(z, h)
        Iv_z = config.wind_profile.get_turbulence_intensity(z, h)

        f_s_z = St * u_z / d_z
        f_s_max = float(np.max(f_s_z))

        B_max = float(np.max([self.base_calc.calculate_bandwidth(iv) for iv in Iv_z]))
        n_sigma = getattr(config.time_domain_config, 'nsigma_coverage', 4)
        s_over = getattr(config.time_domain_config, 'oversampling_factor', 2)

        fmax_target = (1.0 + n_sigma * B_max) * f_s_max
        fnyq_needed = s_over * fmax_target

        dt_was_calculated = (dt is None)

        if dt is None:
            # Calculate from Nyquist
            dt = 1.0 / (2.0 * fnyq_needed)
        else:
            # Validate provided dt against Nyquist
            fnyq = 1.0 / (2.0 * dt)
            if fnyq < fnyq_needed:
                print(f"⚠️  dt={dt:.4g}s → f_Nyq={fnyq:.3g}Hz < required {fnyq_needed:.3g}Hz "
                    f"(s={s_over}, nσ={n_sigma}).")
                print(f"   Consider dt ≤ {1/(2*fnyq_needed):.4g}s.")

        # Check accuracy constraint ALWAYS (regardless of whether dt was provided)
        f_n = structure.f_n
        if f_n > 0:
            dt_max_accuracy = 1.0 / (20.0 * f_n)  # T_n/20
    
            if dt > dt_max_accuracy:
                print(f"⚠️  dt={dt:.4g}s violates accuracy constraint!")
                print(f"   Required: dt ≤ {dt_max_accuracy:.4g}s (T_n/20)")
        
                # Only auto-adjust if dt was calculated (not user-provided)
                if dt_was_calculated:  # This is True when dt was None originally
                    dt = dt_max_accuracy
                    if self.warnings_enabled:
                        print(f"ℹ️  Time step adjusted to accuracy limit: dt = {dt:.6g} s")
                else:
                    print(f"⚠️  User-provided dt exceeds accuracy constraint but will be used as specified.")
                
        T_min_cfg = float(getattr(config.time_domain_config, "duration", 600.0))
        f_n = structure.f_n
        T_min_periods = min_natural_periods / f_n if (f_n > 0 and min_natural_periods > 0) else 0.0

        if duration is None:
            duration = max(T_min_cfg, T_min_periods)
        else:
            duration = max(duration, T_min_cfg, T_min_periods)

        # Build arrays for FFT
        N_float = duration / dt
        N = int(np.ceil(N_float))

        if N % 2 == 1:
            N += 1
        if force_power_of_2:
            N = _next_power_of_2(N)
            if N % 2 == 1:  # ensure even
                N *= 2

        T_fft = N * dt
        df = 1.0 / T_fft
        f_nyquist = 1.0 / (2.0 * dt)

        t = np.arange(N) * dt
        f = np.arange(0, N // 2 + 1) * df

        # Summary
        if self.warnings_enabled:
            margin = f_nyquist / (fmax_target if fmax_target > 0 else np.inf)
            print(
                "Simulation grid:\n"
                f"  dt = {dt:.6g} s   → f_Nyq = {f_nyquist:.3g} Hz\n"
                f"  T  = {T_fft:.3f} s → df    = {df:.6g} Hz  (N = {N})\n"
                f"Coverage target: f_s,max = {f_s_max:.3g} Hz, B_max = {B_max:.3f}, nσ = {n_sigma} "
                f"⇒ f_max,target = {fmax_target:.3g} Hz;  margin f_Nyq/f_max,target = {margin:.3g}\n"
                f"Oversampling factor s = {s_over:.3g}"
            )

        return SimulationGrid(
            dt=dt,
            T=T_fft,
            N=N,
            df=df,
            f_nyquist=f_nyquist,
            t=t,
            f=f
        )

class TimeSeriesSynthesizer:
    """
    Utility class for synthesizing time series from power spectral densities.
    """
    
    @staticmethod
    def synthesize_from_psd(S_xx: np.ndarray, df: float, N: int, 
                           random_seed: Optional[int] = None) -> np.ndarray:
        """
        Synthesize real-valued time series from one-sided power spectral density.
        
        Parameters:
        -----------
        S_xx: np.ndarray
            One-sided power spectral density [units²/Hz]
        df: float
            Frequency resolution [Hz]
        N: int
            Number of time samples desired
        random_seed: int, optional
            Random seed for reproducible results
            
        Returns:
        --------
        np.ndarray: Time series with length N
        """       
        # One-sided length must be K = N//2+1
        K = N // 2 + 1
        if len(S_xx) != K:
            raise ValueError(f"Expected one-sided PSD length {K}, got {len(S_xx)}")
              
        rng = np.random.default_rng(random_seed)

        # Amplitudes
        # DC/Nyquist: N*sqrt(S*df); interior: N*sqrt(0.5*S*df)
        # Factor N accounts for different 
        A = np.empty(K, dtype=float)
        A[0] = N * np.sqrt(S_xx[0] * df) # DC component
        if N % 2 == 0:
            A[-1] = N * np.sqrt(S_xx[-1] * df) # Nyquist component
            interior = slice(1, -1)
        else:
            interior = slice(1, K)
        A[interior] = N * np.sqrt(0.5 * S_xx[interior] * df)
        
        # Random phases for interior bins
        phi = rng.uniform(0.0, 2 * np.pi, K)
        Xh = A * np.exp(1j * phi)
        Xh[0] = Xh[0].real          # DC component
        if N % 2 == 0:
            Xh[-1] = Xh[-1].real    # Nyquist component
        
        # IFFT to get time series
        x_t = np.fft.irfft(Xh, n=N).real
        
        return x_t

def validate_simulation_constraints(grid: SimulationGrid,
                                  structure_frequencies: Optional[Dict[str, float]] = None,
                                  wind_frequencies: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    Validate simulation grid against typical constraints for wind-structure interaction.
    
    Parameters:
    -----------
    grid: SimulationGrid
        Simulation grid to validate
    structure_frequencies: dict, optional
        Structural frequencies like {'f_n': 0.5, 'f_2': 1.2} [Hz]
    wind_frequencies: dict, optional
        Wind-related frequencies like {'f_s': 0.45, 'f_cutoff': 0.1} [Hz]
        
    Returns:
    --------
    dict: Validation results and recommendations
    """
    
    results = {
        'is_valid': True,
        'warnings': [],
        'recommendations': [],
        'grid_info': grid.get_info()
    }
    
    # Check Nyquist criterion for all relevant frequencies
    all_frequencies = {}
    if structure_frequencies:
        all_frequencies.update(structure_frequencies)
    if wind_frequencies:
        all_frequencies.update(wind_frequencies)
    
    for name, freq in all_frequencies.items():
        if freq >= grid.f_nyquist:
            results['warnings'].append(f"Nyquist violation: {name} = {freq} Hz ≥ f_nyquist = {grid.f_nyquist:.2f} Hz")
            results['is_valid'] = False
        elif freq > grid.f_nyquist / 5:  # Less than 5x oversampling
            results['warnings'].append(f"Low sampling ratio for {name}: {grid.f_nyquist/freq:.1f}x (recommend >5x)")
    
    # Check frequency resolution
    if structure_frequencies:
        min_struct_freq = min(structure_frequencies.values())
        if grid.df > min_struct_freq / 50:  # Should resolve structural peaks well
            results['recommendations'].append(f"Consider finer frequency resolution: df = {grid.df:.4f} Hz, "
                                            f"recommend < {min_struct_freq/50:.4f} Hz for structural frequencies")
    
    # Check simulation duration
    if structure_frequencies:
        min_struct_freq = min(structure_frequencies.values())
        min_periods = grid.T * min_struct_freq
        if min_periods < 100:
            results['recommendations'].append(f"Short simulation for structural dynamics: {min_periods:.0f} periods "
                                            f"at {min_struct_freq:.3f} Hz (recommend >1000)")
    
    return results