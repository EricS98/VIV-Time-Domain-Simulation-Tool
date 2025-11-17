# config/aerodynamic_parameters/aerodynamic_damping_reduction.py
"""
Aerodynamic Damping Reduction Methods
=====================================

Defines how turbulence and velocity ratio reduces the aerodynamic damping parameter Ka.

The full CICIND turbulence reduction method (AeroDampingFullCICIND) should only be used
with Vickery-Basu or Eurocode damping formulations.

For Lupi's DMSM (Damping-Modified Spectral Method), use only the simplified turbulence
reduction (AeroDampingSimplified) because:

1. Lupi's envelope curve Ka(σy/d) already captures the 3D behavior Ka(σy/d, V/Vcr)
2. The peak of Ka shifts with amplitude in Lupi's model (Lupi et al. 2021)
3. A fixed V/Vcr reduction factor would misrepresent the underlying physics
4. Lupi's experiments were conducted at Iv ≈ 3% (smooth flow)

For Lupi damping with turbulence, apply:
    Ka(σy/d) = a·exp(-b·σy/d)/(σy/d)^c · max(1 - 3·Iv, 0.25)

where Iv can be height-integrated for varying wind profiles.
"""

from abc import ABC, abstractmethod
from typing import Union, Optional
import numpy as np
from pathlib import Path

class AeroDampingModification(ABC):
    """
    Abstract base class for aerodynamic damping reduction methods.
    
    Converts Ka_max to Ka_eff accounting for turbulence.

    Note: For Lupi damping, Ka_max = 1.0 (normalized), so this returns
    the turbulence reduction factor directly.
    """
    
    @abstractmethod
    def apply(self, Ka_max: float, Iv: float, u_ucr: float = 1.0) -> float:
        """
        Apply reduction to Ka_max.
        
        Parameters:
        -----------
        Ka_max : float
            Maximum aerodynamic damping parameter (from Re curves)
            For Lupi: Ka_max = 1.0 (normalized)
        Iv : float
            Turbulence intensity [-]
        u_ucr : float
            Velocity ratio U/U_crit (default: 1.0)
            
        Returns:
        --------
        float
            Effective Ka or turbulence reduction factor
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get name of turbulence reduction method."""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
    
class AeroDampingSimplified(AeroDampingModification):
    """
    Simplified modification of the aerodynamic damping parameter 
    based on CICIND Model Code formulation.
    
    Valid only at U/U_crit = 1.0 (resonance condition).
    Otherwise can overestimate results significantly.
    
    Formula: Kv(Iv) = max(1 - 3*Iv(z), 0.25)
             Ka_eff = Ka_max * Kv(Iv)

    COMPATIBLE WITH:
    - Vickery-Basu damping
    - Eurocode damping  
    - Lupi DMSM damping

    For Lupi, this applies the turbulence reduction factor to the envelope curve:
        Ka(σy/d) = a·exp(-b·σy/d)/(σy/d)^c · max(1 - 3·Iv, 0.25)
    """
    
    def apply(self, Ka_max: float, Iv: float, u_ucr: float = 1.0) -> float:
        """Apply simplified method."""
        if Iv == 0:
            return Ka_max
        
        # Turbulence reduction factor
        Kv = max(1.0 - 3.0 * Iv, 0.25)
        
        return Ka_max * Kv
    
    def get_name(self) -> str:
        return 'Simplified (CICIND Model Code)'
    
    def __repr__(self) -> str:
        return "AeroDampingSimplified()"
    
class AeroDampingFullCICIND(AeroDampingModification):
    """
    Full turbulence reduction using Ka curves (CICIND Commentaries).
    
    Valid for any U/U_crit ratio.
    Uses 2D interpolation: Ka_eff = Ka_max * K_a0(U/Ucr(z), Iv(z))
    
    COMPATIBLE WITH:
    - Vickery-Basu damping
    - Eurocode damping

    Requires Ka_mean.csv data file.
    """

    @staticmethod
    def _find_ka_data_file(ka_data_file: str) -> Path:
        """
        Find Ka data file using multiple search strategies.
        
        Search order:
        1. Absolute path (if provided)
        2. Relative to current working directory
        3. Relative to project root (config/aerodynamic_parameters/../../data/)
        4. Relative to module directory
        
        Parameters:
        -----------
        ka_data_file : str
            Filename or path to Ka data file
            
        Returns:
        --------
        Path
            Resolved path to Ka data file
            
        Raises:
        -------
        FileNotFoundError
            If file cannot be found in any search location
        """
        ka_path = Path(ka_data_file)
        
        # Strategy 1: Absolute path or exists in current directory
        if ka_path.is_absolute() and ka_path.exists():
            return ka_path
        if ka_path.exists():
            return ka_path.resolve()
        
        # Strategy 2: Relative to project root
        # Assume this file is in: project_root/config/aerodynamic_parameters/
        module_dir = Path(__file__).parent  # config/aerodynamic_parameters/
        project_root = module_dir.parent.parent  # Go up two levels
        project_data_path = project_root / "data" / ka_path.name
        
        if project_data_path.exists():
            return project_data_path
        
        # Strategy 3: Relative to module directory
        module_data_path = module_dir / ka_path.name
        if module_data_path.exists():
            return module_data_path
        
        # Not found anywhere
        raise FileNotFoundError(
            f"Ka data file '{ka_data_file}' not found. Searched:\n"
            f"  - Current directory: {Path.cwd() / ka_path}\n"
            f"  - Project data directory: {project_data_path}\n"
            f"  - Module directory: {module_data_path}\n"
            f"Please ensure the file exists in one of these locations."
        )
    
    def __init__(self, ka_data_file: str = "Ka_mean.csv"):
        """
        Initialize full turbulence reduction.
        
        Parameters:
        -----------
        ka_data_file : str
            Path to Ka data CSV file
        """
        self.ka_interpolator = None
        self.u_ucr_values = None
        self.iv_values = None
        self.loaded = False
        self.ka_data_path = None

        try:
            resolved_path = self._find_ka_data_file(ka_data_file)
            self._load_ka_data(resolved_path)
            self.ka_data_path = resolved_path
            self.loaded = True
        except FileNotFoundError as e:
            print(f"⚠️  {e}")
            print("   Full turbulence reduction not available")
        except Exception as e:
            print(f"⚠️  Failed to load Ka curves: {e}")
            print("   Full turbulence reduction not available")
        
    def _load_ka_data(self, ka_data_file: str):
        """Load Ka(U/Ucr, Iv) data from CSV."""
        import pandas as pd
        from scipy.interpolate import RegularGridInterpolator
        
        # Read CSV
        df = pd.read_csv(ka_data_file)
        df.columns = [col.strip() for col in df.columns]
        
        # Extract U/Ucr values
        self.u_ucr_values = df['Vel'].values
        
        # Extract Iv values from column names
        iv_cols = [col for col in df.columns if col != 'Vel'  and not col.startswith('Unnamed')]
        self.iv_values = np.array([float(col) for col in iv_cols])
        
        # Create Ka matrix (normalized values K_a0 = Ka/Ka_max)
        ka_matrix = np.zeros((len(self.u_ucr_values), len(self.iv_values)))
        for i, iv_col in enumerate(iv_cols):
            ka_matrix[:, i] = df[iv_col].values
        
        # Create 2D interpolator
        self.ka_interpolator = RegularGridInterpolator(
            (self.u_ucr_values, self.iv_values),
            ka_matrix,
            method='linear',
            bounds_error=False,
            fill_value=None  # Allow extrapolation
        )

        print(f"✅ Loaded Ka curves for turbulence reduction:")
        print(f"   U/Ucr range: [{self.u_ucr_values.min():.3f}, {self.u_ucr_values.max():.3f}]")
        print(f"   Iv values: {self.iv_values}")
    
    def apply(self, Ka_max: float, Iv: float, u_ucr: float = 1.0) -> float:
        """Apply full turbulence reduction using interpolation."""
        if not self.loaded or self.ka_interpolator is None:
            raise RuntimeError(
                "Ka data not loaded. Cannot use full turbulence reduction."
            )
        
        try:
            # Interpolate K_a0 (normalized Ka = Ka/Ka_max)
            K_a0 = self.ka_interpolator([u_ucr, Iv])[0]
            
            # Check if extrapolating
            u_in_bounds = self.u_ucr_values.min() <= u_ucr <= self.u_ucr_values.max()
            iv_in_bounds = self.iv_values.min() <= Iv <= self.iv_values.max()
            
            if not (u_in_bounds and iv_in_bounds):
                print(f"ℹ️  Extrapolating Ka: U/Ucr={u_ucr:.3f}, Iv={Iv:.3f}")
            
            return Ka_max * K_a0
            
        except Exception as e:
            raise RuntimeError(f"Failed to interpolate Ka: {e}")
    
    def get_name(self) -> str:
        return 'Full (CICIND Commentaries with Ka curves)'
    
    def __repr__(self) -> str:
        status = "loaded" if self.loaded else "not loaded"
        return f"AeroDampingFullCICIND({status})"
    
class NoAeroDampingModification(AeroDampingModification):
    """
    No turbulence reduction (Ka_eff = Ka_max).

    COMPATIBLE WITH:
    - Vickery-Basu damping
    - Eurocode damping
    - Lupi DMSM damping
    """
    
    def apply(self, Ka_max: float, Iv: float, u_ucr: float = 1.0) -> float:
        """Return Ka_max unchanged."""
        return Ka_max
    
    def get_name(self) -> str:
        return 'None (Ka_eff = Ka_max)'
    
    def __repr__(self) -> str:
        return "NoAeroDampingModification()"
    
# Factory functions
def create_simplified_modification() -> AeroDampingSimplified:
    """Create simplified turbulence reduction method."""
    return AeroDampingSimplified()

def create_full_modification(
        ka_data_file: str = "Ka_mean.csv"
        ) -> AeroDampingFullCICIND:
    """Create full turbulence reduction method with Ka curves."""
    return AeroDampingFullCICIND(ka_data_file)

def create_no_modification() -> NoAeroDampingModification:
    """Create no turbulence reduction (for smooth flow)."""
    return NoAeroDampingModification()