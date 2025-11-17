"""
common/structures.py
Structure properties and database management for VIV analysis.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional, Union
from pathlib import Path

@dataclass
class StructureProperties:
    """Data class for structure properties."""
    name: str
    height: float       # Height [m]
    diameter: float     # Diameter at the top [m]

    f_n: float          # Natural frequency [Hz]
    m_eq: float         # Equivalent mass [kg/m]
    delta_s: float      # Logarithmic damping decrement [-]

    cross_section_variation: str = 'constant'   # 'constant', 'linear', or 'tapered_base'
    diameter_base: Optional[float] = None       # Diameter at base (z=0) [m]
    taper_height: Optional[float] = None        # Height from base to taper end [m]

    # Optional parameters with defaults
    measured_y_d: Optional[float] = None    # Measured normalized response (frequent event)
    measured_y_d_rare: Optional[float] = None   # Measured normalized response (rare event)

    # Physical constants
    rho_air: float = 1.25       # Air density [kg/m^3] (uses standard if None)
    nu_air: float = 1.5e-5      # Kinematic viscosity [m^2/s] (uses standard if None)

    def __post_init__(self):
        """Validate input parameters and calculate derived properties."""
        self._validate_inputs()
        
    def _validate_inputs(self):
        """Validate required input parameters."""
        validations = [
            (self.height    > 0, "Height must be positive"),
            (self.diameter  > 0, "Diameter must be positive"),
            (self.f_n       > 0, "Natural frequency must be positive"),
            (self.m_eq      > 0, "Equivalent mass must be positive"),
            (self.delta_s   > 0, "Damping decrement must be positive"),
        ]
        # Validate geometry-specific inputs
        cs_type = self.cross_section_variation.lower()
        if cs_type == 'constant':
            pass    # Already validated
        elif cs_type == 'linear':
            if self.diameter_base is None or self.diameter_base <= 0:
                raise ValueError("Linear taper requires a positive base diameter (diameter_base)")
            if self.diameter_base < self.diameter:
                raise ValueError("Tapered base: Base diameter must be >= top diameter")
        elif cs_type == 'base_taper':
            if self.diameter_base is None or self.diameter_base <= 0:
                raise ValueError("Tapered base requires a positive base diameter (diameter_base)")
            if self.taper_height is None or not (0 < self.taper_height <= self.height):
                raise ValueError("Tapered base requires taper_height (h1) to be between 0 and structure height")
            if self.diameter_base < self.diameter:
                raise ValueError("Tapered base: Base diameter must be >= top diameter")
        elif cs_type == 'top_taper':
            if self.diameter_base is None or self.diameter_base <= 0:
                raise ValueError("Tapered base requires a positive base diameter (diameter_base)")
            if self.taper_height is None or not (0 < self.taper_height <= self.height):
                raise ValueError("Tapered base requires taper_height (h1) to be between 0 and structure height")
            if self.diameter_base < self.diameter:
                raise ValueError("Tapered top: Base diameter must be >= top diameter")
        else:
            raise ValueError(f"Unknown cross_section_variation type: {self.cross_section_variation}")

        for condition, message in validations:
            if not condition:
                raise ValueError(message)
        
    @property
    def zeta_s(self) -> float:
        """Structural damping ratio [-]."""
        return self.delta_s / (2 * np.pi)
        
    def get_summary(self) -> Dict[str, float]:
        """Get summary of key parameters."""
        summary = {
            'name': self.name,
            'height': self.height,
            'diameter': self.diameter,
            'cross_section_variation': self.cross_section_variation,
            'diameter_base': self.diameter_base if self.diameter_base is not None else self.diameter,
            'taper_height': self.taper_height if self.taper_height is not None else 0.0,
            'natural_frequency': self.f_n,
            'equivalent_mass': self.m_eq,
            'damping_decrement': self.delta_s,
            'damping_ratio': self.zeta_s,
            'rho_air': self.rho_air,
            'nu_air': self.nu_air
        }

        # Add optional measured parameters if provided
        if self.measured_y_d is not None:
            summary['measured_y_d'] = self.measured_y_d
        if self.measured_y_d_rare is not None:
            summary['measured_y_d_rare'] = self.measured_y_d_rare

        return summary

class StructureDatabase:
    """
    Database management for structure properties.
    Handles loading, saving, and managing multiple structures.
    """

    def __init__(self, csv_file_path: Optional[str] = None):
        """
        Initialize structure database.

        Parameters:
        -----------
        csv_file_path: str, optional
            Path to CSV file with structure data
        """
        self.structures = {}
        self._file_path = Path(csv_file_path) if csv_file_path else None

        if self._file_path and self._file_path.exists():
            self.load_from_csv(self._file_path)

    def load_from_csv(self, csv_file_path: Union[str, Path]) -> None:
        """
        Load structure data from CSV file.

        Expected columns: name, height, diameter, f_n, m_eq, delta_s, cross_section_variation
        Optional columns: diameter_base, taper_height, measured_y_d, measured_y_d_rare
        """
        self._file_path = Path(csv_file_path)
        try:
            # Read CSV and skip comment lines (lines starting with #)
            df = pd.read_csv(self._file_path, comment='#')
            print(f"Loaded {len(df)} structures from {self._file_path.name}")

            # Convert to StructureProperties objects
            self.structures.clear()
            for _, row in df.iterrows():
                try:
                    row_dict = row.dropna().to_dict()

                    structure = StructureProperties(**row_dict)
                    self.structures[structure.name] = structure
                except Exception as e:
                    print(f"Warning: Could not create structure from row {row.get('name', 'N/A')}: {e}")                

        except Exception as e:
            print(f"Error loading structure data from {self._file_path.name}: {e}")

    def get_structure(self, name:str) -> Optional[StructureProperties]:
        """Get structure by name."""
        return self.structures.get(name)
    
    def get_all_structures(self) -> List[StructureProperties]:
        """Get all structures as a list."""
        return list(self.structures.values())
    
    def get_structure_names(self) -> List[str]:
        """Get list of all structure names."""
        return list(self.structures.keys())
    
    def add_structure(self, structure: StructureProperties) -> None:
        """Add a structure to the database."""
        self.structures[structure.name] = structure

    def remove_structure(self, name: str) -> bool:
        """Remove a structure from the database."""
        if name in self.structures:
            del self.structures[name]
            return True
        return False

    def get_summary_statistics(self) -> pd.DataFrame:
        """Get summary statistics of the database."""
        if not self.structures:
            return pd.DataFrame()
        
        data = [s.get_summary() for s in self.structures.values()]
        df = pd.DataFrame(data)

        # Calculate statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if not numeric_cols.empty:
            return df[numeric_cols].describe()
        return pd.DataFrame()

    def save_to_csv(self, filepath: Optional[Union[str, Path]] = None) -> None:
        """Save database to CSV file."""
        save_path = Path(filepath) if filepath else self._file_path

        if not save_path:
            raise ValueError("No filepath specified and no original file path available")
        
        if not self.structures:
            print("No structures to save.")
            return
        
        # Convert structures to DataFrame
        data = [s.get_summary() for s in self.structures.values()]
        df = pd.DataFrame(data)
        df.to_csv(save_path, index=False)
        print(f"Saved {len(df)} structures to {save_path.name}")

    def print_database_summary(self):
        """Print summary of the database."""
        print(f"\nSTRUCTURE DATABASE SUMMARY")
        print("=" * 30)
        print(f"Number of structures: {len(self.structures)}")
        
        if self.structures:
            stats = self.get_summary_statistics()
            if not stats.empty:
                print(f"\nKEY PARAMETER RANGES:")
                key_params = ['height', 'diameter', 'frequency', 'equivalent_mass', 'damping_decrement']
                for param in key_params:
                    if param in stats.columns:
                        min_val = stats.loc['min', param]
                        max_val = stats.loc['max', param]
                        mean_val = stats.loc['mean', param]
                        print(f"  {param:18}: {min_val:8.3f} to {max_val:8.3f} (avg: {mean_val:8.3f})")