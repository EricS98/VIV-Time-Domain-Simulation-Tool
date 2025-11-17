# common/interactive_utils.py

from pathlib import Path
from typing import Optional, Union, List, Tuple, Any
from common.structures import StructureDatabase, StructureProperties
try:
    from common.file_picker import pick_file, pick_directory
    HAS_PICKER = True
except ImportError:
    HAS_PICKER = False
    pick_file = None
    pick_directory = None

class InteractiveInputHelpers:
    # Helper for input validation
    def _get_user_choice(self, prompt: str, valid_choices: list, default=None) -> int:
        """Get user choice from a list of valid options."""
        try:
            clean_prompt = prompt.strip()
            if not clean_prompt.endswith(': '):
                if clean_prompt.endswith(':'):
                    clean_prompt += ' '
                elif not clean_prompt.endswith(' '):
                    clean_prompt += ': '

            choice = input(clean_prompt).strip()
            if not choice and default is not None:
                return default
            choice = int(choice)
            return choice if choice in valid_choices else default or valid_choices[0]
        except ValueError:
            return default or valid_choices[0]
        
    def _yesno(self, prompt: str, default: bool = False) -> bool:
        """Helper for yes/no questions."""
        clean_prompt = prompt.strip()
        default_indicator = '[Y/n]' if default else '[y/N]'
    
        if not clean_prompt.endswith('?'):
            clean_prompt += '?'
    
        s = input(f"{clean_prompt} {default_indicator}: ").strip().lower()
        if s == "" and default is not None:
            return default
        return s in ("y", "yes", "j", "ja")

    
    def _get_float_input(self, prompt: str, default: float, min_val=None, max_val=None) -> float:
        """Get validated float input from user."""
        try:
            clean_prompt = prompt.strip()
            if not clean_prompt.endswith(': '):
                if clean_prompt.endswith(':'):
                    clean_prompt += ' '
                elif not clean_prompt.endswith(' '):
                    clean_prompt += ': '

            value_str = input(clean_prompt).strip()
            if not value_str:
                return default
            
            value = float(value_str)
            if min_val is not None:
                value = max(value, min_val)
            if max_val is not None:
                value = min(value, max_val)
            return value
        except ValueError:
            print(f"⚠️ Invalid input, using default: {default}")
            return default
        
    def _get_int_input(self, prompt: str, default: int, min_val=None, max_val=None) -> int:
        """Get validated integer input from user."""
        try:
            clean_prompt = prompt.strip()
            if not clean_prompt.endswith(': '):
                if clean_prompt.endswith(':'):
                    clean_prompt += ' '
                elif not clean_prompt.endswith(' '):
                    clean_prompt += ': '

            value_str = input(clean_prompt).strip()
            if not value_str:
                return default
            
            value = int(value_str)
            if min_val is not None:
                value = max(value, min_val)
            if max_val is not None:
                value = min(value, max_val)
            return value
        except ValueError:
            print(f"⚠️ Invalid input, using default: {default}")
            return default
        
    def _get_terrain_input(self) -> str:
        """Get terrain category from user."""
        print("\nTerrain Categories:")
        print("I   - Sea, lakes (z₀=0.01m)")
        print("II  - Low vegetation (z₀=0.05m)")  
        print("III - Regular vegetation/buildings (z₀=0.30m)")
    
        terrain_idx = self._get_user_choice("Select terrain category (1-3): ", [1, 2, 3], default=2)
        return ["I", "II", "III"][terrain_idx - 1]
    
    def _get_csv_file_path(self, project_root: Path, default_data_dir: str = "data") -> Optional[Path]:
        """Get CSV file path from user via picker or manual input."""
        csv_path = None

        if HAS_PICKER and self._yesno("Use file picker for CSV?", default=True):
            csv_path = pick_file(
                title="Select structure CSV",
                initialdir=(project_root / default_data_dir),
                filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
            )
        else:
            csv_input = input("Enter CSV file path: ").strip()
            if csv_input:
                csv_path = Path(csv_input)
                if not csv_path.is_absolute():
                    csv_path = (project_root / csv_path).resolve()
                    
        return csv_path if csv_path and csv_path.exists() else None
    
    def _select_structure_from_database(self, db: StructureDatabase) -> Optional[str]:
        """Interactive structure selection from database."""
        if not db.structures:
            print("❌ No structures found in database")
            return None
            
        print(f"\n📁 Loaded {len(db.structures)} structures")
        print("Available structures:")
        names = db.get_structure_names()
        
        for i, name in enumerate(names, 1):
            structure = db.get_structure(name)
            print(f"  {i:2d}. {name:<25} (H={structure.height:6.1f}m, D={structure.diameter:5.2f}m, f={structure.f_n:5.3f}Hz)")
        
        # Interactive structure selection
        while True:
            choice = input(f"\nSelect structure (1-{len(names)}, name, or 'q' to quit): ").strip()
            if choice.lower() == 'q':
                return None
            
            # Try as number first
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(names):
                    return names[idx]
            except ValueError:
                pass
            
            # Try as name
            if choice in db.structures:
                return choice
                
            print("❌ Invalid selection. Try again.")

    def _get_manual_structure_parameters(self) -> Optional[StructureProperties]:
        """Get manual structure parameters from user input."""
        try:
            print("\n🏗️  Manual Structure Creation")
            print("=" * 40)
            
            name = input("Structure name: ").strip()
            if not name:
                print("❌ Name cannot be empty")
                return None
                
            height = float(input("Height [m]: "))
            diameter = float(input("Diameter [m]: "))
            frequency = float(input("Natural frequency [Hz]: "))
            mass_eq = float(input("Equivalent mass [kg/m]: "))
            delta_s = float(input("Damping decrement [-]: "))
            
            # Create structure properties
            structure = StructureProperties(
                name=name, 
                height=height, 
                diameter=diameter, 
                f_n=frequency,
                m_eq=mass_eq, 
                delta_s=delta_s,
            )
            return structure
                
        except (ValueError, KeyboardInterrupt):
            print("\n❌ Invalid input or cancelled by user.")
            return None
        
    def handle_csv_structure_input(self, runner: Any, project_root: Path) -> bool:
        """Complete CSV structure input workflow."""
        csv_path = self._get_csv_file_path(project_root)
        if not csv_path:
            print("❌ Invalid or no CSV file selected")
            return False
            
        try:
            db = StructureDatabase(str(csv_path))
            structure_name = self._select_structure_from_database(db)
            
            if not structure_name:
                return False
                
            runner.structure = db.get_structure(structure_name)
            return runner.structure is not None
            
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            return False

    def handle_manual_structure_input(self, analyzer: Any) -> bool:
        """Complete manual structure input workflow."""
        structure = self._get_manual_structure_parameters()
        if not structure:
            return False
            
        # Set the structure directly
        analyzer.structure = structure
        return True