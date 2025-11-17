# common/project_imports.py

import sys
from pathlib import Path

def setup_project_path():
    """Add project root to path if not already present."""
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root

# Call once when imported and export the project_root
project_root = setup_project_path()