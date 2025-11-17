# Root project __init__.py
"""
Modular VIV Analysis Tool
========================

Time-domain simulation tool based on Vickery-Basu spectral model.

Quick start:
-----------
from applications import SingleStructureCICIND
analyzer = SingleStructureCICIND()

or use the execution scripts:
python run_single_analysis.py
"""

__version__ = "1.0.0"

# Make main components easily accessible
from codes import CICINDCalculator, EurocodeCalculator
from common import StructureProperties, StructureDatabase
from applications import SingleStructureCICIND

__all__ = [
    'CICINDCalculator',
    'EurocodeCalculator', 
    'StructureProperties',
    'StructureDatabase',
    'SingleStructureCICIND'
]