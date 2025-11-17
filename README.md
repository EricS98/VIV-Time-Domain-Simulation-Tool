# VIV Analysis Toolkit

A modular Python-based toolkit for analyzing Vortex-Induced Vibrations (VIV) in tall cylindrical structures such as chimneys and towers.

Developed as part of a Master's thesis at RWTH Aachen University.

## Features

- **Multiple VIV Analysis Methods:**
  - Spectral Method (Vickery & Basu)
   - Lupi's Damping Modified Spectral Method (DMSM)
   - Eurocode and CICIND approaches
  - Vortex-Resonance Model

- **Comprehensive Modeling:**
  - Height-varying wind profiles (constant, power law, terrain-based)
  - Tapered structures with variable cross-sections
  - Amplitude-dependent aerodynamic damping

- **Time-Domain Simulation:**
  - Spectral synthesis with turbulence effects
  - Newmark-β integration methods
  - Nonlinear aerodynamic damping

- **Flexible Analysis Modes:**
  - Single-structure analysis
  - Batch analysis for multiple structures
  - Interactive and command-line interfaces
  - Parameter sensitivity studies

## Installation

### Requirements

- Python 3.8 or higher
- Required packages:
```
  numpy
  scipy
  pandas
  matplotlib
```

### Setup

1. Clone or download this repository
2. Install dependencies:
```bash
   pip install -r requirements.txt
```

3. Place the `Ka_mean.csv` file in the `data/` directory (required for full aerodynamic damping analysis)

## Quick Start

### Interactive Mode

Run the analysis tool interactively:
```bash
python run_analysis.py
```

Follow the prompts to:
1. Select single or multi-structure analysis
2. Load structure data from CSV or enter manually
3. Configure analysis parameters
4. Choose analysis type (spectral, time-domain, or both)

### Command-Line Mode

Analyze a structure directly from the command line:
```bash
python run_analysis.py \
    --csv data/structures.csv \
    --structure "Chimney_1" \
    --wind-type constant \
    --u-ref 15.0 \
    --analysis-type spectrum
```

### Structure Database Format

Create a CSV file with structure properties:
```csv
name,height,diameter,f_n,m_eq,delta_s,cross_section_variation
Chimney_1,120,4.5,0.35,8000,0.04,constant
Chimney_2,80,3.2,0.52,5500,0.035,linear
```

**Required columns:**
- `name`: Structure identifier
- `height`: Total height [m]
- `diameter`: Top diameter [m]
- `f_n`: Natural frequency [Hz]
- `m_eq`: Equivalent mass per unit length [kg/m]
- `delta_s`: Logarithmic damping decrement [-]

**Optional columns:**
- `cross_section_variation`: 'constant', 'linear', or 'tapered_base'
- `diameter_base`: Base diameter [m] (for tapered structures)
- `taper_height`: Taper height [m]
- `measured_y_d`: Measured response amplitude (for validation)

## Project Structure
```
VIV-Analysis-Toolkit/
├── run_analysis.py                 # Main entry point
├── common/                         # Common utilities
│   ├── structures.py               # Structure properties
│   ├── analysis_config_helpers.py  # Helper functions for building analysis configuration
│   ├── file_picker.py
│   ├── math_utils.py
│   ├── interactive_utils.py        # User interaction helpers
│   ├── structures.py               # Structure properties and database management
│   ├── time_domain_base.py
│   ├── time_domain_damping.py
│   ├── time_integration.py
│   └── terminal_formatting.py      # Cross-platform output formatting
├── config/                         # Configuration modules
│   ├── analysis_config.py          # Master configuration
│   ├── coherence_function.py       # Coherence/Correlation of vortex shedding
│   ├── damping_model.py            # Aerodynamic damping model formulations
│   ├── wind_profile.py             # Wind profile definitions
│   ├── geometry.py                 # Cross-section geometry
│   ├── mode_shape.py               # Structural mode shapes
│   ├── reference_diameter.py       # Common definitions for the reference diameter
│   └── aerodynamic_parameters/              # Aerodynamic coefficients
├── calculators/                             # Core calculators
│   ├── base_calculator.py                   # Shared calculations
│   ├── structure_properties_calculator.py   # Calculator for structure properties
│   ├── lift_spectrum.py                     # Calculator for sectional lift coefficient and force spectra
│   ├── load_parameter.py              # Calculator for the load parameter
│   ├── generalized_force_spectrum.py  # Generalized force spectrum calculator
│   ├── ka_integrator.py               # Height-integration of aerodynamic damping
│   ├── frequency_domain_response.py   # Calculator for frequency-domain
│   ├── time_domain_response.py        # Calculator for time-doimain
│   ├── statistics.py                  # Calculator for PDF statistics
│   └── vortex_resonance.py            # Calculator for the vortex-resonance model
├── applications/                      # Analysis workflows
│   ├── spectral_viv_analysis.py       # Analysis workflow for spectral methods
│   └── vortex_resonance_analysis.py   # Analysis workflow for vortex-resonance model
├── visualization/                     # Plotting utilities
│   ├── height_profile_data.py
│   ├── height_profile_plotter.py      # Plots for height dependent properties
│   ├── multi_structure_plotter.py     # Plots for multi-structure analysis
│   ├── parameter_sensitivity.py       # Plots for parameter sensitivity analysis
│   └── time_domain_plotter.py         # Plots for time-domain analysis
├── data/                              # Data files
│   ├── Ka_mean.csv                    # Aerodynamic damping curves
│   └── structures.csv                 # Example structures
VIV-Analysis-Toolkit/
├── scripts/
│   └── run_analysis.py                # Main entry point
```

## Usage Examples

### Example 1: Single Structure Analysis
```python
from common.structures import StructureProperties
from config import VIVAnalysisConfig, ConstantWindProfile, FundamentalModeShape
from applications.spectral_viv_analysis import SpectralVIVAnalysis

# Define structure
structure = StructureProperties(
    name="Example_Chimney",
    height=100.0,
    diameter=4.0,
    f_n=0.4,
    m_eq=7000.0,
    delta_s=0.04
)

# Configure analysis
config = VIVAnalysisConfig(
    structure_name=structure.name,
    wind_profile=ConstantWindProfile(u_ref=15.0, Iv=0.1),
    mode_shape=FundamentalModeShape(exponent=2.0),
    # ... other configuration
)

# Run analysis
analyzer = SpectralVIVAnalysis(structure, config)
results = analyzer.run_complete_analysis()
```

### Example 2: Batch Analysis
```python
from common.structures import StructureDatabase

# Load structures from CSV
db = StructureDatabase("data/structures.csv")
structures = db.get_all_structures()

# Analyze each structure
for structure in structures:
    analyzer = SpectralVIVAnalysis(structure, config)
    results = analyzer.run_complete_analysis()
    print(f"{structure.name}: y_max = {results['y_max']:.3f} m")
```

## Command-Line Options
```bash
# Show all available options
python run_analysis.py --help

# Disable emoji in output (for compatibility)
python run_analysis.py --no-emoji

# Specify output directory
python run_analysis.py --output-dir ./results

# Use file picker for CSV selection
python run_analysis.py --pick-csv
```

## Terminal Compatibility

The toolkit uses emoji symbols for enhanced user experience. If your terminal doesn't support emoji:
```bash
# Disable emoji globally via environment variable
export TERM_EMOJI=0
python run_analysis.py

# Or use command-line flag
python run_analysis.py --no-emoji
```

## Validation

See the thesis document for detailed validation results.

## Theoretical Background

This toolkit implements VIV analysis methods based on:

### Key References

1. **Vickery, B.J., Basu, R.I. (1983)**: "Across-wind vibrations of structures of circular cross-section. Part I & II." *Journal of Wind Engineering and Industrial Aerodynamics*

2. **Lupi, F., Niemann, H.-J., Höffer, R. (2017)**: "A novel spectral method for cross-wind vibrations: Application to 27 full-scale chimneys." *Journal of Wind Engineering and Industrial Aerodynamics*

3. **Lupi, F., Niemann, H.-J., Höffer, R. (2018)**: "Aerodynamic damping model in vortex-induced vibrations for wind engineering applications." *Journal of Wind Engineering and Industrial Aerodynamics*

4. **CICIND Model Code (2010)**: Model Code for Steel Chimneys

5. **Eurocode EN 1991-1-4 (2010)**: Actions on structures - Wind actions

### Method Overview

- **Spectral Method**: Models VIV using spectral density of lift forces with amplitude-dependent aerodynamic damping with different formulations
- **Vortex-Resonance Model**: Simplified approach using effective correlation length

## Troubleshooting

### Common Issues

**Problem**: `FileNotFoundError: Ka_mean.csv not found`
- **Solution**: Ensure `Ka_mean.csv` is placed in the `data/` directory

**Problem**: Emoji display issues on Windows
- **Solution**: Use `--no-emoji` flag or set `TERM_EMOJI=0`

**Problem**: `ImportError: No module named 'scipy'`
- **Solution**: Install requirements: `pip install -r requirements.txt`

**Problem**: Interactive mode not working in some terminals
- **Solution**: Use command-line mode or a modern terminal (Windows Terminal, VSCode terminal)

## Author

**[Eric Simon]**  
Master's Thesis Student  
RWTH Aachen University  
[Institute of Steel Construction]

**Supervisors:**  
- [Prof. Dr.-Ing. Frank Kemper]
- [Dr.-Ing. Robert Fontecha]

**Contact:** eric.simon@rwth-aachen.de

---

**Version**: 0.1.0 (Thesis Release)  
**Last Updated**: November 2025
**Status:** Thesis Submission Version