# calculators/statistics.py
"""
Statistical Analysis for Time-Domain VIV Response
=================================================

Provides statistical analysis tools including:
- PDF statistics for non-Gaussian response detection
- Transient detection for steady-state identification
- Rolling RMS calculations

Functions:
----------
- calculate_pdf_statistics: Compute distribution statistics
- calculate_rolling_rms: Moving RMS for stability analysis
"""

import numpy as np
from scipy import stats
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import warnings

@dataclass
class PDFStatistics:
    """Statistics for probability distribution analysis."""

    # Basic statistics
    mean: float
    std: float

    # Non-Gaussian indicators
    kurtosis: float
    skewness: float

    def __repr__(self):
        return (f"PDFStatistics(mean={self.mean:.4f}, std={self.std:.4f}, "
                f"kurtosis={self.kurtosis:.3f}, type={self.distribution_type})")


def calculate_pdf_statistics(data: np.ndarray, trim_fraction: float = 0.0) -> PDFStatistics:
    """
    Calculate probability distribution statistics.
    
    Parameters:
    -----------
    data : np.ndarray
        Time series data (displacement, velocity, etc.)
    trim_fraction : float
        Fraction of data to trim from beginning (default: 0.0)
        
    Returns:
    --------
    PDFStatistics
        Statistical characterization of the distribution
    """
    # Trim transient if requested
    if trim_fraction > 0:
        n_skip = int(len(data) * trim_fraction)
        data_used = data[n_skip:]
    else:
        data_used = data

    # Basic statistics
    mean = np.mean(data_used)
    std = np.std(data_used)

    # Non-Gaussian statistics
    # Kurtosis: Pearson definition (fisher=False) gives 3 for Gaussian
    kurtosis_value = stats.kurtosis(data_used, fisher=False)

    # Skewness: 0 for symmetric distributions
    skewness_value = stats.skew(data_used)

    return PDFStatistics(
        mean=mean,
        std=std,
        kurtosis=kurtosis_value,
        skewness=skewness_value,
    )

def calculate_normalized_data(data: np.ndarray) -> np.ndarray:
    """
    Normalize data to zero mean and unit variance.

    Parameters:
    -----------
    data : np.ndarray
        Raw data

    Returns:
    --------
    np.ndarray
        Normalized data (y - mean) / std
    """
    mean = np.mean(data)
    std = np.std(data)

    if std < 1e-10:
        np.zeros_like(data)

    return (data - mean) / std

def get_gaussian_reference(x: np.ndarray) -> np.ndarray:
    """
    Get Gaussian PDF for comparison.
    
    Parameters:
    -----------
    x : np.ndarray
        x-values (normalized)
        
    Returns:
    --------
    np.ndarray
        Gaussian PDF values
    """
    return stats.norm.pdf(x, loc=0, scale=1)