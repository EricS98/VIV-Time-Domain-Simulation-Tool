# common/math_utils.py
"""
Shared helper calculations.
"""

import numpy as np

def loglin(x, x1, y1, x2, y2):
    """
    Log-linear interpolation in log10-space.
    
    Interpolates linearly between (log10(x1), y1) and (log10(x2), y2).
    
    Parameters:
    -----------
    x : float or np.ndarray
        Value(s) to interpolate at
    x1, x2 : float
        Boundary x-values
    y1, y2 : float
        Boundary y-values
        
    Returns:
    --------
    float or np.ndarray
        Interpolated value(s)
    """
    lx = np.log10(x)
    lx1 = np.log10(x1)
    lx2 = np.log10(x2)
    t = (lx - lx1) / (lx2 - lx1)
    return y1 + t * (y2 - y1)