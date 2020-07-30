"""generate modulation vectors"""
from scipy.interpolate import interp1d
from numpy import ndarray
import numpy as np


def create_modulation(anchors: ndarray, samples: int, kind: str = "nearest") -> ndarray:
    """Use the modulation anchors to interpolate a modulation vector
    
    
    args
    ----
    anchors: ndarray
        an array of modulation anchors, assumed to be equidistantly spread across the modulation signal. The first and last anchor correspond to the first and last value of the final modulation signal. 
    samples:int
        the length of the final modulation signal in samples
    kind:str
        the kind of interpolation, inherited from :func:`scipy.interpolate.interp1d`. Therefore: \"Specifies the kind of interpolation as a string (‘linear’, ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’, ‘next’, where ‘zero’, ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a spline interpolation of zeroth, first, second or third order; ‘previous’ and ‘next’ simply return the previous or next value of the point) or as an integer specifying the order of the spline interpolator to use. Default is ‘nearest\"

    returns
    -------
    modulation: ndarray
        the modulation vector, interpolated from the anchors and with a len of samples
    
    """
    anchors = np.atleast_1d((anchors))
    N = len(anchors)
    if N < 1:
        raise ValueError("You need at least one anchors for any kind of interpolation")
    elif N == 1:
        return np.ones((samples)) * anchors[0]
    elif kind == "quadratic" and N < 3:
        raise ValueError("You need at least three anchors for quadratic interpolation")
    elif kind == "cubic" and N < 4:
        raise ValueError("You need at least four anchors for cubic interpolation")
    else:
        pass

    x_given = np.arange(0.0, N, 1.0)
    x_queried = np.linspace(0, N - 1, samples, endpoint=True)
    f = interp1d(x_given, anchors, kind=kind)
    modulation = f(x_queried)
    return modulation
