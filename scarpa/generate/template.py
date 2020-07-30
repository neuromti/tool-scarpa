import numpy as np
from numpy import ndarray
from scipy.interpolate import interp1d


def stack_periodic(template: ndarray, periodcount: float):
    """stack a period signal from a template 
    
    args
    ----

    template: ndarray
        the waveform along one periodic

    periodcount: float
        the duration of the complete signal in periods. Fractions are allowed


    """
    if periodcount <= 0:
        raise ValueError("Periodcount can not be smaller than 0")

    plen = len(template)
    model = np.empty(0)
    reps, remainder = int(periodcount // 1), int(plen * (periodcount % 1))

    for rep in range(reps):
        model = np.hstack((model, template))

    model = np.hstack((model, template[:remainder]))

    return model