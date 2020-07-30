"""generate shapes as basis for periodic signals"""
from numpy import ndarray
import numpy as np
import scipy.signal as sg


def _validate(period_len: int) -> int:
    if type(period_len) is not int:
        raise ValueError("You need to specify the period_len as a integer")
    if period_len < 1:
        raise ValueError("You need at least one sample for a periodic signal")
    return period_len


def _phi(period_len: int) -> ndarray:
    "period_len equidistant steps along the unit circle in radians"
    t = np.linspace(0, 1, _validate(period_len), endpoint=False)
    return 2 * np.pi * t


def sinus(period_len: int) -> ndarray:
    "a single period of a sine wave over period_len samples"
    return np.sin(_phi(period_len))


def sawtooth(period_len: int) -> ndarray:
    "a single period of a sawtooth wave over period_len samples"
    return sg.sawtooth(_phi(period_len))


def square(period_len: int) -> ndarray:
    "a single period of a square wave over period_len samples"
    return sg.square(_phi(period_len))


def ones(period_len: int) -> ndarray:
    "a flat wave over period_len samples"
    return np.ones(_validate(period_len))
