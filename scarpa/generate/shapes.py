"""generate shapes as basis for periodic signals or amplitude modulation"""
from numpy import ndarray
import numpy as np
import scipy.signal as sg


def _validate(samples: int) -> int:
    "validate that samples is a positive integer"
    if type(samples) is not int:
        raise ValueError("You need to specify the samples as a integer")
    if samples < 1:
        raise ValueError("You need at least one sample for a periodic signal")
    return samples


def _phi(samples: int) -> ndarray:
    "equidistant steps along the unit circle in radians"
    t = np.linspace(0, 1, _validate(samples), endpoint=False)
    return 2 * np.pi * t


def sinus(samples: int) -> ndarray:
    "a single period of a sine wave of samples length"
    return np.sin(_phi(samples))


def sawtooth(samples: int) -> ndarray:
    "a single period of a sawtooth wave of samples length"
    return sg.sawtooth(_phi(samples))


def square(samples: int) -> ndarray:
    "a single period of a square wave of samples length"
    return sg.square(_phi(samples))


def ones(samples: int) -> ndarray:
    "a flat wave of samples length"
    return np.ones(_validate(samples))


def hanning(samples: int) -> ndarray:
    "a hanning window of samples length"
    return sg.windows.hann(_validate(samples))


def noise(samples: int) -> ndarray:
    "white noise of samples length"
    return np.random.randn(_validate(samples))


def gaussdiff(samples: int) -> ndarray:
    "differential gaussian waveform of samples length"
    return np.gradient(gaussian(samples))


def gaussian(samples: int) -> ndarray:
    "gaussian waveform of samples length"
    N: int = _validate(samples)
    return sg.gaussian(M=N, std=N / 10)


def mexicanhat(samples: int) -> ndarray:
    "mexican hat (ricker) of samples length"
    N: int = _validate(samples)
    return sg.ricker(points=N, a=N / 10)
