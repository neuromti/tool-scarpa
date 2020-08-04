import numpy as np
from scipy.interpolate import interp1d


def resample_by_fs(indata, up=1000, down=1000, axis=0):
    l = indata.shape[0]
    new_sample_count = int(np.ceil(l * (up / down)))
    sample_idx = np.atleast_1d(np.linspace(0, l, l))
    f = interp1d(np.atleast_1d(sample_idx), np.atleast_1d(indata), kind="cubic")
    new_sample_idx = np.linspace(0, l, new_sample_count)
    return f(new_sample_idx)


def resample_by_count(indata, new_sample_count, axis=0):
    l = indata.shape[0]
    sample_idx = np.atleast_1d(np.linspace(0, l, l))
    f = interp1d(np.atleast_1d(sample_idx), np.atleast_1d(indata), kind="cubic")
    new_sample_idx = np.linspace(0, l, new_sample_count)
    return f(new_sample_idx)
