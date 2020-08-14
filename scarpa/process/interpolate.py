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


class Resampler:
    def __init__(self, fs: int, freq: float):
        self.true_fs = fs
        self.true_freq = freq
        if freq is not None:
            self.true_period = fs / freq
            self.resample_flag = self.true_period != int(self.true_fs / self.true_freq)
        else:
            self.true_period = None

    def inbound_resample(self, indata):
        "resample so that (artifact_period* artifact_frequency) is an integer"
        self.sample_count = indata.shape[0]
        if self.resample_flag:
            period = int(np.ceil(self.true_period))
            fs = int(np.ceil(period * self.true_freq))
            data = resample_by_fs(indata, up=fs, down=self.true_fs, axis=0)
        else:
            data = indata
            fs = self.true_fs
            period = int(self.true_period)
        return data, period, fs

    def outbound_resample(self, outdata):
        "reverse an earlier resampling, if it was necessary"
        if self.resample_flag:
            outdata = resample_by_count(outdata, self.sample_count, axis=0)
        return outdata
