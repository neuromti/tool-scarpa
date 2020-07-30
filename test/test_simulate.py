from scarpa.simulate import generate_sinus
import pytest
import numpy as np


@pytest.mark.parametrize("amplitude, frequency, duration, fs", [(1, 10, 1, 1000)])
def test_generate_sinus(amplitude, frequency, duration, fs):
    y = generate_sinus(amplitude, frequency, duration, fs)
    assert np.round(np.ptp(y), 2) == 2 * amplitude
    assert y.shape[0] == duration * fs
