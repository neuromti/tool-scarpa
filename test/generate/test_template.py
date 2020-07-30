from scarpa.generate.template import stack_periodic
from numpy import arange
import pytest


@pytest.mark.parametrize(
    "pcount, plen, explen",
    [(1, 10, 10), (2, 11, 22), (1.5, 10, 15), (1.78, 10, 17), (1.91, 10, 19)],
)
def test_stack_periodic_len(pcount, plen, explen):
    template = arange(0, plen)
    signal = stack_periodic(template, periodcount=pcount)
    assert len(signal) == explen


@pytest.mark.parametrize(
    "pcount, plen, expval",
    [(1, 10, 10), (2, 11, 11), (1.5, 10, 5), (1.78, 10, 7), (1.91, 10, 9)],
)
def test_stack_periodic_val(pcount, plen, expval):
    template = arange(1, plen + 1)
    signal = stack_periodic(template, periodcount=pcount)
    assert signal[-1] == expval
    assert signal[0] == 1


@pytest.mark.parametrize("pcount", [0, -1])
def test_stack_periodic_raises(pcount):
    template = arange(0, 10)
    with pytest.raises(ValueError):
        signal = stack_periodic(template, periodcount=pcount)

