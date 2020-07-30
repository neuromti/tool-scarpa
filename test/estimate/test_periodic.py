import pytest
from scarpa.generate.modulation import create_modulation
from scarpa.generate.template import stack_periodic
from scarpa.generate.shapes import sinus, hanning, gaussdiff, noise
from scipy.optimize import minimize


@pytest.fixture
def mixture():
    pcount = 23.5
    plen = 100
    samples = int(pcount * plen)

    modulation = create_modulation(anchors=hanning(23), samples=samples, kind="cubic")

    shape = sinus(plen)

    pure = stack_periodic(shape, periodcount=pcount)
    signal = pure * modulation
    yield signal


def test_fit_periodic(mixture):
    print(mixture)

