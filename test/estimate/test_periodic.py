import pytest
from scarpa.generate.modulation import create_modulation
from scarpa.generate.template import stack_periodic
from scarpa.generate.shapes import sinus, hanning
from scarpa.estimate.periodic import fit_with_plen_known, vec2parms
import numpy as np


@pytest.fixture
def mixture():
    pcount = 23.5
    plen = 100
    samples = int(pcount * plen)
    anchors = hanning(23)
    modulation = create_modulation(anchors=anchors, samples=samples, kind="cubic")

    template = sinus(plen)

    pure = stack_periodic(template, periodcount=pcount)
    data = pure * modulation
    yield data, template, modulation


def test_fit_with_plen_known(mixture):
    data, template, modulation = mixture
    mn_s, template_s, modulation_s = fit_with_plen_known(
        data, p_len=100, am_count=23, am_interp="cubic"
    )

    assert np.corrcoef(modulation_s, modulation)[0, 1] > 0.9999
    assert np.corrcoef(template_s, template)[0, 1] > 0.9999
