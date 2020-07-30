import pytest
from scarpa.generate.modulation import create_modulation


@pytest.mark.parametrize(
    "kind",
    ["linear", "nearest", "zero", "slinear", "quadratic", "cubic", "previous", "next"],
)
def test_create_modulation(kind, anchors=[1.0, 0, 3, 4, -1.0], samples=100):
    mod = create_modulation(anchors, samples, kind)
    assert mod[0] == anchors[0]
    assert mod[-1] == anchors[-1]
    assert len(mod) == samples


def test_create_modulation_fallback_to_constant(anchors=[1.0], samples=100):
    mod = create_modulation(anchors, samples)
    assert mod[0] == anchors[0]
    assert mod[-1] == anchors[0]
    assert len(mod) == samples


@pytest.mark.parametrize(
    "anchors, kind", [([], "linear"), ([1, 1], "quadratic"), ([1, 1, 1], "cubic")]
)
def test_create_modulation_raises(anchors, kind):
    with pytest.raises(ValueError):
        mod = create_modulation(anchors, 100, kind)

