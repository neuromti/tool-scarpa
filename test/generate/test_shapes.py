from scarpa.generate.shapes import *
import pytest


@pytest.mark.parametrize(
    "foo",
    [ones, sinus, sawtooth, square, hanning, gaussian, gaussdiff, mexicanhat, noise,],
)
class Test_shapes:
    @pytest.mark.parametrize("plen", [-1, 0, 10.1])
    def test_shape_raises(self, plen, foo):
        with pytest.raises(ValueError):
            foo(plen)

    @pytest.mark.parametrize("plen", [15, 101])
    def test_shape_valid(self, plen, foo):
        shape = foo(plen)
        assert len(shape) == plen
