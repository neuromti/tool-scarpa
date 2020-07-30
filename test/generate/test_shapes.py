from scarpa.generate.shapes import ones, sinus, sawtooth, square
import pytest


@pytest.mark.parametrize(
    "foo, elem", [(ones, 1), (sinus, 0), (sawtooth, -1), (square, 1)]
)
class Test_shapes:
    @pytest.mark.parametrize("plen", [-1, 0, 10.1])
    def test_shape_raises(self, plen, foo, elem):
        with pytest.raises(ValueError):
            foo(plen)

    @pytest.mark.parametrize("plen", [1, 10])
    def test_shape_valid(self, plen, foo, elem):
        shape = foo(plen)
        assert len(shape) == plen
        assert shape[0] == elem
