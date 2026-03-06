import jax.numpy as jnp
import numpy.testing as npt

from ion import nn


class TestUpsample:
    def test_1d_output_shape(self):
        """Scale factor 2 doubles the temporal dimension."""
        layer = nn.Upsample(1, scale_factor=2)
        x = jnp.ones((4, 3))
        y = layer(x)
        assert y.shape == (8, 3)

    def test_1d_output_shape_batch(self):
        """Arbitrary batch dimensions are preserved."""
        layer = nn.Upsample(1, scale_factor=3)
        x = jnp.ones((2, 5, 4, 3))
        y = layer(x)
        assert y.shape == (2, 5, 12, 3)

    def test_1d_nearest(self):
        """Nearest-neighbor repeats each element."""
        layer = nn.Upsample(1, scale_factor=2, mode="nearest")
        x = jnp.array([[1.0], [2.0], [3.0]])
        y = layer(x)
        expected = jnp.array([[1.0], [1.0], [2.0], [2.0], [3.0], [3.0]])
        npt.assert_allclose(y, expected)

    def test_1d_bilinear(self):
        """Linear mode produces an output of the correct shape."""
        layer = nn.Upsample(1, scale_factor=2, mode="linear")
        x = jnp.ones((6, 3))
        y = layer(x)
        assert y.shape == (12, 3)

    def test_2d_output_shape(self):
        """Scale factor 2 doubles both spatial dimensions."""
        layer = nn.Upsample(2, scale_factor=2)
        x = jnp.ones((4, 4, 3))
        y = layer(x)
        assert y.shape == (8, 8, 3)

    def test_2d_output_shape_batch(self):
        """Arbitrary batch dimensions are preserved."""
        layer = nn.Upsample(2, scale_factor=2)
        x = jnp.ones((2, 5, 4, 4, 3))
        y = layer(x)
        assert y.shape == (2, 5, 8, 8, 3)

    def test_2d_int_scale_factor(self):
        """Int scale factor is applied to both dimensions."""
        layer = nn.Upsample(2, scale_factor=3)
        x = jnp.ones((2, 2, 1))
        y = layer(x)
        assert y.shape == (6, 6, 1)

    def test_2d_tuple_scale_factor(self):
        """Tuple scale factor applies different factors per dimension."""
        layer = nn.Upsample(2, scale_factor=(2, 3))
        x = jnp.ones((4, 4, 1))
        y = layer(x)
        assert y.shape == (8, 12, 1)

    def test_2d_nearest(self):
        """Nearest-neighbor repeats each element in a 2x2 block."""
        layer = nn.Upsample(2, scale_factor=2, mode="nearest")
        x = jnp.array(
            [
                [[1.0], [2.0]],
                [[3.0], [4.0]],
            ]
        )
        y = layer(x)
        expected = jnp.array(
            [
                [[1.0], [1.0], [2.0], [2.0]],
                [[1.0], [1.0], [2.0], [2.0]],
                [[3.0], [3.0], [4.0], [4.0]],
                [[3.0], [3.0], [4.0], [4.0]],
            ]
        )
        npt.assert_allclose(y, expected)

    def test_2d_bilinear(self):
        """Bilinear mode produces an output of the correct shape."""
        layer = nn.Upsample(2, scale_factor=2, mode="bilinear")
        x = jnp.ones((4, 4, 3))
        y = layer(x)
        assert y.shape == (8, 8, 3)
