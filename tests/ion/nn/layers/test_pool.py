import jax.numpy as jnp
import numpy.testing as npt

from ion import nn


class TestMaxPool1d:
    def test_picks_max(self):
        """Each output element is the maximum of its window."""
        layer = nn.MaxPool1d(kernel_size=2)
        x = jnp.array([[1.0], [3.0], [2.0], [5.0], [4.0], [6.0]])
        y = layer(x)
        expected = jnp.array([[3.0], [5.0], [6.0]])
        npt.assert_allclose(y, expected)

    def test_stride_reduces_spatial(self):
        """Stride 2 halves spatial dimension."""
        layer = nn.MaxPool1d(kernel_size=2, stride=2)
        x = jnp.ones((8, 3))
        y = layer(x)
        assert y.shape == (4, 3)


class TestAvgPool1d:
    def test_computes_mean(self):
        """Each output element is the mean of its window."""
        layer = nn.AvgPool1d(kernel_size=2)
        x = jnp.array([[1.0], [3.0], [2.0], [6.0], [4.0], [8.0]])
        y = layer(x)
        expected = jnp.array([[2.0], [4.0], [6.0]])
        npt.assert_allclose(y, expected)

    def test_stride_reduces_spatial(self):
        """Stride 2 halves spatial dimension."""
        layer = nn.AvgPool1d(kernel_size=2, stride=2)
        x = jnp.ones((8, 3))
        y = layer(x)
        assert y.shape == (4, 3)


class TestMaxPool2d:
    def test_picks_max(self):
        """Each output element is the maximum of its 2x2 window."""
        layer = nn.MaxPool2d(kernel_size=2)
        x = jnp.array(
            [
                [[1.0], [2.0], [3.0], [4.0]],
                [[5.0], [6.0], [7.0], [8.0]],
                [[9.0], [10.0], [11.0], [12.0]],
                [[13.0], [14.0], [15.0], [16.0]],
            ]
        )
        y = layer(x)
        expected = jnp.array(
            [
                [[6.0], [8.0]],
                [[14.0], [16.0]],
            ]
        )
        npt.assert_allclose(y, expected)

    def test_stride_reduces_spatial(self):
        """Stride 2 halves spatial dimensions."""
        layer = nn.MaxPool2d(kernel_size=2, stride=2)
        x = jnp.ones((8, 8, 3))
        y = layer(x)
        assert y.shape == (4, 4, 3)


class TestAvgPool2d:
    def test_computes_mean(self):
        """Each output element is the mean of its 2x2 window."""
        layer = nn.AvgPool2d(kernel_size=2)
        x = jnp.array(
            [
                [[1.0], [2.0], [3.0], [4.0]],
                [[5.0], [6.0], [7.0], [8.0]],
                [[9.0], [10.0], [11.0], [12.0]],
                [[13.0], [14.0], [15.0], [16.0]],
            ]
        )
        y = layer(x)
        expected = jnp.array(
            [
                [[3.5], [5.5]],
                [[11.5], [13.5]],
            ]
        )
        npt.assert_allclose(y, expected)

    def test_stride_reduces_spatial(self):
        """Stride 2 halves spatial dimensions."""
        layer = nn.AvgPool2d(kernel_size=2, stride=2)
        x = jnp.ones((8, 8, 3))
        y = layer(x)
        assert y.shape == (4, 4, 3)
