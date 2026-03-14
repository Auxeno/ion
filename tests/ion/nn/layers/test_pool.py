import jax.numpy as jnp
import numpy.testing as npt
import pytest

from ion import nn


class TestMaxPool:
    def test_int_kernel_shape_raises(self):
        """Passing an int instead of a tuple raises TypeError."""
        with pytest.raises(TypeError, match="kernel_shape"):
            nn.MaxPool(kernel_shape=2)  # type: ignore[arg-type]

    def test_1d_picks_max(self):
        """Each output element is the maximum of its window."""
        layer = nn.MaxPool(kernel_shape=(2,))
        x = jnp.array([[[1.0], [3.0], [2.0], [5.0], [4.0], [6.0]]])  # (1, 6, 1)
        y = layer(x)
        expected = jnp.array([[[3.0], [5.0], [6.0]]])  # (1, 3, 1)
        npt.assert_allclose(y, expected)

    def test_1d_stride_reduces_spatial(self):
        """Stride 2 halves spatial dimension."""
        layer = nn.MaxPool(kernel_shape=(2,), stride=2)
        x = jnp.ones((1, 8, 3))
        y = layer(x)
        assert y.shape == (1, 4, 3)

    def test_2d_picks_max(self):
        """Each output element is the maximum of its 2x2 window."""
        layer = nn.MaxPool(kernel_shape=(2, 2))
        x = jnp.array(
            [
                [
                    [[1.0], [2.0], [3.0], [4.0]],
                    [[5.0], [6.0], [7.0], [8.0]],
                    [[9.0], [10.0], [11.0], [12.0]],
                    [[13.0], [14.0], [15.0], [16.0]],
                ]
            ]
        )  # (1, 4, 4, 1)
        y = layer(x)
        expected = jnp.array(
            [
                [
                    [[6.0], [8.0]],
                    [[14.0], [16.0]],
                ]
            ]
        )  # (1, 2, 2, 1)
        npt.assert_allclose(y, expected)

    def test_2d_stride_reduces_spatial(self):
        """Stride 2 halves spatial dimensions."""
        layer = nn.MaxPool(kernel_shape=(2, 2), stride=2)
        x = jnp.ones((1, 8, 8, 3))
        y = layer(x)
        assert y.shape == (1, 4, 4, 3)


class TestMaxPoolConstructor:
    def test_empty_kernel_shape_raises(self):
        """Empty kernel_shape raises ValueError."""
        with pytest.raises(ValueError, match="at least one element"):
            nn.MaxPool(kernel_shape=())

    def test_explicit_tuple_stride(self):
        """Explicit tuple stride controls output spatial size."""
        pool = nn.MaxPool(kernel_shape=(2, 2), stride=(1, 1))
        x = jnp.ones((1, 4, 4, 3))
        assert pool(x).shape == (1, 3, 3, 3)

    def test_string_padding(self):
        """String padding like 'SAME' is passed through to lax."""
        pool = nn.MaxPool(kernel_shape=(2, 2), padding="SAME")
        x = jnp.ones((1, 4, 4, 3))
        assert pool(x).shape == (1, 2, 2, 3)

    def test_tuple_padding(self):
        """Tuple padding specifies per-dimension symmetric padding."""
        pool = nn.MaxPool(kernel_shape=(2, 2), padding=(1, 1))
        x = jnp.ones((1, 4, 4, 3))
        y = pool(x)
        assert y.shape[1] > 0 and y.shape[2] > 0


class TestAvgPool:
    def test_int_kernel_shape_raises(self):
        """Passing an int instead of a tuple raises TypeError."""
        with pytest.raises(TypeError, match="kernel_shape"):
            nn.AvgPool(kernel_shape=2)  # type: ignore[arg-type]

    def test_1d_computes_mean(self):
        """Each output element is the mean of its window."""
        layer = nn.AvgPool(kernel_shape=(2,))
        x = jnp.array([[[1.0], [3.0], [2.0], [6.0], [4.0], [8.0]]])  # (1, 6, 1)
        y = layer(x)
        expected = jnp.array([[[2.0], [4.0], [6.0]]])  # (1, 3, 1)
        npt.assert_allclose(y, expected)

    def test_1d_stride_reduces_spatial(self):
        """Stride 2 halves spatial dimension."""
        layer = nn.AvgPool(kernel_shape=(2,), stride=2)
        x = jnp.ones((1, 8, 3))
        y = layer(x)
        assert y.shape == (1, 4, 3)

    def test_1d_padding_averages_only_real_elements(self):
        """Padding should not count padded zeros in the average."""
        layer = nn.AvgPool(kernel_shape=(3,), stride=1, padding=1)
        x = jnp.array([[[6.0], [6.0], [6.0], [6.0]]])  # (1, 4, 1)
        y = layer(x)
        expected = jnp.array([[[6.0], [6.0], [6.0], [6.0]]])  # (1, 4, 1)
        npt.assert_allclose(y, expected)

    def test_2d_computes_mean(self):
        """Each output element is the mean of its 2x2 window."""
        layer = nn.AvgPool(kernel_shape=(2, 2))
        x = jnp.array(
            [
                [
                    [[1.0], [2.0], [3.0], [4.0]],
                    [[5.0], [6.0], [7.0], [8.0]],
                    [[9.0], [10.0], [11.0], [12.0]],
                    [[13.0], [14.0], [15.0], [16.0]],
                ]
            ]
        )  # (1, 4, 4, 1)
        y = layer(x)
        expected = jnp.array(
            [
                [
                    [[3.5], [5.5]],
                    [[11.5], [13.5]],
                ]
            ]
        )  # (1, 2, 2, 1)
        npt.assert_allclose(y, expected)

    def test_2d_stride_reduces_spatial(self):
        """Stride 2 halves spatial dimensions."""
        layer = nn.AvgPool(kernel_shape=(2, 2), stride=2)
        x = jnp.ones((1, 8, 8, 3))
        y = layer(x)
        assert y.shape == (1, 4, 4, 3)

    def test_2d_padding_averages_only_real_elements(self):
        """Padding should not count padded zeros in the average."""
        layer = nn.AvgPool(kernel_shape=(3, 3), stride=1, padding=1)
        x = jnp.full((1, 4, 4, 1), 10.0)  # (1, 4, 4, 1)
        y = layer(x)
        expected = jnp.full((1, 4, 4, 1), 10.0)  # (1, 4, 4, 1)
        npt.assert_allclose(y, expected)


class TestAvgPoolConstructor:
    def test_empty_kernel_shape_raises(self):
        """Empty kernel_shape raises ValueError."""
        with pytest.raises(ValueError, match="at least one element"):
            nn.AvgPool(kernel_shape=())

    def test_explicit_tuple_stride(self):
        """Explicit tuple stride controls output spatial size."""
        pool = nn.AvgPool(kernel_shape=(2, 2), stride=(1, 1))
        x = jnp.ones((1, 4, 4, 3))
        assert pool(x).shape == (1, 3, 3, 3)

    def test_string_padding(self):
        """String padding like 'SAME' is passed through to lax."""
        pool = nn.AvgPool(kernel_shape=(2, 2), padding="SAME")
        x = jnp.ones((1, 4, 4, 3))
        assert pool(x).shape == (1, 2, 2, 3)

    def test_tuple_padding(self):
        """Tuple padding specifies per-dimension symmetric padding."""
        pool = nn.AvgPool(kernel_shape=(2, 2), padding=(1, 1))
        x = jnp.ones((1, 4, 4, 3))
        y = pool(x)
        assert y.shape[1] > 0 and y.shape[2] > 0
