import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from ion import nn


class TestConv:
    def test_int_kernel_shape_raises(self):
        """Passing an int kernel_shape raises TypeError."""
        with pytest.raises(TypeError, match="kernel_shape"):
            nn.Conv(3, 8, kernel_shape=3, key=jax.random.key(0))

    def test_empty_kernel_shape_raises(self):
        """Empty kernel_shape raises ValueError."""
        with pytest.raises(ValueError, match="kernel_shape"):
            nn.Conv(3, 8, kernel_shape=(), key=jax.random.key(0))

    def test_1d_output_shape_no_padding(self):
        """No padding shrinks spatial dim by kernel_size - 1."""
        layer = nn.Conv(3, 8, kernel_shape=(3,), padding=0, key=jax.random.key(0))
        x = jnp.ones((10, 3))
        y = layer(x)
        assert y.shape == (8, 8)

    def test_1d_output_shape_same_padding(self):
        """SAME padding preserves spatial dimension."""
        layer = nn.Conv(3, 8, kernel_shape=(3,), padding="SAME", key=jax.random.key(0))
        x = jnp.ones((10, 3))
        y = layer(x)
        assert y.shape == (10, 8)

    def test_1d_output_shape_valid_padding(self):
        """VALID padding is equivalent to zero padding."""
        layer = nn.Conv(3, 8, kernel_shape=(3,), padding="VALID", key=jax.random.key(0))
        x = jnp.ones((10, 3))
        y = layer(x)
        assert y.shape == (8, 8)

    def test_1d_output_shape_int_padding(self):
        """Integer padding applies symmetrically."""
        layer = nn.Conv(3, 8, kernel_shape=(3,), padding=1, key=jax.random.key(0))
        x = jnp.ones((10, 3))
        y = layer(x)
        assert y.shape == (10, 8)

    def test_1d_stride_reduces_spatial(self):
        """Stride > 1 reduces spatial dimension."""
        layer = nn.Conv(3, 8, kernel_shape=(3,), stride=2, padding=1, key=jax.random.key(0))
        x = jnp.ones((10, 3))
        y = layer(x)
        assert y.shape == (5, 8)

    def test_1d_he_normal_init(self):
        """He normal initialization gives var(w) close to 2/fan_in."""
        layer = nn.Conv(64, 64, kernel_shape=(3,), key=jax.random.key(42))
        var = jnp.var(layer.w.value)
        fan_in = 3 * 64
        expected_var = 2.0 / fan_in
        npt.assert_allclose(var, expected_var, atol=0.05)

    def test_1d_zero_bias_init(self):
        """Bias is initialized to all zeros."""
        layer = nn.Conv(3, 8, kernel_shape=(3,), key=jax.random.key(0))
        assert jnp.all(layer.b == 0)

    def test_1d_weight_dtype(self):
        """Weights match the requested dtype."""
        layer = nn.Conv(3, 8, kernel_shape=(3,), dtype=jnp.float32, key=jax.random.key(0))
        assert layer.w.dtype == jnp.float32

    def test_1d_groups_output_shape(self):
        """Grouped conv produces correct spatial shape."""
        layer = nn.Conv(8, 16, kernel_shape=(3,), groups=4, padding=1, key=jax.random.key(0))
        x = jnp.ones((10, 8))
        y = layer(x)
        assert y.shape == (10, 16)

    def test_1d_groups_kernel_shape(self):
        """Kernel has in_channels // groups input dim."""
        layer = nn.Conv(8, 16, kernel_shape=(3,), groups=4, key=jax.random.key(0))
        assert layer.w.shape == (3, 2, 16)

    def test_1d_depthwise_output_shape(self):
        """Depthwise conv (groups=in_channels) works."""
        layer = nn.Conv(8, 8, kernel_shape=(3,), groups=8, padding=1, key=jax.random.key(0))
        x = jnp.ones((10, 8))
        y = layer(x)
        assert y.shape == (10, 8)

    def test_1d_groups_invalid(self):
        """Mismatched channels/groups raises ValueError."""
        with pytest.raises(ValueError, match="in_channels"):
            nn.Conv(7, 8, kernel_shape=(3,), groups=4, key=jax.random.key(0))
        with pytest.raises(ValueError, match="out_channels"):
            nn.Conv(8, 7, kernel_shape=(3,), groups=4, key=jax.random.key(0))

    def test_2d_output_shape_no_padding(self):
        """No padding shrinks spatial dims by kernel_size - 1."""
        layer = nn.Conv(3, 8, kernel_shape=(3, 3), padding=0, key=jax.random.key(0))
        x = jnp.ones((6, 6, 3))
        y = layer(x)
        assert y.shape == (4, 4, 8)

    def test_2d_output_shape_same_padding(self):
        """SAME padding preserves spatial dimensions."""
        layer = nn.Conv(3, 8, kernel_shape=(3, 3), padding="SAME", key=jax.random.key(0))
        x = jnp.ones((6, 6, 3))
        y = layer(x)
        assert y.shape == (6, 6, 8)

    def test_2d_output_shape_valid_padding(self):
        """VALID padding is equivalent to zero padding."""
        layer = nn.Conv(3, 8, kernel_shape=(3, 3), padding="VALID", key=jax.random.key(0))
        x = jnp.ones((6, 6, 3))
        y = layer(x)
        assert y.shape == (4, 4, 8)

    def test_2d_output_shape_int_padding(self):
        """Integer padding applies symmetrically to both spatial dims."""
        layer = nn.Conv(3, 8, kernel_shape=(3, 3), padding=1, key=jax.random.key(0))
        x = jnp.ones((6, 6, 3))
        y = layer(x)
        assert y.shape == (6, 6, 8)

    def test_2d_output_shape_tuple_padding(self):
        """Tuple padding applies different values per spatial dim."""
        layer = nn.Conv(3, 8, kernel_shape=(3, 3), padding=(1, 2), key=jax.random.key(0))
        x = jnp.ones((6, 6, 3))
        y = layer(x)
        assert y.shape == (6, 8, 8)

    def test_2d_stride_reduces_spatial(self):
        """Stride > 1 reduces spatial dimensions."""
        layer = nn.Conv(3, 8, kernel_shape=(3, 3), stride=2, padding=1, key=jax.random.key(0))
        x = jnp.ones((6, 6, 3))
        y = layer(x)
        assert y.shape == (3, 3, 8)

    def test_2d_he_normal_init(self):
        """He normal initialization gives var(w) close to 2/fan_in."""
        layer = nn.Conv(64, 64, kernel_shape=(3, 3), key=jax.random.key(42))
        var = jnp.var(layer.w.value)
        fan_in = 3 * 3 * 64
        expected_var = 2.0 / fan_in
        npt.assert_allclose(var, expected_var, atol=0.05)

    def test_2d_zero_bias_init(self):
        """Bias is initialized to all zeros."""
        layer = nn.Conv(3, 8, kernel_shape=(3, 3), key=jax.random.key(0))
        assert jnp.all(layer.b == 0)

    def test_2d_weight_dtype(self):
        """Weights match the requested dtype."""
        layer = nn.Conv(3, 8, kernel_shape=(3, 3), dtype=jnp.float32, key=jax.random.key(0))
        assert layer.w.dtype == jnp.float32

    def test_2d_groups_output_shape(self):
        """Grouped conv produces correct spatial shape."""
        layer = nn.Conv(8, 16, kernel_shape=(3, 3), groups=4, padding=1, key=jax.random.key(0))
        x = jnp.ones((6, 6, 8))
        y = layer(x)
        assert y.shape == (6, 6, 16)

    def test_2d_groups_kernel_shape(self):
        """Kernel has in_channels // groups input dim."""
        layer = nn.Conv(8, 16, kernel_shape=(3, 3), groups=4, key=jax.random.key(0))
        assert layer.w.shape == (3, 3, 2, 16)

    def test_2d_depthwise_output_shape(self):
        """Depthwise conv (groups=in_channels) works."""
        layer = nn.Conv(8, 8, kernel_shape=(3, 3), groups=8, padding=1, key=jax.random.key(0))
        x = jnp.ones((6, 6, 8))
        y = layer(x)
        assert y.shape == (6, 6, 8)

    def test_2d_groups_invalid(self):
        """Mismatched channels/groups raises ValueError."""
        with pytest.raises(ValueError, match="in_channels"):
            nn.Conv(7, 8, kernel_shape=(3, 3), groups=4, key=jax.random.key(0))
        with pytest.raises(ValueError, match="out_channels"):
            nn.Conv(8, 7, kernel_shape=(3, 3), groups=4, key=jax.random.key(0))

    def test_3d_output_shape(self):
        """Conv with 3D kernel_shape works for 3D inputs."""
        layer = nn.Conv(3, 8, kernel_shape=(3, 3, 3), padding=1, key=jax.random.key(0))
        x = jnp.ones((4, 4, 4, 3))
        y = layer(x)
        assert y.shape == (4, 4, 4, 8)


class TestConvTranspose:
    def test_int_kernel_shape_raises(self):
        """Passing an int kernel_shape raises TypeError."""
        with pytest.raises(TypeError, match="kernel_shape"):
            nn.ConvTranspose(3, 8, kernel_shape=3, key=jax.random.key(0))

    def test_empty_kernel_shape_raises(self):
        """Empty kernel_shape raises ValueError."""
        with pytest.raises(ValueError, match="kernel_shape"):
            nn.ConvTranspose(3, 8, kernel_shape=(), key=jax.random.key(0))

    def test_1d_output_shape_no_padding(self):
        """No padding expands spatial dim by kernel_size - 1."""
        layer = nn.ConvTranspose(3, 8, kernel_shape=(3,), padding=0, key=jax.random.key(0))
        x = jnp.ones((10, 3))
        y = layer(x)
        assert y.shape == (12, 8)

    def test_1d_output_shape_same_padding(self):
        """SAME padding preserves spatial dimension."""
        layer = nn.ConvTranspose(3, 8, kernel_shape=(3,), padding="SAME", key=jax.random.key(0))
        x = jnp.ones((10, 3))
        y = layer(x)
        assert y.shape == (10, 8)

    def test_1d_output_shape_valid_padding(self):
        """VALID padding expands spatial dimension."""
        layer = nn.ConvTranspose(3, 8, kernel_shape=(3,), padding="VALID", key=jax.random.key(0))
        x = jnp.ones((10, 3))
        y = layer(x)
        assert y.shape == (12, 8)

    def test_1d_output_shape_int_padding(self):
        """Integer padding applies symmetrically."""
        layer = nn.ConvTranspose(3, 8, kernel_shape=(3,), padding=1, key=jax.random.key(0))
        x = jnp.ones((10, 3))
        y = layer(x)
        assert y.shape == (10, 8)

    def test_1d_stride_upsamples(self):
        """Stride > 1 upsamples spatial dimension."""
        layer = nn.ConvTranspose(3, 8, kernel_shape=(3,), stride=2, padding=1, key=jax.random.key(0))
        x = jnp.ones((10, 3))
        y = layer(x)
        assert y.shape == (19, 8)

    def test_1d_output_padding(self):
        """output_padding adds extra elements to resolve stride ambiguity."""
        layer = nn.ConvTranspose(
            3, 8, kernel_shape=(3,), stride=2, padding=1, output_padding=1, key=jax.random.key(0)
        )
        x = jnp.ones((10, 3))
        y = layer(x)
        assert y.shape == (20, 8)

    def test_1d_dilation(self):
        """Dilation expands the effective kernel size."""
        layer = nn.ConvTranspose(3, 8, kernel_shape=(3,), dilation=2, key=jax.random.key(0))
        x = jnp.ones((10, 3))
        y = layer(x)
        assert y.shape == (14, 8)

    def test_1d_he_normal_init(self):
        """He normal initialization gives var(w) close to 2/fan_in."""
        layer = nn.ConvTranspose(64, 64, kernel_shape=(3,), key=jax.random.key(42))
        var = jnp.var(layer.w.value)
        fan_in = 3 * 64
        expected_var = 2.0 / fan_in
        npt.assert_allclose(var, expected_var, atol=0.05)

    def test_1d_zero_bias_init(self):
        """Bias is initialized to all zeros."""
        layer = nn.ConvTranspose(3, 8, kernel_shape=(3,), key=jax.random.key(0))
        assert jnp.all(layer.b == 0)

    def test_1d_weight_dtype(self):
        """Weights match the requested dtype."""
        layer = nn.ConvTranspose(3, 8, kernel_shape=(3,), dtype=jnp.float32, key=jax.random.key(0))
        assert layer.w.dtype == jnp.float32

    def test_1d_groups_output_shape(self):
        """Grouped transposed conv produces correct spatial shape."""
        layer = nn.ConvTranspose(
            8, 16, kernel_shape=(3,), groups=4, padding=1, key=jax.random.key(0)
        )
        x = jnp.ones((10, 8))
        y = layer(x)
        assert y.shape == (10, 16)

    def test_1d_groups_kernel_shape(self):
        """Kernel has in_channels // groups input dim."""
        layer = nn.ConvTranspose(8, 16, kernel_shape=(3,), groups=4, key=jax.random.key(0))
        assert layer.w.shape == (3, 2, 16)

    def test_1d_groups_invalid(self):
        """Mismatched channels/groups raises ValueError."""
        with pytest.raises(ValueError, match="in_channels"):
            nn.ConvTranspose(7, 8, kernel_shape=(3,), groups=4, key=jax.random.key(0))
        with pytest.raises(ValueError, match="out_channels"):
            nn.ConvTranspose(8, 7, kernel_shape=(3,), groups=4, key=jax.random.key(0))

    def test_1d_output_padding_validation(self):
        """output_padding >= stride raises ValueError."""
        with pytest.raises(ValueError, match="output_padding"):
            nn.ConvTranspose(3, 8, kernel_shape=(3,), stride=2, output_padding=2, key=jax.random.key(0))

    def test_2d_output_shape_no_padding(self):
        """No padding expands spatial dims by kernel_size - 1."""
        layer = nn.ConvTranspose(3, 8, kernel_shape=(3, 3), padding=0, key=jax.random.key(0))
        x = jnp.ones((6, 6, 3))
        y = layer(x)
        assert y.shape == (8, 8, 8)

    def test_2d_output_shape_same_padding(self):
        """SAME padding preserves spatial dimensions."""
        layer = nn.ConvTranspose(3, 8, kernel_shape=(3, 3), padding="SAME", key=jax.random.key(0))
        x = jnp.ones((6, 6, 3))
        y = layer(x)
        assert y.shape == (6, 6, 8)

    def test_2d_output_shape_valid_padding(self):
        """VALID padding expands spatial dimensions."""
        layer = nn.ConvTranspose(3, 8, kernel_shape=(3, 3), padding="VALID", key=jax.random.key(0))
        x = jnp.ones((6, 6, 3))
        y = layer(x)
        assert y.shape == (8, 8, 8)

    def test_2d_output_shape_int_padding(self):
        """Integer padding applies symmetrically to both spatial dims."""
        layer = nn.ConvTranspose(3, 8, kernel_shape=(3, 3), padding=1, key=jax.random.key(0))
        x = jnp.ones((6, 6, 3))
        y = layer(x)
        assert y.shape == (6, 6, 8)

    def test_2d_output_shape_tuple_padding(self):
        """Tuple padding applies different values per spatial dim."""
        layer = nn.ConvTranspose(3, 8, kernel_shape=(3, 3), padding=(1, 0), key=jax.random.key(0))
        x = jnp.ones((6, 6, 3))
        y = layer(x)
        assert y.shape == (6, 8, 8)

    def test_2d_stride_upsamples(self):
        """Stride > 1 upsamples spatial dimensions."""
        layer = nn.ConvTranspose(
            3, 8, kernel_shape=(3, 3), stride=2, padding=1, key=jax.random.key(0)
        )
        x = jnp.ones((6, 6, 3))
        y = layer(x)
        assert y.shape == (11, 11, 8)

    def test_2d_output_padding(self):
        """output_padding adds extra elements to resolve stride ambiguity."""
        layer = nn.ConvTranspose(
            3, 8, kernel_shape=(3, 3), stride=2, padding=1, output_padding=1, key=jax.random.key(0)
        )
        x = jnp.ones((6, 6, 3))
        y = layer(x)
        assert y.shape == (12, 12, 8)

    def test_2d_dilation(self):
        """Dilation expands the effective kernel size."""
        layer = nn.ConvTranspose(3, 8, kernel_shape=(3, 3), dilation=2, key=jax.random.key(0))
        x = jnp.ones((6, 6, 3))
        y = layer(x)
        assert y.shape == (10, 10, 8)

    def test_2d_he_normal_init(self):
        """He normal initialization gives var(w) close to 2/fan_in."""
        layer = nn.ConvTranspose(64, 64, kernel_shape=(3, 3), key=jax.random.key(42))
        var = jnp.var(layer.w.value)
        fan_in = 3 * 3 * 64
        expected_var = 2.0 / fan_in
        npt.assert_allclose(var, expected_var, atol=0.05)

    def test_2d_zero_bias_init(self):
        """Bias is initialized to all zeros."""
        layer = nn.ConvTranspose(3, 8, kernel_shape=(3, 3), key=jax.random.key(0))
        assert jnp.all(layer.b == 0)

    def test_2d_weight_dtype(self):
        """Weights match the requested dtype."""
        layer = nn.ConvTranspose(3, 8, kernel_shape=(3, 3), dtype=jnp.float32, key=jax.random.key(0))
        assert layer.w.dtype == jnp.float32

    def test_2d_groups_output_shape(self):
        """Grouped transposed conv produces correct spatial shape."""
        layer = nn.ConvTranspose(
            8, 16, kernel_shape=(3, 3), groups=4, padding=1, key=jax.random.key(0)
        )
        x = jnp.ones((6, 6, 8))
        y = layer(x)
        assert y.shape == (6, 6, 16)

    def test_2d_groups_kernel_shape(self):
        """Kernel has in_channels // groups input dim."""
        layer = nn.ConvTranspose(8, 16, kernel_shape=(3, 3), groups=4, key=jax.random.key(0))
        assert layer.w.shape == (3, 3, 2, 16)

    def test_2d_groups_invalid(self):
        """Mismatched channels/groups raises ValueError."""
        with pytest.raises(ValueError, match="in_channels"):
            nn.ConvTranspose(7, 8, kernel_shape=(3, 3), groups=4, key=jax.random.key(0))
        with pytest.raises(ValueError, match="out_channels"):
            nn.ConvTranspose(8, 7, kernel_shape=(3, 3), groups=4, key=jax.random.key(0))

    def test_2d_output_padding_validation(self):
        """output_padding >= stride raises ValueError."""
        with pytest.raises(ValueError, match="output_padding"):
            nn.ConvTranspose(
                3, 8, kernel_shape=(3, 3), stride=2, output_padding=2, key=jax.random.key(0)
            )

    def test_3d_output_shape(self):
        """ConvTranspose with 3D kernel_shape works for 3D inputs."""
        layer = nn.ConvTranspose(3, 8, kernel_shape=(3, 3, 3), padding=1, key=jax.random.key(0))
        x = jnp.ones((4, 4, 4, 3))
        y = layer(x)
        assert y.shape == (4, 4, 4, 8)
