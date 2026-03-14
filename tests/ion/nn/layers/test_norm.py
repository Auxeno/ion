import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from ion import nn


class TestLayerNorm:
    def test_zero_mean(self):
        """Output has approximately zero mean along the last axis."""
        layer = nn.LayerNorm(8)
        x = jax.random.normal(jax.random.key(0), (4, 8))
        y = layer(x)
        means = jnp.mean(y, axis=-1)
        npt.assert_allclose(means, 0.0, atol=1e-5)

    def test_unit_variance(self):
        """Output has approximately unit variance along the last axis."""
        layer = nn.LayerNorm(8)
        x = jax.random.normal(jax.random.key(0), (4, 8))
        y = layer(x)
        var = jnp.mean(jnp.square(y - jnp.mean(y, axis=-1, keepdims=True)), axis=-1)
        npt.assert_allclose(var, 1.0, atol=1e-4)

    def test_scale_init(self):
        """Scale is initialized to all ones."""
        layer = nn.LayerNorm(8)
        npt.assert_allclose(layer.scale._value, jnp.ones(8))

    def test_bias_init(self):
        """Bias is initialized to all zeros."""
        layer = nn.LayerNorm(8)
        npt.assert_allclose(layer.b._value, jnp.zeros(8))


class TestGroupNorm:
    def test_zero_mean_per_group(self):
        """Output has approximately zero mean within each group."""
        layer = nn.GroupNorm(8, 2)
        x = jax.random.normal(jax.random.key(0), (4, 8))
        y = layer(x)
        y_groups = y.reshape(4, 2, 4)
        means = jnp.mean(y_groups, axis=-1)
        npt.assert_allclose(means, 0.0, atol=1e-5)

    def test_unit_variance_per_group(self):
        """Output has approximately unit variance within each group."""
        layer = nn.GroupNorm(8, 2)
        x = jax.random.normal(jax.random.key(0), (4, 8))
        y = layer(x)
        y_groups = y.reshape(4, 2, 4)
        var = jnp.mean(jnp.square(y_groups - jnp.mean(y_groups, axis=-1, keepdims=True)), axis=-1)
        npt.assert_allclose(var, 1.0, atol=1e-4)

    def test_scale_init(self):
        """Scale is initialized to all ones."""
        layer = nn.GroupNorm(8, 2)
        npt.assert_allclose(layer.scale._value, jnp.ones(8))

    def test_bias_init(self):
        """Bias is initialized to all zeros."""
        layer = nn.GroupNorm(8, 2)
        npt.assert_allclose(layer.b._value, jnp.zeros(8))

    def test_indivisible_dim_errors(self):
        """dim not divisible by num_groups raises ValueError."""
        with pytest.raises(ValueError, match="divisible"):
            nn.GroupNorm(8, 3)

    def test_single_group_matches_layer_norm(self):
        """With num_groups=1, GroupNorm behaves like LayerNorm."""
        x = jax.random.normal(jax.random.key(0), (4, 8))
        gn = nn.GroupNorm(8, 1)
        ln = nn.LayerNorm(8)
        npt.assert_allclose(gn(x), ln(x), atol=1e-6)

    def test_spatial_zero_mean_per_group(self):
        """With num_spatial_dims=2, output has zero mean over spatial + group channels."""
        layer = nn.GroupNorm(8, 2, num_spatial_dims=2)
        x = jax.random.normal(jax.random.key(0), (1, 6, 6, 8))
        y = layer(x)
        y_groups = y.reshape(1, 6, 6, 2, 4)
        means = jnp.mean(y_groups, axis=(1, 2, 4))
        npt.assert_allclose(means, 0.0, atol=1e-5)

    def test_spatial_unit_variance_per_group(self):
        """With num_spatial_dims=2, output has unit variance over spatial + group channels."""
        layer = nn.GroupNorm(8, 2, num_spatial_dims=2)
        x = jax.random.normal(jax.random.key(0), (1, 6, 6, 8))
        y = layer(x)
        y_groups = y.reshape(1, 6, 6, 2, 4)
        mean = jnp.mean(y_groups, axis=(1, 2, 4), keepdims=True)
        var = jnp.mean(jnp.square(y_groups - mean), axis=(1, 2, 4))
        npt.assert_allclose(var, 1.0, atol=1e-4)

    def test_spatial_vmap_batch(self):
        """jax.vmap adds an extra batch dimension with num_spatial_dims."""
        layer = nn.GroupNorm(8, 2, num_spatial_dims=2)
        x = jax.random.normal(jax.random.key(0), (3, 6, 6, 8))
        y = layer(x)
        assert y.shape == (3, 6, 6, 8)

        x_extra = jnp.stack([x] * 2)
        y_extra = jax.vmap(layer)(x_extra)
        assert y_extra.shape == (2, 3, 6, 6, 8)

    def test_instance_norm_via_group_norm(self):
        """GroupNorm with num_groups=dim and num_spatial_dims gives instance norm."""
        layer = nn.GroupNorm(3, 3, num_spatial_dims=2)
        x = jax.random.normal(jax.random.key(0), (1, 8, 8, 3))
        y = layer(x)
        means = jnp.mean(y, axis=(1, 2))
        npt.assert_allclose(means, 0.0, atol=1e-5)


class TestRMSNorm:
    def test_unit_rms(self):
        """Output has approximately unit RMS along the last axis."""
        layer = nn.RMSNorm(8)
        x = jax.random.normal(jax.random.key(0), (4, 8))
        y = layer(x)
        rms = jnp.sqrt(jnp.mean(jnp.square(y), axis=-1))
        npt.assert_allclose(rms, 1.0, atol=1e-4)

    def test_scale_init(self):
        """Scale is initialized to all ones."""
        layer = nn.RMSNorm(8)
        npt.assert_allclose(layer.scale._value, jnp.ones(8))


