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
        npt.assert_allclose(layer.scale.value, jnp.ones(8))

    def test_bias_init(self):
        """Bias is initialized to all zeros."""
        layer = nn.LayerNorm(8)
        npt.assert_allclose(layer.b.value, jnp.zeros(8))


class TestGroupNorm:
    def test_zero_mean_per_group(self):
        """Output has approximately zero mean within each group."""
        layer = nn.GroupNorm(2, 8)
        x = jax.random.normal(jax.random.key(0), (4, 8))
        y = layer(x)
        y_groups = y.reshape(4, 2, 4)
        means = jnp.mean(y_groups, axis=-1)
        npt.assert_allclose(means, 0.0, atol=1e-5)

    def test_unit_variance_per_group(self):
        """Output has approximately unit variance within each group."""
        layer = nn.GroupNorm(2, 8)
        x = jax.random.normal(jax.random.key(0), (4, 8))
        y = layer(x)
        y_groups = y.reshape(4, 2, 4)
        var = jnp.mean(jnp.square(y_groups - jnp.mean(y_groups, axis=-1, keepdims=True)), axis=-1)
        npt.assert_allclose(var, 1.0, atol=1e-4)

    def test_scale_init(self):
        """Scale is initialized to all ones."""
        layer = nn.GroupNorm(2, 8)
        npt.assert_allclose(layer.scale.value, jnp.ones(8))

    def test_bias_init(self):
        """Bias is initialized to all zeros."""
        layer = nn.GroupNorm(2, 8)
        npt.assert_allclose(layer.b.value, jnp.zeros(8))

    def test_indivisible_dim_errors(self):
        """dim not divisible by num_groups raises ValueError."""
        with pytest.raises(ValueError, match="divisible"):
            nn.GroupNorm(3, 8)

    def test_single_group_matches_layer_norm(self):
        """With num_groups=1, GroupNorm behaves like LayerNorm."""
        x = jax.random.normal(jax.random.key(0), (4, 8))
        gn = nn.GroupNorm(1, 8)
        ln = nn.LayerNorm(8)
        npt.assert_allclose(gn(x), ln(x), atol=1e-6)


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
        npt.assert_allclose(layer.scale.value, jnp.ones(8))


class TestBatchNorm:
    def test_training_normalized_output(self):
        """Training output is normalized (zero mean, unit var) over the batch dimension."""
        layer = nn.BatchNorm(8, training=True)
        x = jax.random.normal(jax.random.key(0), (32, 8))
        y = layer(x)
        npt.assert_allclose(jnp.mean(y, axis=0), jnp.zeros(8), atol=1e-5)
        npt.assert_allclose(jnp.mean(jnp.square(y), axis=0), jnp.ones(8), atol=1e-4)

    def test_eval_uses_running_stats(self):
        """Eval mode uses running stats, not batch stats."""
        layer = nn.BatchNorm(8)
        x = jax.random.normal(jax.random.key(0), (32, 8)) * 5 + 3
        y_eval = layer(x)
        y_train = layer.replace(training=True)(x)
        assert not jnp.allclose(y_eval, y_train)

    def test_update_changes_running_stats(self):
        """update() returns a BatchNorm with updated running_mean and running_var."""
        layer = nn.BatchNorm(8, momentum=0.1)
        x = jax.random.normal(jax.random.key(0), (32, 8)) * 2 + 1
        updated = layer.update(x)
        assert not jnp.allclose(updated.running_mean, jnp.zeros(8))
        assert not jnp.allclose(updated.running_var, jnp.ones(8))

    def test_call_does_not_change_running_stats(self):
        """Calling __call__ does not modify running statistics."""
        layer = nn.BatchNorm(8, training=True)
        x = jax.random.normal(jax.random.key(0), (32, 8)) * 2 + 1
        layer(x)
        npt.assert_allclose(layer.running_mean, jnp.zeros(8))
        npt.assert_allclose(layer.running_var, jnp.ones(8))

    def test_scale_init(self):
        """Scale is initialized to all ones."""
        layer = nn.BatchNorm(8)
        npt.assert_allclose(layer.scale.value, jnp.ones(8))

    def test_bias_init(self):
        """Bias is initialized to all zeros."""
        layer = nn.BatchNorm(8)
        npt.assert_allclose(layer.b.value, jnp.zeros(8))  # type: ignore[union-attr]

    def test_no_bias(self):
        """bias=False produces None bias and still works."""
        layer = nn.BatchNorm(8, bias=False, training=True)
        assert layer.b is None
        x = jax.random.normal(jax.random.key(0), (32, 8))
        y = layer(x)
        assert y.shape == x.shape


class TestInstanceNorm:
    def test_zero_spatial_dims_raises(self):
        """num_spatial_dims=0 raises ValueError."""
        with pytest.raises(ValueError, match="num_spatial_dims"):
            nn.InstanceNorm(4, num_spatial_dims=0)

    def test_zero_mean_per_channel_1d(self):
        """Output has approximately zero mean over the temporal axis for each channel."""
        layer = nn.InstanceNorm(4)
        x = jax.random.normal(jax.random.key(0), (16, 4))
        y = layer(x)
        means = jnp.mean(y, axis=-2)
        npt.assert_allclose(means, 0.0, atol=1e-5)

    def test_unit_variance_per_channel_1d(self):
        """Output has approximately unit variance over the temporal axis for each channel."""
        layer = nn.InstanceNorm(4)
        x = jax.random.normal(jax.random.key(0), (16, 4))
        y = layer(x)
        var = jnp.mean(jnp.square(y - jnp.mean(y, axis=-2, keepdims=True)), axis=-2)
        npt.assert_allclose(var, 1.0, atol=1e-4)

    def test_zero_mean_per_channel_2d(self):
        """Output has approximately zero mean over spatial axes for each channel."""
        layer = nn.InstanceNorm(3, num_spatial_dims=2)
        x = jax.random.normal(jax.random.key(0), (8, 8, 3))
        y = layer(x)
        means = jnp.mean(y, axis=(-3, -2))
        npt.assert_allclose(means, 0.0, atol=1e-5)

    def test_unit_variance_per_channel_2d(self):
        """Output has approximately unit variance over spatial axes for each channel."""
        layer = nn.InstanceNorm(3, num_spatial_dims=2)
        x = jax.random.normal(jax.random.key(0), (8, 8, 3))
        y = layer(x)
        var = jnp.mean(jnp.square(y - jnp.mean(y, axis=(-3, -2), keepdims=True)), axis=(-3, -2))
        npt.assert_allclose(var, 1.0, atol=1e-4)

    def test_batch_dims(self):
        """Arbitrary batch dimensions are preserved."""
        layer = nn.InstanceNorm(4)
        x = jax.random.normal(jax.random.key(0), (2, 3, 16, 4))
        y = layer(x)
        assert y.shape == (2, 3, 16, 4)

    def test_batch_dims_2d(self):
        """Arbitrary batch dimensions are preserved for 2d."""
        layer = nn.InstanceNorm(3, num_spatial_dims=2)
        x = jax.random.normal(jax.random.key(0), (2, 5, 8, 8, 3))
        y = layer(x)
        assert y.shape == (2, 5, 8, 8, 3)

    def test_scale_init(self):
        """Scale is initialized to all ones."""
        layer = nn.InstanceNorm(4)
        npt.assert_allclose(layer.scale.value, jnp.ones(4))

    def test_bias_init(self):
        """Bias is initialized to all zeros."""
        layer = nn.InstanceNorm(4)
        npt.assert_allclose(layer.b.value, jnp.zeros(4))
