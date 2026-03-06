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
        layer = nn.BatchNorm(8)
        x = jax.random.normal(jax.random.key(0), (32, 8))
        y, _ = layer(x, layer.initial_state, training=True)
        npt.assert_allclose(jnp.mean(y, axis=0), jnp.zeros(8), atol=1e-5)
        npt.assert_allclose(jnp.mean(jnp.square(y), axis=0), jnp.ones(8), atol=1e-4)

    def test_eval_uses_running_stats(self):
        """Eval mode uses running stats, not batch stats."""
        layer = nn.BatchNorm(8)
        x = jax.random.normal(jax.random.key(0), (32, 8)) * 5 + 3
        # In eval with default running_mean=0, running_var=1, output should differ from training
        y_eval, _ = layer(x, layer.initial_state, training=False)
        y_train, _ = layer(x, layer.initial_state, training=True)
        assert not jnp.allclose(y_eval, y_train)

    def test_running_stats_update(self):
        """Returned state has updated running_mean and running_var."""
        layer = nn.BatchNorm(8, momentum=0.1)
        x = jax.random.normal(jax.random.key(0), (32, 8)) * 2 + 1
        _, state = layer(x, layer.initial_state, training=True)
        running_mean, running_var = state
        # Running stats should have moved from their initial values
        assert not jnp.allclose(running_mean, jnp.zeros(8))
        assert not jnp.allclose(running_var, jnp.ones(8))

    def test_running_stats_unchanged_in_eval(self):
        """Running stats are not modified in eval mode."""
        layer = nn.BatchNorm(8)
        x = jax.random.normal(jax.random.key(0), (32, 8)) * 2 + 1
        _, state = layer(x, layer.initial_state, training=False)
        running_mean, running_var = state
        npt.assert_allclose(running_mean, jnp.zeros(8))
        npt.assert_allclose(running_var, jnp.ones(8))

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
        layer = nn.BatchNorm(8, bias=False)
        assert layer.b is None
        x = jax.random.normal(jax.random.key(0), (32, 8))
        y, _ = layer(x, layer.initial_state, training=True)
        assert y.shape == x.shape

    def test_return_type_both_modes(self):
        """Both modes return a (output, state_tuple) tuple."""
        layer = nn.BatchNorm(8)
        x = jax.random.normal(jax.random.key(0), (32, 8))
        result_train = layer(x, layer.initial_state, training=True)
        result_eval = layer(x, layer.initial_state, training=False)
        assert isinstance(result_train, tuple) and len(result_train) == 2
        assert isinstance(result_eval, tuple) and len(result_eval) == 2
        assert isinstance(result_train[1], tuple) and len(result_train[1]) == 2
        assert isinstance(result_eval[1], tuple) and len(result_eval[1]) == 2

    def test_no_state_errors(self):
        """Calling without state when self.state is None raises ValueError."""
        layer = nn.BatchNorm(8)
        x = jax.random.normal(jax.random.key(0), (32, 8))
        with pytest.raises(ValueError, match="No state provided"):
            layer(x)

    def test_fallback_to_self_state(self):
        """After replace(state=...), calling without explicit state works."""
        layer = nn.BatchNorm(8)
        x = jax.random.normal(jax.random.key(0), (32, 8))
        # Train to get updated state
        _, state = layer(x, layer.initial_state, training=True)
        # Store state on module
        layer = layer.replace(state=state)
        # Call without explicit state, should use self.state
        y, returned_state = layer(x)
        assert y.shape == x.shape
        npt.assert_allclose(returned_state[0], state[0])
        npt.assert_allclose(returned_state[1], state[1])


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
