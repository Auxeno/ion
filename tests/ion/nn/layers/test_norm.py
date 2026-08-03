import dataclasses

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from ion import nn


class TestBatchNorm:
    def test_running_stats_init(self):
        """Running mean initializes to zero and variance to one."""
        layer = nn.BatchNorm(3)
        running_mean, running_var = layer.init_buffers()[layer]
        npt.assert_array_equal(running_mean, jnp.zeros(3, dtype=jnp.float32))
        npt.assert_array_equal(running_var, jnp.ones(3, dtype=jnp.float32))

    def test_training_normalizes_batch(self):
        """Training output has approximately zero mean and unit variance."""
        layer = nn.BatchNorm(3, eps=1e-8)
        buffers = layer.init_buffers()
        x = jax.random.normal(jax.random.key(0), (32, 4, 3))
        y, _ = layer(x, buffers, training=True)
        npt.assert_allclose(jnp.mean(y, axis=(0, 1)), 0.0, atol=1e-6)
        npt.assert_allclose(jnp.var(y, axis=(0, 1)), 1.0, atol=1e-5)

    def test_running_stats_update(self):
        """Training applies the configured momentum to running statistics."""
        layer = nn.BatchNorm(2, momentum=0.25)
        buffers = layer.init_buffers()
        x = jnp.array([[1.0, 2.0], [5.0, 8.0]])
        _, updated = layer(x, buffers, training=True)
        running_mean, running_var = updated[layer]
        batch_mean = jnp.mean(x, axis=0)
        batch_var = jnp.var(x, axis=0)
        npt.assert_allclose(running_mean, 0.25 * batch_mean)
        npt.assert_allclose(running_var, 0.75 + 0.25 * batch_var)

    def test_evaluation_uses_running_stats(self):
        """Evaluation normalizes with stored running statistics."""
        layer = nn.BatchNorm(2, eps=1e-5)
        buffers = layer.init_buffers()
        buffers = buffers.set(
            layer,
            (jnp.array([1.0, 2.0]), jnp.array([4.0, 9.0])),
        )
        x = jnp.array([[3.0, 5.0]])
        y, _ = layer(x, buffers, training=False)
        expected = (x - jnp.array([1.0, 2.0])) / jnp.sqrt(jnp.array([4.0, 9.0]) + layer.eps)
        npt.assert_allclose(y, expected)

    def test_evaluation_preserves_buffers(self):
        """Evaluation returns the original buffer collection unchanged."""
        layer = nn.BatchNorm(2)
        buffers = layer.init_buffers()
        _, returned = layer(jnp.ones((2, 2)), buffers, training=False)
        assert returned is buffers

    def test_affine(self):
        """Learned scale and bias are applied after normalization."""
        layer = nn.BatchNorm(2)
        layer = layer.at.scale.set(nn.Param(jnp.array([2.0, 3.0])))
        layer = layer.at.b.set(nn.Param(jnp.array([4.0, 5.0])))
        buffers = layer.init_buffers()
        x = jnp.array([[1.0, -2.0], [3.0, 4.0]])
        normalized = (x - jnp.mean(x, axis=0)) / jnp.sqrt(jnp.var(x, axis=0) + layer.eps)
        y, _ = layer(x, buffers, training=True)
        npt.assert_allclose(
            y,
            normalized * jnp.array([2.0, 3.0]) + jnp.array([4.0, 5.0]),
            rtol=1e-6,
        )

    def test_no_bias(self):
        """bias=False drops the bias parameter."""
        layer = nn.BatchNorm(4, bias=False)
        assert layer.b is None
        assert layer.num_params == 4

    @pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
    def test_mixed_precision(self, dtype):
        """Low-precision inputs retain their dtype and float32 running statistics."""
        layer = nn.BatchNorm(2).astype(dtype)
        buffers = layer.init_buffers()
        x = jnp.array([[1.0, 3.0], [2.0, 7.0]], dtype=dtype)
        y, buffers = layer(x, buffers, training=True)
        running_mean, running_var = buffers[layer]
        assert y.dtype == dtype
        assert running_mean.dtype == jnp.float32
        assert running_var.dtype == jnp.float32
        assert jnp.all(jnp.isfinite(y))

    def test_input_grad(self):
        """Input gradients are finite."""
        layer = nn.BatchNorm(3)
        buffers = layer.init_buffers()
        x = jax.random.normal(jax.random.key(0), (4, 3))
        input_grad = jax.grad(
            lambda value: jnp.square(layer(value, buffers, training=True)[0]).sum()
        )(x)

        assert jnp.all(jnp.isfinite(input_grad))

    def test_parameter_grad(self):
        """Scale and bias gradients are finite."""
        layer = nn.BatchNorm(3)
        buffers = layer.init_buffers()
        x = jax.random.normal(jax.random.key(0), (4, 3))
        model_grad = jax.grad(lambda model: jnp.square(model(x, buffers, training=True)[0]).sum())(
            layer
        )

        assert jnp.all(jnp.isfinite(model_grad.scale._value))
        assert model_grad.b is not None
        assert jnp.all(jnp.isfinite(model_grad.b._value))

    def test_jit_training(self):
        """Training and buffer updates work under jax.jit."""
        layer = nn.BatchNorm(3)
        buffers = layer.init_buffers()
        x = jax.random.normal(jax.random.key(0), (4, 3))
        train = jax.jit(lambda x, b: layer(x, b, training=True))
        y, updated = train(x, buffers)

        assert y.shape == x.shape
        assert updated is not buffers

    def test_jit_evaluation(self):
        """Evaluation works under jax.jit without changing buffers."""
        layer = nn.BatchNorm(3)
        buffers = layer.init_buffers()
        x = jax.random.normal(jax.random.key(0), (4, 3))
        evaluate = jax.jit(lambda x, b: layer(x, b, training=False))
        y, returned = evaluate(x, buffers)

        assert y.shape == x.shape
        for actual, expected in zip(jax.tree.leaves(returned), jax.tree.leaves(buffers)):
            npt.assert_array_equal(actual, expected)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"dim": -1},
            {"dim": 2, "momentum": -0.1},
            {"dim": 2, "momentum": 1.1},
            {"dim": 2, "eps": 0.0},
        ],
    )
    def test_invalid_hyperparameters(self, kwargs):
        """Invalid dimensions, momentum, and epsilon raise an error."""
        with pytest.raises((TypeError, ValueError)):
            nn.BatchNorm(**kwargs)

    def test_requires_reduction_dimension(self):
        """Input requires at least one dimension before features."""
        layer = nn.BatchNorm(3)
        buffers = layer.init_buffers()
        with pytest.raises(ValueError, match="reduction dimension"):
            layer(jnp.ones(3), buffers, training=True)

    def test_wrong_feature_dim_raises(self):
        """An input feature dimension that does not match dim raises an error."""
        layer = nn.BatchNorm(3)
        buffers = layer.init_buffers()
        with pytest.raises((TypeError, ValueError)):
            layer(jnp.ones((2, 4)), buffers, training=True)


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
        assert layer.b is not None
        npt.assert_allclose(layer.b._value, jnp.zeros(8))

    def test_no_bias(self):
        """bias=False drops the bias parameter."""
        layer = nn.LayerNorm(8, bias=False)
        assert layer.b is None
        assert layer.num_params == 8

    def test_no_bias_matches_zero_bias(self):
        """A zero bias is equivalent to no bias."""
        x = jax.random.normal(jax.random.key(0), (4, 8))
        npt.assert_allclose(nn.LayerNorm(8, bias=False)(x), nn.LayerNorm(8)(x), rtol=1e-6)

    def test_no_bias_is_not_rms_norm(self):
        """Bias-less LayerNorm still centers the mean, unlike RMSNorm."""
        x = jax.random.normal(jax.random.key(0), (4, 8)) + 5.0
        assert not jnp.allclose(nn.LayerNorm(8, bias=False)(x), nn.RMSNorm(8)(x))


class TestGroupNorm:
    def test_zero_mean_per_group(self):
        """Output has approximately zero mean within each group."""
        layer = nn.GroupNorm(8, 2, num_spatial_dims=0)
        x = jax.random.normal(jax.random.key(0), (4, 8))
        y = layer(x)
        y_groups = y.reshape(4, 2, 4)
        means = jnp.mean(y_groups, axis=-1)
        npt.assert_allclose(means, 0.0, atol=1e-5)

    def test_unit_variance_per_group(self):
        """Output has approximately unit variance within each group."""
        layer = nn.GroupNorm(8, 2, num_spatial_dims=0)
        x = jax.random.normal(jax.random.key(0), (4, 8))
        y = layer(x)
        y_groups = y.reshape(4, 2, 4)
        var = jnp.mean(jnp.square(y_groups - jnp.mean(y_groups, axis=-1, keepdims=True)), axis=-1)
        npt.assert_allclose(var, 1.0, atol=1e-4)

    def test_scale_init(self):
        """Scale is initialized to all ones."""
        layer = nn.GroupNorm(8, 2, num_spatial_dims=0)
        npt.assert_allclose(layer.scale._value, jnp.ones(8))

    def test_bias_init(self):
        """Bias is initialized to all zeros."""
        layer = nn.GroupNorm(8, 2, num_spatial_dims=0)
        npt.assert_allclose(layer.b._value, jnp.zeros(8))

    def test_indivisible_dim_errors(self):
        """dim not divisible by num_groups raises ValueError."""
        with pytest.raises(ValueError, match="divisible"):
            nn.GroupNorm(8, 3, num_spatial_dims=0)

    def test_single_group_matches_layer_norm(self):
        """With num_groups=1, GroupNorm behaves like LayerNorm."""
        x = jax.random.normal(jax.random.key(0), (4, 8))
        gn = nn.GroupNorm(8, 1, num_spatial_dims=0)
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

    def test_spatial_unbatched_matches_batched(self):
        """A spatial input without a batch dim matches the batched result."""
        layer = nn.GroupNorm(8, 2, num_spatial_dims=2)
        x = jax.random.normal(jax.random.key(0), (6, 6, 8))
        npt.assert_allclose(layer(x), layer(x[None])[0], atol=1e-6)

    def test_extra_leading_dims(self):
        """Arbitrary leading dims are normalized independently."""
        layer = nn.GroupNorm(8, 2, num_spatial_dims=2)
        x = jax.random.normal(jax.random.key(0), (2, 3, 6, 6, 8))
        y = layer(x)
        assert y.shape == x.shape
        npt.assert_allclose(y[1, 2], layer(x[1, 2]), atol=1e-6)

        layer = nn.GroupNorm(8, 2, num_spatial_dims=0)
        x = jax.random.normal(jax.random.key(1), (2, 5, 8))
        npt.assert_allclose(layer(x)[0], layer(x[0]), atol=1e-6)

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


class TestSpectralNorm:
    def test_buffer_init_reproducible(self):
        """The same initialization key produces identical buffers."""
        layer = nn.SpectralNorm(nn.Linear(4, 5, key=jax.random.key(0)))
        first = layer.init_buffers(key=jax.random.key(1))
        second = layer.init_buffers(key=jax.random.key(1))
        for a, b in zip(first[layer], second[layer]):
            npt.assert_array_equal(a, b)

    def test_buffer_init_requires_key(self):
        """Buffer initialization without a key raises ValueError."""
        layer = nn.SpectralNorm(nn.Linear(4, 5, key=jax.random.key(0)))
        with pytest.raises(ValueError, match=r"init_buffers\(key=key\)"):
            layer.init_buffers()

    def test_spectral_norm(self):
        """The normalized weight has largest singular value near one."""
        linear = nn.Linear(8, 8, bias=False, key=jax.random.key(0))
        layer = nn.SpectralNorm(linear, power_iterations=20)
        buffers = layer.init_buffers(key=jax.random.key(1))
        u, v = buffers[layer]
        weight = jnp.asarray(linear.w)
        matrix = weight.reshape(-1, weight.shape[-1]).T
        sigma_estimate = u @ matrix @ v
        normalized = matrix / sigma_estimate
        sigma = jnp.linalg.svd(normalized, compute_uv=False)[0]
        npt.assert_allclose(sigma, 1.0, rtol=1e-4, atol=1e-4)

    def test_training_updates_buffers(self):
        """Training updates the power-iteration vectors."""
        layer = nn.SpectralNorm(nn.Linear(4, 5, key=jax.random.key(0)))
        buffers = layer.init_buffers(key=jax.random.key(1))
        x = jnp.ones((2, 4))
        _, trained = layer(x, buffers, training=True)
        assert any(not jnp.array_equal(a, b) for a, b in zip(trained[layer], buffers[layer]))

    def test_evaluation_preserves_buffers(self):
        """Evaluation returns the original buffer collection unchanged."""
        layer = nn.SpectralNorm(nn.Linear(4, 5, key=jax.random.key(0)))
        buffers = layer.init_buffers(key=jax.random.key(1))
        _, evaluated = layer(jnp.ones((2, 4)), buffers, training=False)
        assert evaluated is buffers

    def test_evaluation_stable(self):
        """Repeated evaluation with fixed buffers produces identical output."""
        layer = nn.SpectralNorm(nn.Linear(4, 5, key=jax.random.key(0)))
        buffers = layer.init_buffers(key=jax.random.key(1))
        x = jax.random.normal(jax.random.key(2), (3, 4))
        y1, b1 = layer(x, buffers, training=False)
        y2, b2 = layer(x, buffers, training=False)
        npt.assert_array_equal(y1, y2)
        assert b1 is b2 is buffers

    def test_input_grad(self):
        """Input gradients are finite."""
        layer = nn.SpectralNorm(nn.Linear(4, 5, key=jax.random.key(0)))
        buffers = layer.init_buffers(key=jax.random.key(1))
        x = jax.random.normal(jax.random.key(2), (3, 4))
        input_grad = jax.grad(lambda value: layer(value, buffers, training=True)[0].sum())(x)

        assert jnp.all(jnp.isfinite(input_grad))

    def test_parameter_grad(self):
        """Wrapped parameter gradients are finite."""
        layer = nn.SpectralNorm(nn.Linear(4, 5, key=jax.random.key(0)))
        buffers = layer.init_buffers(key=jax.random.key(1))
        x = jax.random.normal(jax.random.key(2), (3, 4))
        model_grad = jax.grad(lambda model: model(x, buffers, training=True)[0].sum())(layer)

        for leaf in jax.tree.leaves(model_grad):
            assert jnp.all(jnp.isfinite(leaf))

    def test_linear(self):
        """Linear weights can be spectrally normalized."""
        linear = nn.SpectralNorm(nn.Linear(4, 5, key=jax.random.key(0)))
        linear_buffers = linear.init_buffers(key=jax.random.key(1))
        y, _ = linear(jnp.ones((2, 4)), linear_buffers, training=True)
        assert y.shape == (2, 5)

    def test_conv(self):
        """Convolutional weights flatten across non-output dimensions."""
        conv = nn.SpectralNorm(nn.Conv(3, 6, kernel_shape=(3, 3), padding=1, key=jax.random.key(2)))
        conv_buffers = conv.init_buffers(key=jax.random.key(3))
        y, _ = conv(jnp.ones((2, 8, 8, 3)), conv_buffers, training=True)
        assert y.shape == (2, 8, 8, 6)

    def test_invalid_parameter_name(self):
        """A missing parameter name raises AttributeError."""
        with pytest.raises(AttributeError):
            nn.SpectralNorm(nn.Linear(4, 5, key=jax.random.key(0)), parameter="missing")

    def test_invalid_parameter_type(self):
        """A plain-array parameter raises TypeError."""

        class NotParam(nn.Module):
            w: jax.Array

            def __init__(self):
                self.w = jnp.ones((2, 2))

        with pytest.raises(TypeError, match="must be a Param"):
            nn.SpectralNorm(NotParam())

    def test_invalid_parameter_shape(self):
        """A parameter with fewer than two dimensions raises ValueError."""

        class Vector(nn.Module):
            w: nn.Param

            def __init__(self):
                self.w = nn.Param(jnp.ones(2))

        with pytest.raises(ValueError, match="at least 2D"):
            nn.SpectralNorm(Vector())

    def test_invalid_power_iterations(self):
        """power_iterations below one raises ValueError."""
        linear = nn.Linear(4, 5, key=jax.random.key(0))
        with pytest.raises(ValueError, match="power_iterations"):
            nn.SpectralNorm(linear, power_iterations=0)

    def test_invalid_eps(self):
        """A non-positive epsilon raises ValueError."""
        linear = nn.Linear(4, 5, key=jax.random.key(0))
        with pytest.raises(ValueError, match="eps"):
            nn.SpectralNorm(linear, eps=0)

    def test_nested_buffer_module_raises(self):
        """Wrapping another BufferModule raises ValueError during initialization."""

        class BufferedMatrix(nn.BufferModule):
            w: nn.Param

            def __init__(self):
                self.w = nn.Param(jnp.ones((2, 2)))

            def _init_buffer(self, *, key=None):
                return jnp.zeros(1)

        layer = nn.SpectralNorm(BufferedMatrix())
        with pytest.raises(ValueError, match="cannot contain"):
            layer.init_buffers(key=jax.random.key(0))

    def test_wrong_buffer_shape_raises(self):
        """Power vectors with stale shapes raise an error."""
        layer = nn.SpectralNorm(nn.Linear(4, 5, key=jax.random.key(0)))
        buffers = layer.init_buffers(key=jax.random.key(1))
        malformed = dataclasses.replace(buffers, _values=((jnp.ones(4), jnp.ones(4)),))
        with pytest.raises((TypeError, ValueError)):
            layer(jnp.ones((2, 4)), malformed, training=True)

    def test_jit_training(self):
        """Training works under jax.jit."""
        layer = nn.SpectralNorm(nn.Linear(4, 5, key=jax.random.key(0)))
        buffers = layer.init_buffers(key=jax.random.key(1))
        x = jax.random.normal(jax.random.key(2), (3, 4))
        train = jax.jit(lambda x, b: layer(x, b, training=True))
        y, _ = train(x, buffers)

        assert y.shape == (3, 5)

    def test_jit_evaluation(self):
        """Evaluation works under jax.jit without changing buffers."""
        layer = nn.SpectralNorm(nn.Linear(4, 5, key=jax.random.key(0)))
        buffers = layer.init_buffers(key=jax.random.key(1))
        x = jax.random.normal(jax.random.key(2), (3, 4))
        evaluate = jax.jit(lambda x, b: layer(x, b, training=False))
        y, returned = evaluate(x, buffers)

        assert y.shape == (3, 5)
        for actual, expected in zip(jax.tree.leaves(returned), jax.tree.leaves(buffers)):
            npt.assert_array_equal(actual, expected)
