import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from ion import nn


class TestBatchNorm:
    def test_running_stats_init(self):
        """Running mean initializes to zero and variance to one."""
        layer = nn.BatchNorm(3)
        npt.assert_array_equal(layer.running_mean.value, jnp.zeros(3, dtype=jnp.float32))
        npt.assert_array_equal(layer.running_var.value, jnp.ones(3, dtype=jnp.float32))

    def test_training_normalizes_batch(self):
        """Training output has approximately zero mean and unit variance."""
        layer = nn.BatchNorm(3, eps=1e-8)
        x = jax.random.normal(jax.random.key(0), (32, 4, 3))
        y = layer(x, training=True)
        npt.assert_allclose(jnp.mean(y, axis=(0, 1)), 0.0, atol=1e-6)
        npt.assert_allclose(jnp.var(y, axis=(0, 1)), 1.0, atol=1e-5)

    def test_running_stats_update(self):
        """Running variance uses the Bessel-corrected batch estimate."""
        layer = nn.BatchNorm(2, momentum=0.25)
        x = jnp.array([[1.0, 2.0], [5.0, 8.0]])
        layer(x, training=True)
        batch_mean = jnp.mean(x, axis=0)
        batch_var = jnp.var(x, axis=0, ddof=1)
        npt.assert_allclose(layer.running_mean.value, 0.25 * batch_mean)
        npt.assert_allclose(layer.running_var.value, 0.75 + 0.25 * batch_var)

    def test_single_sample_keeps_running_stats_finite(self):
        """A batch of one has no correction to apply, so the running variance stays finite."""
        layer = nn.BatchNorm(2, momentum=0.25)
        x = jnp.array([[1.0, 2.0]])

        layer(x, training=True)

        npt.assert_allclose(layer.running_mean.value, 0.25 * x[0])
        npt.assert_allclose(layer.running_var.value, jnp.array([0.75, 0.75]))

    def test_running_variance_counts_all_reduction_axes(self):
        """The variance correction includes batch and spatial positions."""
        layer = nn.BatchNorm(2, momentum=1.0)
        x = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 2, 2)

        layer(x, training=True)

        npt.assert_allclose(layer.running_var.value, jnp.var(x, axis=(0, 1, 2), ddof=1))

    def test_evaluation_uses_running_stats(self):
        """Evaluation normalizes with stored running statistics."""
        layer = nn.BatchNorm(2, eps=1e-5)
        layer.running_mean.set(jnp.array([1.0, 2.0]))
        layer.running_var.set(jnp.array([4.0, 9.0]))
        x = jnp.array([[3.0, 5.0]])
        y = layer(x, training=False)
        expected = (x - jnp.array([1.0, 2.0])) / jnp.sqrt(jnp.array([4.0, 9.0]) + layer.eps)
        npt.assert_allclose(y, expected)

    def test_evaluation_preserves_buffers(self):
        """Evaluation leaves the running statistics unchanged."""
        layer = nn.BatchNorm(2)
        layer(jnp.ones((2, 2)), training=False)
        npt.assert_array_equal(layer.running_mean.value, jnp.zeros(2))
        npt.assert_array_equal(layer.running_var.value, jnp.ones(2))

    def test_affine(self):
        """Learned scale and bias are applied after normalization."""
        layer = nn.BatchNorm(2)
        layer = layer.at.scale.set(nn.Param(jnp.array([2.0, 3.0])))
        layer = layer.at.b.set(nn.Param(jnp.array([4.0, 5.0])))
        x = jnp.array([[1.0, -2.0], [3.0, 4.0]])
        normalized = (x - jnp.mean(x, axis=0)) / jnp.sqrt(jnp.var(x, axis=0) + layer.eps)
        y = layer(x, training=True)
        npt.assert_allclose(
            y,
            normalized * jnp.array([2.0, 3.0]) + jnp.array([4.0, 5.0]),
            rtol=1e-6,
        )

    def test_no_bias(self):
        """use_bias=False drops the bias parameter."""
        layer = nn.BatchNorm(4, use_bias=False)
        assert layer.b is None
        assert layer.num_params == 4

    @pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
    def test_mixed_precision(self, dtype):
        """Statistics use float32 without promoting the output."""
        layer = nn.BatchNorm(2).astype(dtype)
        x = (100 * jax.random.normal(jax.random.key(0), (2, 4096, 2))).astype(dtype)
        y = layer(x, training=True)
        expected_layer = nn.BatchNorm(2)
        expected = expected_layer(x.astype(jnp.float32), training=True).astype(dtype)

        assert y.dtype == dtype
        assert layer.running_mean.value.dtype == jnp.float32
        assert layer.running_var.value.dtype == jnp.float32
        npt.assert_allclose(y, expected, rtol=1e-2, atol=1e-2)
        npt.assert_allclose(layer.running_mean.value, expected_layer.running_mean.value)
        npt.assert_allclose(layer.running_var.value, expected_layer.running_var.value)

    @pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
    @pytest.mark.parametrize("training", [False, True])
    def test_preserves_input_dtype(self, dtype, training):
        """The output follows the input dtype even with default float32 params."""
        layer = nn.BatchNorm(2)
        x = jnp.ones((3, 2), dtype=dtype)

        assert layer(x, training=training).dtype == dtype

    def test_input_grad(self):
        """Input gradients are finite."""
        layer = nn.BatchNorm(3)
        x = jax.random.normal(jax.random.key(0), (4, 3))
        input_grad = jax.grad(lambda value: jnp.square(layer(value, training=True)).sum())(x)

        assert jnp.all(jnp.isfinite(input_grad))

    def test_parameter_grad(self):
        """Scale and bias gradients are finite."""
        layer = nn.BatchNorm(3)
        x = jax.random.normal(jax.random.key(0), (4, 3))
        model_grad = jax.grad(lambda model: jnp.square(model(x, training=True)).sum())(layer)

        assert jnp.all(jnp.isfinite(model_grad.scale._value))
        assert model_grad.b is not None
        assert jnp.all(jnp.isfinite(model_grad.b._value))

    def test_jit_training(self):
        """Training and running statistic updates work under jax.jit."""
        layer = nn.BatchNorm(3)
        x = jax.random.normal(jax.random.key(0), (4, 3))
        train = jax.jit(lambda x: layer(x, training=True))
        y = train(x)

        assert y.shape == x.shape
        assert not jnp.array_equal(layer.running_mean.value, jnp.zeros(3))

    def test_jit_evaluation(self):
        """Evaluation works under jax.jit without changing running statistics."""
        layer = nn.BatchNorm(3)
        x = jax.random.normal(jax.random.key(0), (4, 3))
        evaluate = jax.jit(lambda x: layer(x, training=False))
        y = evaluate(x)

        assert y.shape == x.shape
        npt.assert_array_equal(layer.running_mean.value, jnp.zeros(3))

    def test_requires_reduction_dimension(self):
        """Input requires at least one dimension before features."""
        layer = nn.BatchNorm(3)
        with pytest.raises(ValueError, match="reduction dimension"):
            layer(jnp.ones(3), training=True)

    def test_wrong_feature_dim_raises(self):
        """An input feature dimension that does not match dim raises an error."""
        layer = nn.BatchNorm(3)
        with pytest.raises((TypeError, ValueError)):
            layer(jnp.ones((2, 4)), training=True)


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
        """use_bias=False drops the bias parameter."""
        layer = nn.LayerNorm(8, use_bias=False)
        assert layer.b is None
        assert layer.num_params == 8

    def test_no_bias_matches_zero_bias(self):
        """A zero bias is equivalent to no bias."""
        x = jax.random.normal(jax.random.key(0), (4, 8))
        npt.assert_allclose(nn.LayerNorm(8, use_bias=False)(x), nn.LayerNorm(8)(x), rtol=1e-6)

    def test_no_bias_is_not_rms_norm(self):
        """Bias-less LayerNorm still centers the mean, unlike RMSNorm."""
        x = jax.random.normal(jax.random.key(0), (4, 8)) + 5.0
        assert not jnp.allclose(nn.LayerNorm(8, use_bias=False)(x), nn.RMSNorm(8)(x))

    @pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
    def test_mixed_precision(self, dtype):
        """Normalization uses float32 while preserving the input dtype."""
        layer = nn.LayerNorm(4096).astype(dtype)
        x = (100 * jax.random.normal(jax.random.key(0), (2, 4096))).astype(dtype)

        y = layer(x)
        expected = nn.LayerNorm(4096)(x.astype(jnp.float32)).astype(dtype)

        assert y.dtype == dtype
        npt.assert_allclose(y, expected, rtol=1e-2, atol=1e-2)


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

    @pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
    def test_mixed_precision(self, dtype):
        """Normalization uses float32 while preserving the input dtype."""
        layer = nn.GroupNorm(8, 2, num_spatial_dims=2).astype(dtype)
        x = (100 * jax.random.normal(jax.random.key(0), (2, 32, 32, 8))).astype(dtype)

        y = layer(x)
        expected = nn.GroupNorm(8, 2, num_spatial_dims=2)(x.astype(jnp.float32)).astype(dtype)

        assert y.dtype == dtype
        npt.assert_allclose(y, expected, rtol=1e-2, atol=1e-2)


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

    @pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
    def test_mixed_precision(self, dtype):
        """Normalization uses float32 while preserving the input dtype."""
        layer = nn.RMSNorm(4096).astype(dtype)
        x = (100 * jax.random.normal(jax.random.key(0), (2, 4096))).astype(dtype)

        y = layer(x)
        expected = nn.RMSNorm(4096)(x.astype(jnp.float32)).astype(dtype)

        assert y.dtype == dtype
        npt.assert_allclose(y, expected, rtol=1e-2, atol=1e-2)


class TestSpectralNorm:
    def test_vectors_init_reproducible(self):
        """The same key produces identical power-iteration vectors."""
        linear = nn.Linear(4, 5, key=jax.random.key(0))
        first = nn.SpectralNorm(linear, key=jax.random.key(1))
        second = nn.SpectralNorm(linear, key=jax.random.key(1))
        npt.assert_array_equal(first.u.value, second.u.value)
        npt.assert_array_equal(first.v.value, second.v.value)

    def test_key_is_required(self):
        """Construction without a key raises TypeError."""
        with pytest.raises(TypeError):
            nn.SpectralNorm(nn.Linear(4, 5, key=jax.random.key(0)))  # type: ignore[call-arg]

    @pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
    def test_mixed_precision(self, dtype):
        """Power-iteration vectors keep their own dtype without promoting the output."""
        layer = nn.SpectralNorm(nn.Linear(4, 5, key=jax.random.key(0)), key=jax.random.key(1))
        low_precision_layer = layer.astype(dtype)
        x = jnp.ones((2, 4), dtype=dtype)

        y = low_precision_layer(x, training=True)

        assert y.dtype == dtype
        assert low_precision_layer.u.value.dtype == jnp.float32
        assert low_precision_layer.v.value.dtype == jnp.float32

    @pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
    @pytest.mark.parametrize("training", [False, True])
    def test_preserves_input_dtype(self, dtype, training):
        """The output follows the input dtype even with default float32 params."""
        layer = nn.SpectralNorm(
            nn.Linear(4, 5, key=jax.random.key(0)),
            key=jax.random.key(1),
        )
        x = jnp.ones((2, 4), dtype=dtype)

        assert layer(x, training=training).dtype == dtype

    @pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
    def test_low_precision_construction(self, dtype):
        """Direct low-precision construction keeps zero weights finite."""
        linear = nn.Linear(4, 5, use_bias=False, key=jax.random.key(0)).astype(dtype)
        linear = linear.at.w.set(nn.Param(jnp.zeros((4, 5), dtype=dtype)))

        layer = nn.SpectralNorm(linear, key=jax.random.key(1))
        y = layer(jnp.ones((2, 4), dtype=dtype), training=True)

        assert jnp.all(jnp.isfinite(y))
        assert y.dtype == dtype
        assert layer.u.value.dtype == jnp.float32
        assert layer.v.value.dtype == jnp.float32

    def test_spectral_norm(self):
        """The normalized weight has largest singular value near one."""
        linear = nn.Linear(8, 8, use_bias=False, key=jax.random.key(0))
        layer = nn.SpectralNorm(linear, power_iterations=20, key=jax.random.key(1))
        weight = jnp.asarray(linear.w)
        matrix = weight.reshape(-1, weight.shape[-1]).T
        sigma_estimate = layer.u.value @ matrix @ layer.v.value
        normalized = matrix / sigma_estimate
        sigma = jnp.linalg.svd(normalized, compute_uv=False)[0]
        npt.assert_allclose(sigma, 1.0, rtol=1e-4, atol=1e-4)

    def test_training_updates_buffers(self):
        """Training updates the power-iteration vectors."""
        layer = nn.SpectralNorm(nn.Linear(4, 5, key=jax.random.key(0)), key=jax.random.key(1))
        before = layer.u.value
        layer(jnp.ones((2, 4)), training=True)
        assert not jnp.array_equal(layer.u.value, before)

    def test_evaluation_preserves_buffers(self):
        """Evaluation leaves the power-iteration vectors unchanged."""
        layer = nn.SpectralNorm(nn.Linear(4, 5, key=jax.random.key(0)), key=jax.random.key(1))
        before = layer.u.value
        layer(jnp.ones((2, 4)), training=False)
        npt.assert_array_equal(layer.u.value, before)

    def test_evaluation_stable(self):
        """Repeated evaluation produces identical output."""
        layer = nn.SpectralNorm(nn.Linear(4, 5, key=jax.random.key(0)), key=jax.random.key(1))
        x = jax.random.normal(jax.random.key(2), (3, 4))
        npt.assert_array_equal(layer(x, training=False), layer(x, training=False))

    def test_input_grad(self):
        """Input gradients are finite."""
        layer = nn.SpectralNorm(nn.Linear(4, 5, key=jax.random.key(0)), key=jax.random.key(1))
        x = jax.random.normal(jax.random.key(2), (3, 4))
        input_grad = jax.grad(lambda value: layer(value, training=True).sum())(x)

        assert jnp.all(jnp.isfinite(input_grad))

    def test_parameter_grad(self):
        """Wrapped parameter gradients are finite."""
        layer = nn.SpectralNorm(nn.Linear(4, 5, key=jax.random.key(0)), key=jax.random.key(1))
        x = jax.random.normal(jax.random.key(2), (3, 4))
        model_grad = jax.grad(lambda model: model(x, training=True).sum())(layer)

        for leaf in jax.tree.leaves(model_grad):
            assert jnp.all(jnp.isfinite(leaf))

    def test_linear(self):
        """Linear weights can be spectrally normalized."""
        linear = nn.SpectralNorm(nn.Linear(4, 5, key=jax.random.key(0)), key=jax.random.key(1))
        y = linear(jnp.ones((2, 4)), training=True)
        assert y.shape == (2, 5)

    def test_conv(self):
        """Convolutional weights flatten across non-output dimensions."""
        conv = nn.SpectralNorm(
            nn.Conv(3, 6, kernel_shape=(3, 3), padding=1, key=jax.random.key(2)),
            key=jax.random.key(3),
        )
        y = conv(jnp.ones((2, 8, 8, 3)), training=True)
        assert y.shape == (2, 8, 8, 6)

    def test_wraps_a_stateful_module(self):
        """A wrapped module may own buffers of its own."""

        class Stateful(nn.Module):
            w: nn.Param
            count: nn.Buffer

            def __init__(self):
                self.w = nn.Param(jnp.eye(2))
                self.count = nn.Buffer(jnp.zeros(2))

            def __call__(self, x):
                self.count.set(self.count.value + 1)
                return x @ self.w

        wrapped = Stateful()
        layer = nn.SpectralNorm(wrapped, key=jax.random.key(0))
        layer(jnp.ones((3, 2)), training=True)
        npt.assert_array_equal(wrapped.count.value, jnp.ones(2))

    def test_invalid_parameter_name(self):
        """A missing parameter name raises AttributeError."""
        with pytest.raises(AttributeError):
            nn.SpectralNorm(
                nn.Linear(4, 5, key=jax.random.key(0)),
                parameter="missing",
                key=jax.random.key(1),
            )

    def test_invalid_parameter_type(self):
        """A plain-array parameter raises TypeError."""

        class NotParam(nn.Module):
            w: jax.Array

            def __init__(self):
                self.w = jnp.ones((2, 2))

        with pytest.raises(TypeError, match="must be a Param"):
            nn.SpectralNorm(NotParam(), key=jax.random.key(0))

    def test_invalid_parameter_shape(self):
        """A parameter with fewer than two dimensions raises ValueError."""

        class Vector(nn.Module):
            w: nn.Param

            def __init__(self):
                self.w = nn.Param(jnp.ones(2))

        with pytest.raises(ValueError, match="at least 2D"):
            nn.SpectralNorm(Vector(), key=jax.random.key(0))

    def test_invalid_power_iterations(self):
        """power_iterations below one raises ValueError."""
        linear = nn.Linear(4, 5, key=jax.random.key(0))
        with pytest.raises(ValueError, match="power_iterations"):
            nn.SpectralNorm(linear, power_iterations=0, key=jax.random.key(1))

    def test_jit_training(self):
        """Training works under jax.jit."""
        layer = nn.SpectralNorm(nn.Linear(4, 5, key=jax.random.key(0)), key=jax.random.key(1))
        x = jax.random.normal(jax.random.key(2), (3, 4))
        train = jax.jit(lambda x: layer(x, training=True))
        y = train(x)

        assert y.shape == (3, 5)

    def test_jit_evaluation(self):
        """Evaluation works under jax.jit without changing the vectors."""
        layer = nn.SpectralNorm(nn.Linear(4, 5, key=jax.random.key(0)), key=jax.random.key(1))
        x = jax.random.normal(jax.random.key(2), (3, 4))
        before = layer.u.value
        evaluate = jax.jit(lambda x: layer(x, training=False))
        y = evaluate(x)

        assert y.shape == (3, 5)
        npt.assert_array_equal(layer.u.value, before)
