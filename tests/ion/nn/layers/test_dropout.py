import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from ion import nn


class TestDropout:
    def test_training_is_required(self):
        """Omitting the required training mode raises TypeError."""
        layer = nn.Dropout(p=0.5)
        with pytest.raises(TypeError, match="training"):
            layer(jnp.ones(4))  # type: ignore[call-arg]

    def test_evaluation_is_identity(self):
        """Evaluation returns input unchanged and needs no key."""
        layer = nn.Dropout(p=0.5)
        x = jnp.ones((4, 8))
        npt.assert_array_equal(layer(x, training=False), x)

    def test_evaluation_ignores_supplied_key(self):
        """Evaluation ignores a supplied random key."""
        layer = nn.Dropout(p=0.5)
        x = jnp.ones((4, 8))
        npt.assert_array_equal(layer(x, training=False, key=jax.random.key(0)), x)

    def test_no_deterministic_argument(self):
        """The removed deterministic constructor argument raises TypeError."""
        with pytest.raises(TypeError, match="deterministic"):
            nn.Dropout(p=0.5, deterministic=True)  # type: ignore[call-arg]

    def test_p_out_of_range_raises(self):
        """p outside [0, 1] raises ValueError at construction."""
        with pytest.raises(ValueError, match="must be in"):
            nn.Dropout(p=-0.3)
        with pytest.raises(ValueError, match="must be in"):
            nn.Dropout(p=1.5)

    def test_p_boundaries_construct(self):
        """p=0 and p=1 are valid."""
        assert nn.Dropout(p=0.0).p == 0.0
        assert nn.Dropout(p=1.0).p == 1.0

    def test_p_zero_is_identity(self):
        """p=0 returns input unchanged during training without a key."""
        layer = nn.Dropout(p=0.0)
        x = jnp.ones((4, 8))
        npt.assert_array_equal(layer(x, training=True), x)

    def test_drops_some_values(self):
        """Training zeros out some elements."""
        layer = nn.Dropout(p=0.5)
        x = jnp.ones((1000,))
        y = layer(x, training=True, key=jax.random.key(0))
        num_zeros = jnp.sum(y == 0.0)
        assert num_zeros > 0
        assert num_zeros < 1000

    def test_inverted_scaling(self):
        """Kept values are scaled by 1/(1-p)."""
        layer = nn.Dropout(p=0.5)
        x = jnp.ones((1000,))
        y = layer(x, training=True, key=jax.random.key(0))
        kept = y[y != 0.0]
        npt.assert_allclose(kept, 2.0, rtol=1e-5, atol=1e-5)

    def test_expected_value_preserved(self):
        """Mean output approximately matches mean input over many samples."""
        layer = nn.Dropout(p=0.3)
        x = jnp.ones((10000,))
        y = layer(x, training=True, key=jax.random.key(42))
        npt.assert_allclose(jnp.mean(y), 1.0, atol=0.05)

    def test_different_keys_different_masks(self):
        """Different random keys produce different dropout masks."""
        layer = nn.Dropout(p=0.5)
        x = jnp.ones((100,))
        y1 = layer(x, training=True, key=jax.random.key(0))
        y2 = layer(x, training=True, key=jax.random.key(1))
        assert not jnp.allclose(y1, y2)

    def test_output_shape_preserved(self):
        """Output shape matches input shape."""
        layer = nn.Dropout(p=0.5)
        x = jnp.ones((2, 3, 4))
        y = layer(x, training=True, key=jax.random.key(0))
        assert y.shape == x.shape

    def test_output_dtype_preserved(self):
        """Output dtype matches input dtype."""
        layer = nn.Dropout(p=0.5)
        x = jnp.ones((8,), dtype=jnp.float32)
        y = layer(x, training=True, key=jax.random.key(0))
        assert y.dtype == x.dtype

    def test_drop_rate(self):
        """Fraction of zeros is approximately p for large inputs."""
        layer = nn.Dropout(p=0.3)
        x = jnp.ones((10000,))
        y = layer(x, training=True, key=jax.random.key(42))
        drop_frac = jnp.mean(y == 0.0)
        npt.assert_allclose(drop_frac, 0.3, atol=0.03)

    def test_p_one_returns_zeros(self):
        """p=1 drops everything without producing NaN."""
        layer = nn.Dropout(p=1.0)
        x = jnp.ones((4, 8))
        y = layer(x, training=True, key=jax.random.key(0))
        npt.assert_array_equal(y, jnp.zeros_like(x))

    def test_training_requires_key(self):
        """Training with active dropout and no key raises ValueError."""
        with pytest.raises(ValueError, match="key is required when training=True"):
            nn.Dropout(p=0.5)(jnp.ones(4), training=True)

    def test_jit_training(self):
        """Training works under jax.jit."""
        layer = nn.Dropout(p=0.5)
        x = jnp.ones(100)
        train = jax.jit(lambda x, key: layer(x, training=True, key=key))
        assert jnp.any(train(x, jax.random.key(0)) == 0)

    def test_jit_evaluation(self):
        """Evaluation works under jax.jit."""
        layer = nn.Dropout(p=0.5)
        x = jnp.ones(100)
        evaluate = jax.jit(lambda x: layer(x, training=False))
        npt.assert_array_equal(evaluate(x), x)

    def test_jit_static_training(self):
        """A shared jitted call works when training is static."""
        layer = nn.Dropout(p=0.5)
        apply = jax.jit(layer, static_argnames=("training",))
        x = jnp.ones(100)
        assert jnp.any(apply(x, training=True, key=jax.random.key(0)) == 0)
        npt.assert_array_equal(apply(x, training=False), x)
