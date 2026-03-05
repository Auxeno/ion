import jax
import jax.numpy as jnp
import numpy.testing as npt

from ion import nn


class TestDropout:
    def test_deterministic_is_identity(self):
        """Deterministic mode returns input unchanged."""
        layer = nn.Dropout(p=0.5, deterministic=True)
        x = jnp.ones((4, 8))
        y = layer(x, key=jax.random.key(0))
        npt.assert_allclose(y, x, rtol=0, atol=0)

    def test_deterministic_override(self):
        """Call-time deterministic flag overrides init-time flag."""
        layer = nn.Dropout(p=0.5, deterministic=False)
        x = jnp.ones((4, 8))
        y = layer(x, deterministic=True, key=jax.random.key(0))
        npt.assert_allclose(y, x, rtol=0, atol=0)

    def test_p_zero_is_identity(self):
        """p=0 returns input unchanged even in stochastic mode."""
        layer = nn.Dropout(p=0.0)
        x = jnp.ones((4, 8))
        y = layer(x, key=jax.random.key(0))
        npt.assert_allclose(y, x, rtol=0, atol=0)

    def test_drops_some_values(self):
        """Stochastic mode zeros out some elements."""
        layer = nn.Dropout(p=0.5)
        x = jnp.ones((1000,))
        y = layer(x, key=jax.random.key(0))
        num_zeros = jnp.sum(y == 0.0)
        assert num_zeros > 0
        assert num_zeros < 1000

    def test_inverted_scaling(self):
        """Kept values are scaled by 1/(1-p) to preserve expected value."""
        layer = nn.Dropout(p=0.5)
        x = jnp.ones((1000,))
        y = layer(x, key=jax.random.key(0))
        kept = y[y != 0.0]
        npt.assert_allclose(kept, 2.0, rtol=1e-5, atol=1e-5)

    def test_expected_value_preserved(self):
        """Mean output is approximately equal to mean input over many samples."""
        layer = nn.Dropout(p=0.3)
        x = jnp.ones((10000,))
        y = layer(x, key=jax.random.key(42))
        npt.assert_allclose(jnp.mean(y), 1.0, atol=0.05)

    def test_different_keys_different_masks(self):
        """Different PRNG keys produce different dropout masks."""
        layer = nn.Dropout(p=0.5)
        x = jnp.ones((100,))
        y1 = layer(x, key=jax.random.key(0))
        y2 = layer(x, key=jax.random.key(1))
        assert not jnp.array_equal(y1, y2)

    def test_output_shape_preserved(self):
        """Output shape matches input shape."""
        layer = nn.Dropout(p=0.5)
        x = jnp.ones((2, 3, 4))
        y = layer(x, key=jax.random.key(0))
        assert y.shape == x.shape

    def test_output_dtype_preserved(self):
        """Output dtype matches input dtype."""
        layer = nn.Dropout(p=0.5)
        x = jnp.ones((8,), dtype=jnp.float32)
        y = layer(x, key=jax.random.key(0))
        assert y.dtype == x.dtype

    def test_drop_rate(self):
        """Fraction of zeros is approximately p for large inputs."""
        layer = nn.Dropout(p=0.3)
        x = jnp.ones((10000,))
        y = layer(x, key=jax.random.key(42))
        drop_frac = jnp.mean(y == 0.0)
        npt.assert_allclose(drop_frac, 0.3, atol=0.03)
