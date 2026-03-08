import jax
import jax.numpy as jnp
import numpy.testing as npt

from ion import nn


class TestLinear:
    def test_output_manual(self):
        """Output matches manual x @ w + b computation."""
        key = jax.random.key(0)
        layer = nn.Linear(4, 8, key=key)
        x = jax.random.normal(jax.random.key(1), (4,))
        y = layer(x)
        expected = x @ layer.w + layer.b  # type: ignore[operator]
        npt.assert_allclose(y, expected, rtol=1e-5, atol=1e-5)

    def test_no_bias(self):
        """No-bias mode: output matches x @ w with no bias term."""
        key = jax.random.key(0)
        layer = nn.Linear(4, 8, bias=False, key=key)
        assert layer.b is None
        x = jax.random.normal(jax.random.key(1), (4,))
        y = layer(x)
        expected = x @ layer.w
        npt.assert_allclose(y, expected, rtol=1e-5, atol=1e-5)

    def test_he_normal_init(self):
        """He normal initialization gives var(w) close to 2/fan_in."""
        key = jax.random.key(42)
        layer = nn.Linear(2048, 2048, key=key)
        var = jnp.var(layer.w._value)
        expected_var = 2.0 / 2048
        npt.assert_allclose(var, expected_var, atol=0.05)

    def test_zero_bias_init(self):
        """Bias is initialized to all zeros."""
        key = jax.random.key(0)
        layer = nn.Linear(8, 16, key=key)
        assert jnp.all(layer.b == 0)

    def test_weight_dtype(self):
        """Weights match the requested dtype."""
        key = jax.random.key(0)
        layer = nn.Linear(8, 16, dtype=jnp.float32, key=key)
        assert layer.w.dtype == jnp.float32


class TestIdentity:
    def test_passthrough(self):
        """Identity returns the input unchanged."""
        layer = nn.Identity()
        x = jnp.ones((3, 4))
        y = layer(x)
        npt.assert_allclose(y, x, rtol=0, atol=0)  # type: ignore[arg-type]

    def test_ignores_args(self):
        """Identity constructor accepts and ignores arbitrary arguments."""
        layer = nn.Identity(1, 2, foo="bar")
        x = jnp.ones((3,))
        y = layer(x)
        npt.assert_allclose(y, x, rtol=0, atol=0)  # type: ignore[arg-type]
