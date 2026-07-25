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

    def test_glorot_uniform_init(self):
        """Glorot uniform initialization gives var(w) close to 2/(fan_in + fan_out)."""
        key = jax.random.key(42)
        layer = nn.Linear(2048, 2048, key=key)
        var = jnp.var(layer.w._value)
        expected_var = 2.0 / (2048 + 2048)
        npt.assert_allclose(var, expected_var, rtol=0.05)

    def test_zero_bias_init(self):
        """Bias is initialized to all zeros."""
        key = jax.random.key(0)
        layer = nn.Linear(8, 16, key=key)
        assert jnp.all(layer.b == 0)

    def test_default_dtype(self):
        """Weights default to float32."""
        key = jax.random.key(0)
        layer = nn.Linear(8, 16, key=key)
        assert layer.w.dtype == jnp.float32
