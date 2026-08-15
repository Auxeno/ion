import jax
import jax.numpy as jnp
import numpy.testing as npt

from ion import nn


class TestResidual:
    def test_matches_manual_addition(self):
        """Output is the sum of the input and wrapped layer output."""
        linear = nn.Linear(4, 4, key=jax.random.key(0))
        residual = nn.Residual(linear)
        x = jax.random.normal(jax.random.key(1), (3, 4))

        npt.assert_allclose(residual(x), x + linear(x), rtol=1e-5, atol=1e-5)

    def test_forwards_arguments(self):
        """Additional positional and keyword arguments reach the wrapped layer."""

        def scale(x, factor, *, offset):
            return x * factor + offset

        residual = nn.Residual(scale)
        x = jnp.ones((4,))

        npt.assert_array_equal(residual(x, 2, offset=3), jnp.full((4,), 6.0))

    def test_plain_callable(self):
        """The wrapped layer may be an ordinary callable."""
        residual = nn.Residual(jax.nn.relu)
        x = jnp.array([-1.0, 2.0])

        npt.assert_array_equal(residual(x), jnp.array([-1.0, 4.0]))

    def test_nested_dropout(self):
        """Training mode and random keys route through nested Sequential blocks."""
        dropout = nn.Dropout(0.5)
        residual = nn.Sequential(nn.Residual(nn.Sequential(dropout)))
        x = jnp.ones((100,))
        key = jax.random.key(0)

        outer_key = jax.random.split(key, 1)[0]
        inner_key = jax.random.split(outer_key, 1)[0]
        expected = x + dropout(x, training=True, key=inner_key)
        result = residual(x, training=True, key=key)

        npt.assert_array_equal(result, expected)

    def test_nested_dropout_requires_training(self):
        """A nested stochastic layer still requires an explicit training mode."""
        residual = nn.Sequential(nn.Residual(nn.Sequential(nn.Dropout(0.5))))

        with npt.assert_raises_regex(ValueError, "requires training"):
            residual(jnp.ones((4,)))
