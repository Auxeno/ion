import jax.numpy as jnp
import numpy.testing as npt

from ion import nn


class TestIdentity:
    def test_passthrough(self):
        """Identity returns the input unchanged."""
        layer = nn.Identity()
        x = jnp.ones((3, 4))
        y = layer(x)
        npt.assert_array_equal(y, x)

    def test_ignores_args(self):
        """Identity constructor accepts and ignores arbitrary arguments."""
        layer = nn.Identity(1, 2, foo="bar")
        x = jnp.ones((3,))
        y = layer(x)
        npt.assert_array_equal(y, x)
