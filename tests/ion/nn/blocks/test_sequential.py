import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from ion import nn


class TestSequential:
    def test_output_shape(self):
        """Output shape follows the last layer's output dim."""
        key = jax.random.key(0)
        keys = jax.random.split(key, 2)
        model = nn.Sequential(
            nn.Linear(4, 8, key=keys[0]),
            jax.nn.relu,
            nn.Linear(8, 2, key=keys[1]),
        )
        x = jnp.ones((4,))
        y = model(x)
        assert y.shape == (2,)

    def test_forward_matches_manual(self):
        """Forward pass matches manually chaining each layer."""
        key = jax.random.key(0)
        keys = jax.random.split(key, 2)
        l1 = nn.Linear(4, 8, key=keys[0])
        l2 = nn.Linear(8, 2, key=keys[1])
        model = nn.Sequential(l1, jax.nn.relu, l2)

        x = jax.random.normal(jax.random.key(1), (4,))
        expected = l2(jax.nn.relu(l1(x)))
        npt.assert_allclose(model(x), expected, rtol=1e-5, atol=1e-5)

    def test_len(self):
        """len() returns the number of layers."""
        model = nn.Sequential(jax.nn.relu, jax.nn.sigmoid, jax.nn.tanh)
        assert len(model) == 3

    def test_getitem_int(self):
        """Integer indexing returns the corresponding layer."""
        key = jax.random.key(0)
        linear = nn.Linear(4, 8, key=key)
        model = nn.Sequential(linear, jax.nn.relu)
        assert model[0] is linear
        assert model[1] is jax.nn.relu

    def test_getitem_slice(self):
        """Slice indexing returns a new Sequential with the sliced layers."""
        key = jax.random.key(0)
        keys = jax.random.split(key, 2)
        l1 = nn.Linear(4, 8, key=keys[0])
        l2 = nn.Linear(8, 2, key=keys[1])
        model = nn.Sequential(l1, jax.nn.relu, l2)

        sliced = model[:2]
        assert isinstance(sliced, nn.Sequential)
        assert len(sliced) == 2
        assert sliced[0] is l1
        assert sliced[1] is jax.nn.relu

    def test_iter(self):
        """Iterating yields each layer in order."""
        key = jax.random.key(0)
        linear = nn.Linear(4, 8, key=key)
        layers = [linear, jax.nn.relu]
        model = nn.Sequential(*layers)
        assert list(model) == layers

    def test_empty(self):
        """Empty Sequential returns input unchanged."""
        model = nn.Sequential()
        x = jnp.ones((4,))
        npt.assert_allclose(model(x), x, rtol=0, atol=0)
        assert len(model) == 0

    def test_non_callable_raises(self):
        """Passing a non-callable raises TypeError."""
        with pytest.raises(TypeError, match="callable"):
            nn.Sequential(42)  # type: ignore[arg-type]
