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
        npt.assert_array_equal(model(x), x)
        assert len(model) == 0

    def test_dropout_deterministic(self):
        """Deterministic Dropout layers pass through unchanged, no key needed."""
        key = jax.random.key(0)
        linear = nn.Linear(4, 4, key=key)
        model = nn.Sequential(linear, nn.Dropout(0.5, deterministic=True))
        x = jnp.ones((4,))
        npt.assert_allclose(model(x), linear(x), rtol=1e-5, atol=1e-5)

    def test_dropout_matches_manual(self):
        """Forward with dropout matches manually chaining with per-layer split keys."""
        key = jax.random.key(0)
        drop = nn.Dropout(0.5)
        model = nn.Sequential(drop, jax.nn.relu, drop)

        x = jnp.ones((100,))
        keys = jax.random.split(key, 3)
        expected = drop(jax.nn.relu(drop(x, key=keys[0])), key=keys[2])
        npt.assert_allclose(model(x, key=key), expected, rtol=1e-5, atol=1e-5)

    def test_dropout_requires_key(self):
        """Calling with active dropout and no key raises ValueError."""
        model = nn.Sequential(nn.Dropout(0.5))
        with pytest.raises(ValueError, match="key"):
            model(jnp.ones((4,)))

    def test_key_routed_by_signature(self):
        """Any callable accepting a key kwarg receives a per-layer subkey."""
        key = jax.random.key(0)

        def noise(x, *, key):
            return x + jax.random.normal(key, x.shape)

        model = nn.Sequential(noise, jax.nn.relu)

        x = jnp.zeros((100,))
        keys = jax.random.split(key, 2)
        expected = jax.nn.relu(noise(x, key=keys[0]))
        npt.assert_allclose(model(x, key=key), expected, rtol=1e-5, atol=1e-5)

    def test_non_callable_raises(self):
        """Passing a non-callable raises TypeError."""
        with pytest.raises(TypeError, match="callable"):
            nn.Sequential(42)  # type: ignore[arg-type]
