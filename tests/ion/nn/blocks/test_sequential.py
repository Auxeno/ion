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

    def test_dropout_evaluation(self):
        """Evaluation mode is forwarded and needs no random key."""
        key = jax.random.key(0)
        linear = nn.Linear(4, 4, key=key)
        model = nn.Sequential(linear, nn.Dropout(0.5))
        x = jnp.ones((4,))
        npt.assert_allclose(model(x, training=False), linear(x), rtol=1e-5, atol=1e-5)

    def test_dropout_matches_manual(self):
        """Forward with dropout matches manually chaining with per-layer split keys."""
        key = jax.random.key(0)
        drop = nn.Dropout(0.5)
        model = nn.Sequential(drop, jax.nn.relu, drop)

        x = jnp.ones((100,))
        keys = jax.random.split(key, 3)
        expected = drop(
            jax.nn.relu(drop(x, training=True, key=keys[0])),
            training=True,
            key=keys[2],
        )
        npt.assert_allclose(model(x, training=True, key=key), expected, rtol=1e-5, atol=1e-5)

    def test_dropout_requires_training(self):
        """A contained Dropout requires an explicit training mode."""
        model = nn.Sequential(nn.Dropout(0.5))
        with pytest.raises(ValueError, match="requires training"):
            model(jnp.ones((4,)))

    def test_optional_training_uses_default(self):
        """An optional training argument does not require an explicit mode."""

        def scale(x, *, training=False):
            return x * (2 if training else 1)

        model = nn.Sequential(scale)
        x = jnp.ones((4,))

        npt.assert_array_equal(model(x), x)
        npt.assert_array_equal(model(x, training=True), 2 * x)

    def test_dropout_training_requires_key(self):
        """Training with a contained Dropout requires a random key."""
        model = nn.Sequential(nn.Dropout(0.5))
        with pytest.raises(ValueError, match="key"):
            model(jnp.ones((4,)), training=True)

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

    def test_stateful_layer_updates_in_place(self):
        """A contained stateful layer updates its buffers and returns one value."""
        norm = nn.BatchNorm(4)
        model = nn.Sequential(norm, jax.nn.relu)
        x = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)
        y = model(x, training=True)
        assert y.shape == x.shape
        assert not jnp.array_equal(norm.running_mean.value, jnp.zeros(4))

    def test_mixed_layer_routing(self):
        """Mode and keys route through mixed layer types."""
        key = jax.random.key(0)
        model = nn.Sequential(
            nn.BatchNorm(4),
            jax.nn.relu,
            nn.Dropout(0.2),
            nn.Linear(4, 2, key=key),
        )
        x = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)
        y = model(x, training=True, key=jax.random.key(1))
        assert y.shape == (3, 2)
        evaluated = model(x, training=False)
        assert evaluated.shape == (3, 2)

    def test_jit_training_and_evaluation(self):
        """Mixed stateful and stochastic layers work under jax.jit."""
        model = nn.Sequential(nn.BatchNorm(4), nn.Dropout(0.2))
        x = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)

        train = jax.jit(lambda x, k: model(x, training=True, key=k))
        evaluate = jax.jit(lambda x: model(x, training=False))
        y = train(x, jax.random.key(0))
        eval_y = evaluate(x)
        assert y.shape == eval_y.shape == x.shape
