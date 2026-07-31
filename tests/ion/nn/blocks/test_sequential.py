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
        npt.assert_allclose(
            model(x, training=True, key=key), expected, rtol=1e-5, atol=1e-5
        )

    def test_dropout_requires_training(self):
        """A contained Dropout requires an explicit training mode."""
        model = nn.Sequential(nn.Dropout(0.5))
        with pytest.raises(ValueError, match="requires training"):
            model(jnp.ones((4,)))

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

    def test_non_callable_raises(self):
        """Passing a non-callable raises TypeError."""
        with pytest.raises(TypeError, match="callable"):
            nn.Sequential(42)  # type: ignore[arg-type]

    def test_buffers_forwarded(self):
        """Buffers are forwarded and returned after stateful layers update them."""
        model = nn.Sequential(nn.BatchNorm(4), jax.nn.relu)
        buffers = model.init_buffers()
        x = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)
        y, updated = model(x, buffers, training=True)
        assert y.shape == x.shape
        assert updated is not buffers

    def test_mixed_layer_routing(self):
        """Buffers, mode, and keys route through mixed layer types."""
        key = jax.random.key(0)
        model = nn.Sequential(
            nn.BatchNorm(4),
            jax.nn.relu,
            nn.Dropout(0.2),
            nn.Linear(4, 2, key=key),
        )
        buffers = model.init_buffers()
        x = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)
        y, buffers = model(x, buffers, training=True, key=jax.random.key(1))
        assert y.shape == (3, 2)
        evaluated, same_buffers = model(x, buffers, training=False)
        assert evaluated.shape == (3, 2)
        assert same_buffers is buffers

    def test_stateful_layer_requires_buffers(self):
        """A contained stateful layer requires initialized buffers."""
        model = nn.Sequential(nn.BatchNorm(4))
        with pytest.raises(ValueError, match=r"model.init_buffers\(\)"):
            model(jnp.ones((2, 4)), training=True)

    def test_explicit_buffers_returns_pair(self):
        """Passing buffers makes a stateless sequence return an output-buffer pair."""
        model = nn.Sequential(jax.nn.relu)
        buffers = model.init_buffers()
        x = jnp.array([-1.0, 2.0])
        y, returned = model(x, buffers)
        npt.assert_array_equal(y, jnp.array([0.0, 2.0]))
        assert returned is buffers

    def test_jit_training_and_evaluation(self):
        """Mixed stateful and stochastic layers work under jax.jit."""
        model = nn.Sequential(nn.BatchNorm(4), nn.Dropout(0.2))
        buffers = model.init_buffers()
        x = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)

        train = jax.jit(lambda x, b, k: model(x, b, training=True, key=k))
        evaluate = jax.jit(lambda x, b: model(x, b, training=False))
        y, buffers = train(x, buffers, jax.random.key(0))
        eval_y, _ = evaluate(x, buffers)
        assert y.shape == eval_y.shape == x.shape
