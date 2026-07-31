import dataclasses
import tempfile
from typing import cast

import jax
import jax.numpy as jnp
import numpy.testing as npt
import optax
import pytest

import ion
from ion import nn
from ion.nn.buffer import _Buffers


class Counter(nn.BufferModule):
    size: int

    def __init__(self, size=2):
        self.size = size

    def _init_buffer(self, *, key=None):
        return jnp.zeros((self.size,), dtype=jnp.float32)

    def __call__(self, x, buffers, *, training: bool):
        value = buffers[self]
        return x + value, buffers.set(self, value + 1) if training else buffers


class RandomBuffer(nn.BufferModule):
    size: int

    def __init__(self, size=2):
        self.size = size

    def _init_buffer(self, *, key=None):
        if key is None:
            raise ValueError("call `model.init_buffers(key=key)`")
        return jax.random.normal(key, (self.size,))


def test_stateless_model_has_empty_buffers():
    """A stateless model initializes an empty buffer collection."""
    buffers = nn.Linear(2, 3, key=jax.random.key(0)).init_buffers()
    assert isinstance(buffers, _Buffers)
    assert jax.tree.leaves(buffers) == []


def test_nested_container_discovery():
    """Nested buffer modules are discovered in structural order."""

    class Model(nn.Module):
        direct: Counter
        nested: tuple
        listed: list
        mapped: dict

        def __init__(self):
            self.direct = Counter(1)
            self.nested = ((Counter(2),),)
            self.listed = [Counter(3)]
            self.mapped = {"b": Counter(5), "a": Counter(4)}

    model = Model()
    buffers = model.init_buffers()
    assert [leaf.shape for leaf in jax.tree.leaves(buffers)] == [(1,), (2,), (3,), (4,), (5,)]


def test_sequential_discovery():
    """Buffer modules inside Sequential are discovered."""
    layer = Counter()
    model = nn.Sequential(layer, lambda x: x)
    buffers = model.init_buffers()
    assert len(jax.tree.leaves(buffers)) == 1


def test_repeated_instance_deduplicated():
    """A repeated buffer-module instance receives one payload."""
    repeated = Counter()
    model = nn.Sequential(repeated, (lambda x: x), repeated)
    buffers = model.init_buffers()
    assert len(jax.tree.leaves(buffers)) == 1


def test_deterministic_init_without_key():
    """Deterministic buffers initialize without a random key."""
    layer = Counter(3)
    npt.assert_array_equal(layer.init_buffers()[layer], jnp.zeros(3))
    assert "_buffer_key" not in repr(layer)


def test_random_init_reproducible():
    """The same random key produces identical buffer values."""

    class Model(nn.Module):
        layers: tuple

        def __init__(self):
            self.layers = (RandomBuffer(), RandomBuffer())

    model = Model()
    key = jax.random.key(4)
    first = model.init_buffers(key=key)
    second = model.init_buffers(key=key)
    for layer in model.layers:
        npt.assert_array_equal(first[layer], second[layer])


def test_random_keys_split_per_layer():
    """Each buffer module receives its structural split of the random key."""

    class Model(nn.Module):
        layers: tuple

        def __init__(self):
            self.layers = (RandomBuffer(), RandomBuffer())

    model = Model()
    key = jax.random.key(4)
    buffers = model.init_buffers(key=key)
    expected_keys = jax.random.split(key, 2)
    for layer, expected_key in zip(model.layers, expected_keys):
        npt.assert_array_equal(buffers[layer], jax.random.normal(expected_key, (2,)))


def test_random_init_requires_key():
    """A random buffer initializer can require model.init_buffers(key=key)."""
    with pytest.raises(ValueError, match=r"init_buffers\(key=key\)"):
        RandomBuffer().init_buffers()


def test_set_returns_new_collection():
    """set returns an updated collection without changing the original."""
    layer = Counter()
    buffers = layer.init_buffers()
    updated = buffers.set(layer, jnp.ones(2))
    npt.assert_array_equal(buffers[layer], jnp.zeros(2))
    npt.assert_array_equal(updated[layer], jnp.ones(2))


def test_collection_is_immutable():
    """Buffer collection fields cannot be assigned directly."""
    buffers = Counter().init_buffers()
    with pytest.raises(dataclasses.FrozenInstanceError):
        buffers._values = ()  # type: ignore[misc]


def test_jit():
    """Buffer reads and updates work under jax.jit."""
    layer = Counter()
    buffers = layer.init_buffers()

    @jax.jit
    def step(buffers):
        return layer(jnp.zeros(2), buffers, training=True)

    _, updated = step(buffers)
    npt.assert_array_equal(updated[layer], jnp.ones(2))


def test_scan():
    """A buffer collection can be carried through lax.scan."""
    layer = Counter()
    buffers = layer.init_buffers()

    def body(buffers, _):
        _, buffers = layer(jnp.zeros(2), buffers, training=True)
        return buffers, None

    scanned, _ = jax.lax.scan(body, buffers, xs=None, length=3)
    npt.assert_array_equal(scanned[layer], jnp.full(2, 3.0))


def test_get_stops_gradients():
    """Reading a buffer blocks gradients through its stored value."""
    layer = Counter()
    buffers = layer.init_buffers()

    get_grad = jax.grad(lambda b: jnp.sum(b[layer]))(buffers)
    npt.assert_array_equal(jax.tree.leaves(get_grad)[0], jnp.zeros(2))


def test_set_stops_gradients():
    """Writing a buffer blocks gradients through the new value."""
    layer = Counter()
    buffers = layer.init_buffers()

    def through_update(value):
        updated = buffers.set(layer, value)
        return jnp.sum(updated[layer])

    npt.assert_array_equal(jax.grad(through_update)(jnp.ones(2)), jnp.zeros(2))


def test_wrong_model_raises():
    """Buffers initialized for another model cannot be read."""
    first = Counter()
    second = Counter()
    buffers = first.init_buffers()
    with pytest.raises(ValueError, match="init_buffers"):
        buffers[second]


def test_replaced_layer_raises():
    """Replacing a BufferModule invalidates its old payload."""
    layer = Counter()
    buffers = layer.init_buffers()

    replaced = layer.at.set(Counter())
    with pytest.raises(ValueError, match="init_buffers"):
        buffers[replaced]


def test_parameter_surgery_preserves_buffer_identity():
    """Editing parameters on an existing layer preserves buffer compatibility."""
    layer = nn.BatchNorm(2)
    buffers = layer.init_buffers()
    modified = layer.at.scale.set(nn.Param(jnp.full(2, 2.0)))
    npt.assert_array_equal(buffers[modified], buffers[layer])


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ((jnp.zeros(2), jnp.zeros(2)), "structure"),
        ({"value": jnp.zeros(2)}, "structure"),
        (jnp.zeros(3), "shape"),
        (jnp.zeros(2, dtype=jnp.int32), "dtype"),
    ],
)
def test_update_validation(value, message):
    """Updates must preserve payload structure, shape, and dtype."""
    layer = Counter()
    buffers = layer.init_buffers()
    with pytest.raises((TypeError, ValueError), match=message):
        buffers.set(layer, value)


def test_payload_must_be_a_nonempty_pytree_of_arrays():
    """A buffer payload must be a non-empty pytree containing only arrays."""

    class Invalid(nn.BufferModule):
        def _init_buffer(self, *, key=None):
            return {"bad": 1}

    with pytest.raises(TypeError, match="pytree of arrays"):
        Invalid().init_buffers()  # type: ignore[call-arg]


def test_buffer_module_requires_an_initializer():
    """A BufferModule subclass must implement _init_buffer."""
    with pytest.raises(NotImplementedError, match=r"must implement _init_buffer\(\)"):
        nn.BufferModule().init_buffers()  # type: ignore[call-arg]


def test_nested_buffer_modules_are_rejected():
    """A BufferModule cannot contain another BufferModule."""

    class Nested(nn.BufferModule):
        child: Counter

        def __init__(self):
            self.child = Counter()

        def _init_buffer(self, *, key=None):
            return jnp.zeros(1)

    with pytest.raises(ValueError, match="cannot contain"):
        Nested().init_buffers()


def test_checkpoint_roundtrip():
    """Checkpoint roundtrips restore parameters and buffer values."""
    layer = nn.BatchNorm(2)
    buffers = layer.init_buffers()
    x = jnp.array([[1.0, 3.0], [5.0, 7.0]])
    _, buffers = layer(x, buffers, training=True)
    changed = layer.at.scale.set(nn.Param(jnp.array([2.0, 3.0])))

    reference = nn.BatchNorm(2)
    reference_buffers = reference.init_buffers()
    with tempfile.NamedTemporaryFile(suffix=".ion") as file:
        ion.save(file.name, (changed, buffers))
        loaded, loaded_buffers = ion.load(file.name, (reference, reference_buffers))

    npt.assert_array_equal(loaded.scale._value, changed.scale._value)
    loaded_buffers = cast(_Buffers, loaded_buffers)
    npt.assert_array_equal(loaded_buffers[loaded], buffers[changed])


def test_custom_model_training_step():
    """A custom buffered model trains with buffers outside differentiation."""

    class Model(nn.Module):
        linear: nn.Linear
        norm: nn.BatchNorm
        dropout: nn.Dropout
        output: nn.SpectralNorm

        def __init__(self, *, key):
            keys = jax.random.split(key, 2)
            self.linear = nn.Linear(3, 4, key=keys[0])
            self.norm = nn.BatchNorm(4)
            self.dropout = nn.Dropout(0.2)
            self.output = nn.SpectralNorm(nn.Linear(4, 2, key=keys[1]))

        def __call__(self, x, buffers, *, training: bool, key=None):
            x = self.linear(x)
            x, buffers = self.norm(x, buffers, training=training)
            x = jax.nn.relu(x)
            x = self.dropout(x, training=training, key=key)
            x, buffers = self.output(x, buffers, training=training)
            return x, buffers

    model = Model(key=jax.random.key(0))
    buffers = model.init_buffers(key=jax.random.key(1))
    optimizer = ion.Optimizer(optax.adam(1e-3), model)
    x = jax.random.normal(jax.random.key(2), (8, 3))
    target = jax.random.normal(jax.random.key(3), (8, 2))

    @jax.jit
    def train_step(model, buffers, optimizer, x, target, key):
        def loss_fn(model):
            prediction, new_buffers = model(x, buffers, training=True, key=key)
            return jnp.mean(jnp.square(prediction - target)), new_buffers

        (loss, buffers), grads = jax.value_and_grad(loss_fn, has_aux=True)(model)
        model, optimizer = optimizer.update(model, grads)
        return model, buffers, optimizer, loss

    old_mean = buffers[model.norm][0]
    model, buffers, optimizer, loss = train_step(
        model, buffers, optimizer, x, target, jax.random.key(4)
    )
    prediction, returned = model(x, buffers, training=False)
    assert jnp.isfinite(loss)
    assert prediction.shape == target.shape
    assert optimizer.step == 1
    assert not jnp.array_equal(buffers[model.norm][0], old_mean)
    assert returned is buffers
