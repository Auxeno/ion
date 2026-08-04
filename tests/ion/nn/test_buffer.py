import tempfile

import jax
import jax.numpy as jnp
import numpy.testing as npt
import optax
import pytest
import treescope

import ion
from ion import nn


class Counter(nn.Module):
    count: nn.Buffer

    def __init__(self, size=2):
        self.count = nn.Buffer(jnp.zeros(size))

    def __call__(self, x, *, training: bool):
        if training:
            self.count.set(self.count.value + 1)
        return x + self.count.value


def test_buffers_contribute_no_leaves():
    """Buffers are invisible to leaf-based traversal, keeping them out of autodiff."""
    assert jax.tree.leaves(Counter()) == []
    assert len(jax.tree.leaves(nn.BatchNorm(2))) == 2


def test_buffers_are_discoverable_by_predicate():
    """Buffers are reachable with is_buffer, at their attribute paths."""
    model = nn.Sequential(nn.Linear(2, 3, key=jax.random.key(0)), nn.BatchNorm(3))
    leaves = jax.tree.leaves(model, is_leaf=ion.is_buffer)
    assert sum(ion.is_buffer(leaf) for leaf in leaves) == 2


def test_repr():
    """A buffer reports its dtype and shape, and renders in treescope."""
    assert repr(nn.Buffer(jnp.zeros(3))) == "Buffer(f32[3])"
    assert "running_mean=Buffer(f32[2])" in repr(nn.BatchNorm(2))
    assert "Buffer" in treescope.render_to_text(nn.BatchNorm(2))


def test_value_and_set():
    """A buffer reads back the value it was given, and set replaces it."""
    buffer = nn.Buffer(jnp.zeros(2))
    npt.assert_allclose(buffer.value, jnp.zeros(2))
    buffer.set(jnp.ones(2))
    npt.assert_allclose(buffer.value, jnp.ones(2))


def test_set_stops_gradients():
    """Writing a buffer blocks gradients through the new value."""
    layer = Counter()

    def through_update(value):
        layer.count.set(value)
        return jnp.sum(layer.count.value)

    npt.assert_allclose(jax.grad(through_update)(jnp.ones(2)), jnp.zeros(2))


def test_gradients_ignore_buffers():
    """Differentiating a stateful model yields a tree with no buffer entries."""
    model = nn.BatchNorm(2)
    x = jnp.array([[1.0, 3.0], [5.0, 7.0]])

    grads = jax.grad(lambda m: jnp.sum(m(x, training=True)))(model)
    assert jax.tree.structure(grads) == jax.tree.structure(model)


def test_jit():
    """Buffer updates cross the jit boundary."""
    layer = Counter()

    @jax.jit
    def step():
        return layer(jnp.zeros(2), training=True)

    step()
    npt.assert_allclose(layer.count.value, jnp.ones(2))


def test_scan():
    """Buffer updates accumulate through lax.scan."""
    layer = Counter()

    def body(carry, _):
        return carry, layer(jnp.zeros(2), training=True)

    jax.lax.scan(body, None, xs=None, length=3)
    npt.assert_allclose(layer.count.value, jnp.full(2, 3.0))


def test_vmap_raises():
    """Mutating a shared buffer under vmap is rejected rather than silently batched."""
    layer = Counter()

    with pytest.raises(Exception, match="array reference"):
        jax.vmap(lambda x: layer(x, training=True))(jnp.zeros((4, 2)))


@pytest.mark.parametrize("build", [lambda m: Counter(), lambda m: m.clone(), lambda m: m.freeze()])
def test_construction_inside_a_transform_raises(build):
    """Buffers reject traced values, since a reference built inside a trace dies with it."""
    layer = Counter()

    with pytest.raises(ValueError, match="inside a JAX transform"):
        jax.jit(build)(layer)


def test_tree_map_copies_share_state():
    """A plain tree.map copy shares buffers with the original."""
    layer = Counter()
    copy = jax.tree.map(lambda leaf: leaf, layer)

    copy(jnp.zeros(2), training=True)
    npt.assert_allclose(layer.count.value, jnp.ones(2))


@pytest.mark.parametrize(
    "derive", [lambda m: m.clone(), lambda m: m.freeze(), lambda m: m.unfreeze()]
)
def test_copies_own_their_buffers(derive):
    """clone, freeze and unfreeze all give the model they return its own buffers."""
    layer = Counter()
    derived = derive(layer)

    derived(jnp.zeros(2), training=True)
    npt.assert_allclose(layer.count.value, jnp.zeros(2))
    npt.assert_allclose(derived.count.value, jnp.ones(2))


def test_cast_shares_buffers():
    """astype shares buffers, which is what lets a cast inside a loss update the master."""
    layer = Counter()
    cast = layer.astype(jnp.bfloat16)

    cast(jnp.zeros(2), training=True)
    npt.assert_allclose(layer.count.value, jnp.ones(2))


def test_astype_leaves_buffer_dtype_alone():
    """Casting a model casts its params, while buffers keep the dtype their layer chose."""
    model = nn.BatchNorm(2).astype(jnp.bfloat16)

    assert model.scale.dtype == jnp.bfloat16
    assert model.running_mean.value.dtype == jnp.float32


def test_cast_inside_loss_updates_buffers():
    """The mixed-precision workflow casts inside the loss and still updates the master."""
    norm = nn.BatchNorm(8)
    model = nn.Sequential(nn.Linear(4, 8, key=jax.random.key(0)), norm)
    x = jax.random.normal(jax.random.key(1), (16, 4))

    def loss_fn(model, x):
        model = model.astype(jnp.bfloat16)
        return model(x.astype(jnp.bfloat16), training=True).sum().astype(jnp.float32)

    jax.grad(loss_fn)(model, x)

    assert not jnp.array_equal(norm.running_mean.value, jnp.zeros(8))
    assert norm.running_mean.value.dtype == jnp.float32


def test_optimizer_update_preserves_buffers():
    """An optimizer step carries buffer state forward, since it continues one model."""
    layer = Counter()
    optimizer = ion.Optimizer(optax.adam(1e-3), layer)

    layer(jnp.zeros(2), training=True)
    updated, _ = optimizer.update(layer, jax.tree.map(jnp.zeros_like, layer))
    npt.assert_allclose(updated.count.value, jnp.ones(2))


def test_sequential():
    """A stateful layer inside Sequential needs no routing and returns one value."""
    norm = nn.BatchNorm(3)
    model = nn.Sequential(nn.Linear(2, 3, key=jax.random.key(0)), norm)
    x = jnp.array([[1.0, 3.0], [5.0, 7.0]])

    output = model(x, training=True)
    assert output.shape == (2, 3)
    assert not jnp.array_equal(norm.running_mean.value, jnp.zeros(3))


def test_checkpoint_roundtrip():
    """Checkpoints carry buffer values, and loading leaves the reference model alone."""
    layer = nn.BatchNorm(2)
    x = jnp.array([[1.0, 3.0], [5.0, 7.0]])
    layer(x, training=True)
    changed = layer.at.scale.set(nn.Param(jnp.array([2.0, 3.0])))

    reference = nn.BatchNorm(2)
    with tempfile.NamedTemporaryFile(suffix=".ion") as file:
        ion.save(file.name, changed)
        loaded = ion.load(file.name, reference)

    npt.assert_allclose(loaded.scale._value, changed.scale._value)
    npt.assert_allclose(loaded.running_mean.value, layer.running_mean.value)
    npt.assert_allclose(reference.running_mean.value, jnp.zeros(2))


def test_custom_model_training_step():
    """A custom stateful model trains with buffers outside differentiation."""

    class Model(nn.Module):
        linear: nn.Linear
        norm: nn.BatchNorm
        dropout: nn.Dropout
        output: nn.SpectralNorm

        def __init__(self, *, key):
            keys = jax.random.split(key, 3)
            self.linear = nn.Linear(3, 4, key=keys[0])
            self.norm = nn.BatchNorm(4)
            self.dropout = nn.Dropout(0.2)
            self.output = nn.SpectralNorm(nn.Linear(4, 2, key=keys[1]), key=keys[2])

        def __call__(self, x, *, training: bool, key=None):
            x = self.linear(x)
            x = self.norm(x, training=training)
            x = jax.nn.relu(x)
            x = self.dropout(x, training=training, key=key)
            return self.output(x, training=training)

    model = Model(key=jax.random.key(0))
    optimizer = ion.Optimizer(optax.adam(1e-3), model)
    x = jax.random.normal(jax.random.key(2), (8, 3))
    target = jax.random.normal(jax.random.key(3), (8, 2))

    @jax.jit
    def train_step(model, optimizer, x, target, key):
        def loss_fn(model):
            prediction = model(x, training=True, key=key)
            return jnp.mean(jnp.square(prediction - target))

        loss, grads = jax.value_and_grad(loss_fn)(model)
        model, optimizer = optimizer.update(model, grads)
        return model, optimizer, loss

    old_mean = model.norm.running_mean.value
    model, optimizer, loss = train_step(model, optimizer, x, target, jax.random.key(4))
    prediction = model(x, training=False)

    assert jnp.isfinite(loss)
    assert prediction.shape == target.shape
    assert optimizer.step == 1
    assert not jnp.array_equal(model.norm.running_mean.value, old_mean)
