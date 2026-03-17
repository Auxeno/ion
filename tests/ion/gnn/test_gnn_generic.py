import tempfile

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy.testing as npt

import ion


def test_jit(gnn_layer_and_graph):
    """jax.jit produces the same output as eager execution."""
    layer, x, senders, receivers = gnn_layer_and_graph
    expected = layer(x, senders, receivers)
    result = jax.jit(layer)(x, senders, receivers)
    npt.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


def test_grad(gnn_layer_and_graph):
    """jax.grad w.r.t. input produces finite gradients."""
    layer, x, senders, receivers = gnn_layer_and_graph
    g = jax.grad(lambda x: layer(x, senders, receivers).sum())(x)
    assert jnp.all(jnp.isfinite(g))


def test_param_grad(gnn_layer_and_graph):
    """jax.grad w.r.t. model params produces finite gradients."""
    layer, x, senders, receivers = gnn_layer_and_graph
    grads = jax.grad(lambda m: m(x, senders, receivers).sum())(layer)
    for leaf in jax.tree.leaves(grads):
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.inexact):
            assert jnp.all(jnp.isfinite(leaf))


def test_jit_grad(gnn_layer_and_graph):
    """Composing jax.jit with jax.grad produces finite gradients."""
    layer, x, senders, receivers = gnn_layer_and_graph
    g = jax.jit(jax.grad(lambda x: layer(x, senders, receivers).sum()))(x)
    assert jnp.all(jnp.isfinite(g))


def test_frozen_params_get_zero_gradient(gnn_layer_and_graph):
    """Frozen layer produces zero gradients for all weights."""
    layer, x, senders, receivers = gnn_layer_and_graph
    frozen = layer.freeze()
    grads = jax.grad(lambda m: m(x, senders, receivers).sum())(frozen)
    for leaf in jax.tree.leaves(grads):
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.inexact):
            npt.assert_allclose(leaf, jnp.zeros_like(leaf), atol=1e-7)


def test_determinism(gnn_layer_and_graph):
    """Same inputs produce identical outputs across calls."""
    layer, x, senders, receivers = gnn_layer_and_graph
    y1 = layer(x, senders, receivers)
    y2 = layer(x, senders, receivers)
    npt.assert_allclose(y1, y2, rtol=0, atol=0)


def test_output_dtype(gnn_layer_and_graph):
    """Output dtype matches input dtype."""
    layer, x, senders, receivers = gnn_layer_and_graph
    y = layer(x, senders, receivers)
    assert y.dtype == x.dtype


def test_pytree_roundtrip(gnn_layer_and_graph):
    """Flatten then unflatten reconstructs the layer exactly."""
    layer, _, _, _ = gnn_layer_and_graph
    leaves, treedef = jtu.tree_flatten(layer)
    reconstructed = jtu.tree_unflatten(treedef, leaves)
    for a, b in zip(jtu.tree_leaves(layer), jtu.tree_leaves(reconstructed)):
        if isinstance(a, jnp.ndarray):
            npt.assert_allclose(a, b, rtol=0, atol=0)


def test_serialization(gnn_layer_and_graph):
    """Serialize then deserialize produces identical outputs."""
    layer, x, senders, receivers = gnn_layer_and_graph
    with tempfile.NamedTemporaryFile(suffix=".npz") as f:
        ion.save(f.name, layer)
        loaded = ion.load(f.name, layer)
    y_orig = layer(x, senders, receivers)
    y_loaded = loaded(x, senders, receivers)
    npt.assert_allclose(y_orig, y_loaded, rtol=0, atol=0)


def test_different_graph_different_output(gnn_layer_and_graph):
    """Changing the graph topology changes the output."""
    layer, _, _, _ = gnn_layer_and_graph
    x = jax.random.normal(jax.random.key(1), (4, 8))
    s1 = jnp.array([0, 1])
    r1 = jnp.array([1, 0])
    s2 = jnp.array([0, 2])
    r2 = jnp.array([2, 0])
    y1 = layer(x, s1, r1)
    y2 = layer(x, s2, r2)
    assert not jnp.array_equal(y1, y2)
