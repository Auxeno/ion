import tempfile

import jax
import jax.numpy as jnp
import numpy.testing as npt

import ion


def _sum_output(output):
    return sum(leaf.sum() for leaf in jax.tree.leaves(output))


def _assert_allclose(actual, expected):
    for actual_leaf, expected_leaf in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        npt.assert_allclose(actual_leaf, expected_leaf, rtol=1e-5, atol=1e-5)


def test_jit(gnn_layer_and_graph):
    """jax.jit produces the same output as eager execution."""
    layer, x, senders, receivers, x_edge = gnn_layer_and_graph
    args = (senders, receivers)
    kwargs = {} if x_edge is None else {"x_edge": x_edge}
    expected = layer(x, *args, **kwargs)
    result = jax.jit(layer)(x, *args, **kwargs)
    _assert_allclose(result, expected)


def test_grad(gnn_layer_and_graph):
    """jax.grad w.r.t. input produces finite gradients."""
    layer, x, senders, receivers, x_edge = gnn_layer_and_graph
    args = (senders, receivers)
    kwargs = {} if x_edge is None else {"x_edge": x_edge}
    g = jax.grad(lambda x: _sum_output(layer(x, *args, **kwargs)))(x)
    assert jnp.all(jnp.isfinite(g))


def test_param_grad(gnn_layer_and_graph):
    """jax.grad w.r.t. model params produces finite gradients."""
    layer, x, senders, receivers, x_edge = gnn_layer_and_graph
    args = (senders, receivers)
    kwargs = {} if x_edge is None else {"x_edge": x_edge}
    grads = jax.grad(lambda m: _sum_output(m(x, *args, **kwargs)))(layer)
    for leaf in jax.tree.leaves(grads):
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.inexact):
            assert jnp.all(jnp.isfinite(leaf))


def test_jit_grad(gnn_layer_and_graph):
    """Composing jax.jit with jax.grad produces finite gradients."""
    layer, x, senders, receivers, x_edge = gnn_layer_and_graph
    args = (senders, receivers)
    kwargs = {} if x_edge is None else {"x_edge": x_edge}
    g = jax.jit(jax.grad(lambda x: _sum_output(layer(x, *args, **kwargs))))(x)
    assert jnp.all(jnp.isfinite(g))


def test_frozen_params_get_zero_gradient(gnn_layer_and_graph):
    """Frozen layer produces zero gradients for all weights."""
    layer, x, senders, receivers, x_edge = gnn_layer_and_graph
    args = (senders, receivers)
    kwargs = {} if x_edge is None else {"x_edge": x_edge}
    frozen = layer.freeze()
    grads = jax.grad(lambda m: _sum_output(m(x, *args, **kwargs)))(frozen)
    for leaf in jax.tree.leaves(grads):
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.inexact):
            npt.assert_allclose(leaf, jnp.zeros_like(leaf), atol=1e-7)


def test_determinism(gnn_layer_and_graph):
    """Same inputs produce identical outputs across calls."""
    layer, x, senders, receivers, x_edge = gnn_layer_and_graph
    args = (senders, receivers)
    kwargs = {} if x_edge is None else {"x_edge": x_edge}
    y1 = layer(x, *args, **kwargs)
    y2 = layer(x, *args, **kwargs)
    for y1_leaf, y2_leaf in zip(jax.tree.leaves(y1), jax.tree.leaves(y2)):
        npt.assert_array_equal(y1_leaf, y2_leaf)


def test_output_dtype(gnn_layer_and_graph):
    """Output dtype matches input dtype."""
    layer, x, senders, receivers, x_edge = gnn_layer_and_graph
    args = (senders, receivers)
    kwargs = {} if x_edge is None else {"x_edge": x_edge}
    y = layer(x, *args, **kwargs)
    assert all(leaf.dtype == x.dtype for leaf in jax.tree.leaves(y))


def test_pytree_roundtrip(gnn_layer_and_graph):
    """Flatten then unflatten reconstructs the layer exactly."""
    layer, _, _, _, _ = gnn_layer_and_graph
    leaves, treedef = jax.tree.flatten(layer)
    reconstructed = jax.tree.unflatten(treedef, leaves)
    for a, b in zip(jax.tree.leaves(layer), jax.tree.leaves(reconstructed)):
        if isinstance(a, jnp.ndarray):
            npt.assert_array_equal(a, b)


def test_serialization(gnn_layer_and_graph):
    """Serialize then deserialize produces identical outputs."""
    layer, x, senders, receivers, x_edge = gnn_layer_and_graph
    args = (senders, receivers)
    kwargs = {} if x_edge is None else {"x_edge": x_edge}
    with tempfile.NamedTemporaryFile(suffix=".npz") as f:
        ion.save(f.name, layer)
        loaded = ion.load(f.name, layer)
    y_orig = layer(x, *args, **kwargs)
    y_loaded = loaded(x, *args, **kwargs)
    for original_leaf, loaded_leaf in zip(jax.tree.leaves(y_orig), jax.tree.leaves(y_loaded)):
        npt.assert_array_equal(original_leaf, loaded_leaf)


def test_different_graph_different_output(gnn_layer_and_graph):
    """Changing the graph topology changes the output."""
    layer, _, _, _, x_edge = gnn_layer_and_graph
    x = jax.random.normal(jax.random.key(1), (4, 8))
    s1 = jnp.array([0, 1])
    r1 = jnp.array([1, 0])
    s2 = jnp.array([0, 2])
    r2 = jnp.array([2, 0])
    if x_edge is None:
        kwargs = {}
    else:
        kwargs = {"x_edge": jax.random.normal(jax.random.key(2), (2, x_edge.shape[1]))}
    y1 = layer(x, s1, r1, **kwargs)
    y2 = layer(x, s2, r2, **kwargs)
    assert any(
        not jnp.allclose(y1_leaf, y2_leaf)
        for y1_leaf, y2_leaf in zip(jax.tree.leaves(y1), jax.tree.leaves(y2))
    )


# bfloat16 tests


def test_bf16_output_dtype(gnn_layer_and_graph):
    """bfloat16 inputs produce bfloat16 outputs."""
    layer, x, senders, receivers, x_edge = gnn_layer_and_graph
    layer = layer.astype(jnp.bfloat16)
    x = x.astype(jnp.bfloat16)
    if x_edge is not None:
        x_edge = x_edge.astype(jnp.bfloat16)
    args = (senders, receivers)
    kwargs = {} if x_edge is None else {"x_edge": x_edge}
    y = layer(x, *args, **kwargs)
    assert all(leaf.dtype == jnp.bfloat16 for leaf in jax.tree.leaves(y))


def test_bf16_finiteness(gnn_layer_and_graph):
    """bfloat16 outputs are finite."""
    layer, x, senders, receivers, x_edge = gnn_layer_and_graph
    layer = layer.astype(jnp.bfloat16)
    x = x.astype(jnp.bfloat16)
    if x_edge is not None:
        x_edge = x_edge.astype(jnp.bfloat16)
    args = (senders, receivers)
    kwargs = {} if x_edge is None else {"x_edge": x_edge}
    y = layer(x, *args, **kwargs)
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(y))
