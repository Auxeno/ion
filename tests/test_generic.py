import tempfile

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy.testing as npt

import ion
from ion import nn


def test_batch_dims(layer_and_input):
    """Prepending batch dims to the input prepends them to the output."""
    layer, x = layer_and_input
    out_core = layer(x).shape

    x_1 = jnp.stack([x] * 2)
    assert layer(x_1).shape == (2, *out_core)

    x_2 = jnp.stack([x_1] * 3)
    assert layer(x_2).shape == (3, 2, *out_core)


def test_jit(layer_and_input):
    """jax.jit produces the same output as eager execution."""
    layer, x = layer_and_input
    expected = layer(x)
    result = jax.jit(lambda x: layer(x))(x)
    npt.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


def test_vmap(layer_and_input):
    """jax.vmap over a leading batch dim matches unbatched output."""
    layer, x = layer_and_input
    expected = layer(x)[None]
    result = jax.vmap(layer)(x[None])
    npt.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


def test_grad(layer_and_input):
    """jax.grad w.r.t. input produces finite gradients."""
    layer, x = layer_and_input
    if not jnp.issubdtype(x.dtype, jnp.floating):
        return  # Integer inputs (e.g. Embedding) are not differentiable
    g = jax.grad(lambda x: layer(x).sum())(x)
    assert jnp.all(jnp.isfinite(g))


def test_param_grad(layer_and_input):
    """tree.grad w.r.t. model params produces finite gradients."""
    layer, x = layer_and_input

    @ion.grad
    def loss_grad(model, x):
        return model(x).sum()

    grads = loss_grad(layer, x)
    leaves = jax.tree.leaves(ion.filter(grads, ion.is_float_array))
    for leaf in leaves:
        assert jnp.all(jnp.isfinite(leaf))


def test_jit_grad(layer_and_input):
    """Composing jax.jit with jax.grad produces finite gradients."""
    layer, x = layer_and_input
    if not jnp.issubdtype(x.dtype, jnp.floating):
        return  # Integer inputs (e.g. Embedding) are not differentiable
    grad_fn = jax.grad(lambda x: layer(x).sum())
    g = jax.jit(grad_fn)(x)
    assert jnp.all(jnp.isfinite(g))


def test_output_dtype(layer_and_input):
    """Output dtype matches input dtype."""
    layer, x = layer_and_input
    if not jnp.issubdtype(x.dtype, jnp.floating):
        return  # Integer inputs (e.g. Embedding) produce float outputs
    y = layer(x)
    assert y.dtype == x.dtype


def test_determinism(layer_and_input):
    """Same layer and input produce identical outputs across calls."""
    layer, x = layer_and_input
    y1 = layer(x)
    y2 = layer(x)
    npt.assert_allclose(y1, y2, rtol=0, atol=0)


def test_different_keys():
    """Different PRNG keys produce different weights."""
    l1 = nn.Linear(8, 8, key=jax.random.key(0))
    l2 = nn.Linear(8, 8, key=jax.random.key(1))
    assert not jnp.array_equal(l1.w.value, l2.w.value)


def test_pytree_roundtrip(layer_and_input):
    """Flatten then unflatten reconstructs the layer exactly."""
    layer, _ = layer_and_input
    leaves, treedef = jtu.tree_flatten(layer)
    reconstructed = jtu.tree_unflatten(treedef, leaves)
    orig_leaves = jtu.tree_leaves(layer)
    recon_leaves = jtu.tree_leaves(reconstructed)
    for a, b in zip(orig_leaves, recon_leaves):
        if isinstance(a, jnp.ndarray):
            npt.assert_allclose(a, b, rtol=0, atol=0)
        else:
            assert a == b


def _is_module(layer):
    """Check if layer is a Module (not a functools.partial wrapper)."""
    return isinstance(layer, nn.Module)


def test_params_property(layer_and_input):
    """`.params` returns only inexact (trainable) arrays."""
    layer, _ = layer_and_input
    if not _is_module(layer):
        return  # Wrapper (e.g. partial), skip
    if isinstance(
        layer,
        (
            nn.Identity,
            nn.MaxPool1d,
            nn.MaxPool2d,
            nn.AvgPool1d,
            nn.AvgPool2d,
            nn.Dropout,
        ),
    ):
        return  # No params expected
    params = layer.params
    leaves = jax.tree.leaves(params)
    assert len(leaves) > 0
    for leaf in leaves:
        assert ion.is_float_array(leaf)


def test_serialization(layer_and_input):
    """Serialize then deserialize produces identical outputs."""
    layer, x = layer_and_input
    if not _is_module(layer):
        return  # Wrapper (e.g. partial), skip
    with tempfile.NamedTemporaryFile(suffix=".npz") as f:
        ion.save(f.name, layer)
        loaded = ion.load(f.name, layer)
    y_orig = layer(x)
    y_loaded = loaded(x)
    npt.assert_allclose(y_orig, y_loaded, rtol=0, atol=0)
