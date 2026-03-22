import tempfile

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy.testing as npt
import pytest

import ion
from ion import nn


def test_batch_dims(layer_and_input):
    """Use jax.vmap to add extra batch dimensions."""
    layer, x = layer_and_input
    out = layer(x)

    # vmap adds an extra leading batch dim
    x_extra = jnp.stack([x] * 3)
    out_extra = jax.vmap(layer)(x_extra)
    assert out_extra.shape == (3, *out.shape)


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
    """jax.grad w.r.t. model params produces finite gradients."""
    layer, x = layer_and_input
    grads = jax.grad(lambda model: model(x).sum())(layer)
    for leaf in jax.tree.leaves(grads):
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.inexact):
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
    assert not jnp.array_equal(l1.w._value, l2.w._value)


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


def test_params_property(layer_and_input):
    """`.params` returns only inexact (trainable) arrays."""
    layer, _ = layer_and_input
    if isinstance(
        layer,
        (
            nn.Identity,
            nn.MaxPool,
            nn.AvgPool,
            nn.Dropout,
        ),
    ):
        return  # No params expected
    params = layer.params
    leaves = jax.tree.leaves(params)
    assert len(leaves) > 0
    for leaf in leaves:
        assert hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.inexact)


def test_wrong_rank_raises(structural_layer_and_input):
    """Structural layers reject inputs with wrong number of dimensions."""
    layer, x = structural_layer_and_input
    unbatched = x[0]
    with pytest.raises(Exception):
        layer(unbatched)


def test_wrong_rank_raises_under_jit(structural_layer_and_input):
    """Rank errors are caught even under JIT (at trace time)."""
    layer, x = structural_layer_and_input
    unbatched = x[0]
    with pytest.raises(Exception):
        jax.jit(layer)(unbatched)


def test_serialization(layer_and_input):
    """Serialize then deserialize produces identical outputs."""
    layer, x = layer_and_input
    with tempfile.NamedTemporaryFile(suffix=".npz") as f:
        ion.save(f.name, layer)
        loaded = ion.load(f.name, layer)
    y_orig = layer(x)
    y_loaded = loaded(x)
    npt.assert_allclose(y_orig, y_loaded, rtol=0, atol=0)


def _assert_trees_close(a, b, rtol=1e-5, atol=1e-5):
    """Assert all leaves of two pytrees are numerically close."""
    for leaf_a, leaf_b in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        npt.assert_allclose(leaf_a, leaf_b, rtol=rtol, atol=atol)


def test_seq_batch_dims(seq_layer_and_input):
    """jax.vmap adds an extra batch dimension."""
    layer, x = seq_layer_and_input
    y, _ = layer(x)
    x_extra = jnp.stack([x] * 3)
    y_extra, _ = jax.vmap(layer)(x_extra)
    assert y_extra.shape == (3, *y.shape)


def test_seq_jit(seq_layer_and_input):
    """jax.jit produces the same output as eager execution."""
    layer, x = seq_layer_and_input
    expected = layer(x)
    result = jax.jit(layer)(x)
    _assert_trees_close(result, expected)


def test_seq_vmap(seq_layer_and_input):
    """jax.vmap over a leading batch dim matches unbatched output."""
    layer, x = seq_layer_and_input
    y, _ = layer(x)
    y_vmap, _ = jax.vmap(layer)(x[None])
    npt.assert_allclose(y_vmap[0], y, rtol=1e-5, atol=1e-5)


def test_seq_grad(seq_layer_and_input):
    """jax.grad w.r.t. input produces finite gradients."""
    layer, x = seq_layer_and_input
    g = jax.grad(lambda x: layer(x)[0].sum())(x)
    assert jnp.all(jnp.isfinite(g))


def test_seq_param_grad(seq_layer_and_input):
    """jax.grad w.r.t. model params produces finite gradients."""
    layer, x = seq_layer_and_input
    grads = jax.grad(lambda model: model(x)[0].sum())(layer)
    for leaf in jax.tree.leaves(grads):
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.inexact):
            assert jnp.all(jnp.isfinite(leaf))


def test_seq_jit_grad(seq_layer_and_input):
    """Composing jax.jit with jax.grad produces finite gradients."""
    layer, x = seq_layer_and_input
    grad_fn = jax.grad(lambda x: layer(x)[0].sum())
    g = jax.jit(grad_fn)(x)
    assert jnp.all(jnp.isfinite(g))


def test_seq_output_dtype(seq_layer_and_input):
    """Output dtype matches input dtype."""
    layer, x = seq_layer_and_input
    y, _ = layer(x)
    assert y.dtype == x.dtype


def test_seq_determinism(seq_layer_and_input):
    """Same layer and input produce identical outputs across calls."""
    layer, x = seq_layer_and_input
    y1, _ = layer(x)
    y2, _ = layer(x)
    npt.assert_allclose(y1, y2, rtol=0, atol=0)


def test_seq_pytree_roundtrip(seq_layer_and_input):
    """Flatten then unflatten reconstructs the layer exactly."""
    layer, _ = seq_layer_and_input
    leaves, treedef = jtu.tree_flatten(layer)
    reconstructed = jtu.tree_unflatten(treedef, leaves)
    orig_leaves = jtu.tree_leaves(layer)
    recon_leaves = jtu.tree_leaves(reconstructed)
    for a, b in zip(orig_leaves, recon_leaves):
        if isinstance(a, jnp.ndarray):
            npt.assert_allclose(a, b, rtol=0, atol=0)
        else:
            assert a == b


def test_seq_serialization(seq_layer_and_input):
    """Serialize then deserialize produces identical outputs."""
    layer, x = seq_layer_and_input
    with tempfile.NamedTemporaryFile(suffix=".npz") as f:
        ion.save(f.name, layer)
        loaded = ion.load(f.name, layer)
    y_orig, _ = layer(x)
    y_loaded, _ = loaded(x)
    npt.assert_allclose(y_orig, y_loaded, rtol=0, atol=0)


def test_seq_params_property(seq_layer_and_input):
    """`.params` returns only inexact (trainable) arrays."""
    layer, _ = seq_layer_and_input
    params = layer.params
    leaves = jax.tree.leaves(params)
    assert len(leaves) > 0
    for leaf in leaves:
        assert hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.inexact)


def test_seq_wrong_rank_raises(seq_layer_and_input):
    """Sequence layers reject inputs with wrong number of dimensions."""
    layer, x = seq_layer_and_input
    unbatched = x[0]
    with pytest.raises(Exception):
        layer(unbatched)


def test_seq_wrong_rank_raises_under_jit(seq_layer_and_input):
    """Rank errors are caught even under JIT (at trace time)."""
    layer, x = seq_layer_and_input
    unbatched = x[0]
    with pytest.raises(Exception):
        jax.jit(layer)(unbatched)


def test_cell_jit(cell_and_input):
    """jax.jit produces the same output as eager execution."""
    cell, x = cell_and_input
    expected = cell(x, cell.initial_state)
    result = jax.jit(lambda x: cell(x, cell.initial_state))(x)
    _assert_trees_close(result, expected)


def test_cell_vmap(cell_and_input):
    """jax.vmap over a leading batch dim matches unbatched output."""
    cell, x = cell_and_input
    expected = cell(x, cell.initial_state)
    result = jax.vmap(lambda x: cell(x, cell.initial_state))(x[None])
    for a, b in zip(jax.tree.leaves(result), jax.tree.leaves(expected)):
        npt.assert_allclose(a[0], b, rtol=1e-5, atol=1e-5)


def test_cell_batch_dims(cell_and_input):
    """jax.vmap adds an extra batch dimension."""
    cell, x = cell_and_input
    out = cell(x, cell.initial_state)
    x_extra = jnp.stack([x] * 3)
    out_extra = jax.vmap(lambda x: cell(x, cell.initial_state))(x_extra)
    for a, b in zip(jax.tree.leaves(out_extra), jax.tree.leaves(out)):
        assert a.shape == (3, *b.shape)


def test_cell_grad(cell_and_input):
    """jax.grad w.r.t. input produces finite gradients."""
    cell, x = cell_and_input

    def loss(x):
        return sum(leaf.sum() for leaf in jax.tree.leaves(cell(x, cell.initial_state)))

    g = jax.grad(loss)(x)
    assert jnp.all(jnp.isfinite(g))


def test_cell_param_grad(cell_and_input):
    """jax.grad w.r.t. cell params produces finite gradients."""
    cell, x = cell_and_input

    def loss(cell):
        return sum(leaf.sum() for leaf in jax.tree.leaves(cell(x, cell.initial_state)))

    grads = jax.grad(loss)(cell)
    for leaf in jax.tree.leaves(grads):
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.inexact):
            assert jnp.all(jnp.isfinite(leaf))


def test_cell_jit_grad(cell_and_input):
    """Composing jax.jit with jax.grad produces finite gradients."""
    cell, x = cell_and_input

    def loss(x):
        return sum(leaf.sum() for leaf in jax.tree.leaves(cell(x, cell.initial_state)))

    g = jax.jit(jax.grad(loss))(x)
    assert jnp.all(jnp.isfinite(g))


def test_cell_output_dtype(cell_and_input):
    """Output dtype matches input dtype."""
    cell, x = cell_and_input
    result = cell(x, cell.initial_state)
    for leaf in jax.tree.leaves(result):
        assert leaf.dtype == x.dtype


def test_cell_determinism(cell_and_input):
    """Same cell and input produce identical outputs across calls."""
    cell, x = cell_and_input
    r1 = cell(x, cell.initial_state)
    r2 = cell(x, cell.initial_state)
    _assert_trees_close(r1, r2, rtol=0, atol=0)


def test_cell_pytree_roundtrip(cell_and_input):
    """Flatten then unflatten reconstructs the cell exactly."""
    cell, _ = cell_and_input
    leaves, treedef = jtu.tree_flatten(cell)
    reconstructed = jtu.tree_unflatten(treedef, leaves)
    orig_leaves = jtu.tree_leaves(cell)
    recon_leaves = jtu.tree_leaves(reconstructed)
    for a, b in zip(orig_leaves, recon_leaves):
        if isinstance(a, jnp.ndarray):
            npt.assert_allclose(a, b, rtol=0, atol=0)
        else:
            assert a == b


def test_cell_serialization(cell_and_input):
    """Serialize then deserialize produces identical outputs."""
    cell, x = cell_and_input
    with tempfile.NamedTemporaryFile(suffix=".npz") as f:
        ion.save(f.name, cell)
        loaded = ion.load(f.name, cell)
    expected = cell(x, cell.initial_state)
    result = loaded(x, loaded.initial_state)
    _assert_trees_close(result, expected, rtol=0, atol=0)


def test_cell_params_property(cell_and_input):
    """`.params` returns only inexact (trainable) arrays."""
    cell, _ = cell_and_input
    params = cell.params
    leaves = jax.tree.leaves(params)
    assert len(leaves) > 0
    for leaf in leaves:
        assert hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.inexact)


def test_ssm_batch_dims(ssm_layer_and_input):
    """jax.vmap adds an extra batch dimension."""
    layer, x = ssm_layer_and_input
    y, _ = layer(x)
    x_extra = jnp.stack([x] * 3)
    y_extra, _ = jax.vmap(layer)(x_extra)
    assert y_extra.shape == (3, *y.shape)


def test_ssm_jit(ssm_layer_and_input):
    """jax.jit produces the same output as eager execution."""
    layer, x = ssm_layer_and_input
    expected = layer(x)
    result = jax.jit(layer)(x)
    _assert_trees_close(result, expected)


def test_ssm_vmap(ssm_layer_and_input):
    """jax.vmap over a leading batch dim matches unbatched output."""
    layer, x = ssm_layer_and_input
    y, _ = layer(x)
    y_vmap, _ = jax.vmap(layer)(x[None])
    npt.assert_allclose(y_vmap[0], y, rtol=1e-5, atol=1e-5)


def test_ssm_grad(ssm_layer_and_input):
    """jax.grad w.r.t. input produces finite gradients."""
    layer, x = ssm_layer_and_input
    g = jax.grad(lambda x: layer(x)[0].sum())(x)
    assert jnp.all(jnp.isfinite(g))


def test_ssm_param_grad(ssm_layer_and_input):
    """jax.grad w.r.t. model params produces finite gradients."""
    layer, x = ssm_layer_and_input
    grads = jax.grad(lambda model: model(x)[0].sum())(layer)
    for leaf in jax.tree.leaves(grads):
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.inexact):
            assert jnp.all(jnp.isfinite(leaf))


def test_ssm_jit_grad(ssm_layer_and_input):
    """Composing jax.jit with jax.grad produces finite gradients."""
    layer, x = ssm_layer_and_input
    grad_fn = jax.grad(lambda x: layer(x)[0].sum())
    g = jax.jit(grad_fn)(x)
    assert jnp.all(jnp.isfinite(g))


def test_ssm_output_dtype(ssm_layer_and_input):
    """Real output dtype matches input dtype (state may be complex)."""
    layer, x = ssm_layer_and_input
    y, _ = layer(x)
    assert y.dtype == x.dtype


def test_ssm_determinism(ssm_layer_and_input):
    """Same layer and input produce identical outputs across calls."""
    layer, x = ssm_layer_and_input
    y1, _ = layer(x)
    y2, _ = layer(x)
    npt.assert_allclose(y1, y2, rtol=0, atol=0)


def test_ssm_pytree_roundtrip(ssm_layer_and_input):
    """Flatten then unflatten reconstructs the layer exactly."""
    layer, _ = ssm_layer_and_input
    leaves, treedef = jtu.tree_flatten(layer)
    reconstructed = jtu.tree_unflatten(treedef, leaves)
    orig_leaves = jtu.tree_leaves(layer)
    recon_leaves = jtu.tree_leaves(reconstructed)
    for a, b in zip(orig_leaves, recon_leaves):
        if isinstance(a, jnp.ndarray):
            npt.assert_allclose(a, b, rtol=0, atol=0)
        else:
            assert a == b


def test_ssm_serialization(ssm_layer_and_input):
    """Serialize then deserialize produces identical outputs."""
    layer, x = ssm_layer_and_input
    with tempfile.NamedTemporaryFile(suffix=".npz") as f:
        ion.save(f.name, layer)
        loaded = ion.load(f.name, layer)
    y_orig, _ = layer(x)
    y_loaded, _ = loaded(x)
    npt.assert_allclose(y_orig, y_loaded, rtol=0, atol=0)


def test_ssm_params_property(ssm_layer_and_input):
    """`.params` returns only inexact (trainable) arrays."""
    layer, _ = ssm_layer_and_input
    params = layer.params
    leaves = jax.tree.leaves(params)
    assert len(leaves) > 0
    for leaf in leaves:
        assert hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.inexact)


def test_ssm_wrong_rank_raises(ssm_layer_and_input):
    """SSM layers reject inputs with wrong number of dimensions."""
    layer, x = ssm_layer_and_input
    unbatched = x[0]
    with pytest.raises(Exception):
        layer(unbatched)


def test_ssm_wrong_rank_raises_under_jit(ssm_layer_and_input):
    """Rank errors are caught even under JIT (at trace time)."""
    layer, x = ssm_layer_and_input
    unbatched = x[0]
    with pytest.raises(Exception):
        jax.jit(layer)(unbatched)


def test_ssm_cell_jit(ssm_cell_and_input):
    """jax.jit produces the same output as eager execution."""
    cell, x = ssm_cell_and_input
    expected = cell(x, cell.initial_state)
    result = jax.jit(lambda x: cell(x, cell.initial_state))(x)
    _assert_trees_close(result, expected)


def test_ssm_cell_vmap(ssm_cell_and_input):
    """jax.vmap over a leading batch dim matches unbatched output."""
    cell, x = ssm_cell_and_input
    expected = cell(x, cell.initial_state)
    result = jax.vmap(lambda x: cell(x, cell.initial_state))(x[None])
    for a, b in zip(jax.tree.leaves(result), jax.tree.leaves(expected)):
        npt.assert_allclose(a[0], b, rtol=1e-5, atol=1e-5)


def test_ssm_cell_batch_dims(ssm_cell_and_input):
    """jax.vmap adds an extra batch dimension."""
    cell, x = ssm_cell_and_input
    out = cell(x, cell.initial_state)
    x_extra = jnp.stack([x] * 3)
    out_extra = jax.vmap(lambda x: cell(x, cell.initial_state))(x_extra)
    for a, b in zip(jax.tree.leaves(out_extra), jax.tree.leaves(out)):
        assert a.shape == (3, *b.shape)


def test_ssm_cell_grad(ssm_cell_and_input):
    """jax.grad w.r.t. input produces finite gradients."""
    cell, x = ssm_cell_and_input

    def loss(x):
        y, _ = cell(x, cell.initial_state)
        return y.sum()

    g = jax.grad(loss)(x)
    assert jnp.all(jnp.isfinite(g))


def test_ssm_cell_param_grad(ssm_cell_and_input):
    """jax.grad w.r.t. cell params produces finite gradients."""
    cell, x = ssm_cell_and_input

    def loss(cell):
        y, _ = cell(x, cell.initial_state)
        return y.sum()

    grads = jax.grad(loss)(cell)
    for leaf in jax.tree.leaves(grads):
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.inexact):
            assert jnp.all(jnp.isfinite(leaf))


def test_ssm_cell_jit_grad(ssm_cell_and_input):
    """Composing jax.jit with jax.grad produces finite gradients."""
    cell, x = ssm_cell_and_input

    def loss(x):
        y, _ = cell(x, cell.initial_state)
        return y.sum()

    g = jax.jit(jax.grad(loss))(x)
    assert jnp.all(jnp.isfinite(g))


def test_ssm_cell_output_dtype(ssm_cell_and_input):
    """Real output dtype matches input dtype (state is complex)."""
    cell, x = ssm_cell_and_input
    y, _ = cell(x, cell.initial_state)
    assert y.dtype == x.dtype


def test_ssm_cell_determinism(ssm_cell_and_input):
    """Same cell and input produce identical outputs across calls."""
    cell, x = ssm_cell_and_input
    r1 = cell(x, cell.initial_state)
    r2 = cell(x, cell.initial_state)
    _assert_trees_close(r1, r2, rtol=0, atol=0)


def test_ssm_cell_pytree_roundtrip(ssm_cell_and_input):
    """Flatten then unflatten reconstructs the cell exactly."""
    cell, _ = ssm_cell_and_input
    leaves, treedef = jtu.tree_flatten(cell)
    reconstructed = jtu.tree_unflatten(treedef, leaves)
    orig_leaves = jtu.tree_leaves(cell)
    recon_leaves = jtu.tree_leaves(reconstructed)
    for a, b in zip(orig_leaves, recon_leaves):
        if isinstance(a, jnp.ndarray):
            npt.assert_allclose(a, b, rtol=0, atol=0)
        else:
            assert a == b


def test_ssm_cell_serialization(ssm_cell_and_input):
    """Serialize then deserialize produces identical outputs."""
    cell, x = ssm_cell_and_input
    with tempfile.NamedTemporaryFile(suffix=".npz") as f:
        ion.save(f.name, cell)
        loaded = ion.load(f.name, cell)
    expected = cell(x, cell.initial_state)
    result = loaded(x, loaded.initial_state)
    _assert_trees_close(result, expected, rtol=0, atol=0)


def test_ssm_cell_params_property(ssm_cell_and_input):
    """`.params` returns only inexact (trainable) arrays."""
    cell, _ = ssm_cell_and_input
    params = cell.params
    leaves = jax.tree.leaves(params)
    assert len(leaves) > 0
    for leaf in leaves:
        assert hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.inexact)


# bfloat16 tests (standard layers, seq layers, cells -- not SSMs)


def _cast_bf16(layer, x):
    """Cast layer weights and float inputs to bfloat16."""
    layer = layer.astype(jnp.bfloat16)
    if jnp.issubdtype(x.dtype, jnp.floating):
        x = x.astype(jnp.bfloat16)
    return layer, x


def test_bf16_output_dtype(layer_and_input):
    """bfloat16 inputs produce bfloat16 outputs."""
    layer, x = layer_and_input
    layer, x = _cast_bf16(layer, x)
    if not jnp.issubdtype(x.dtype, jnp.floating):
        return  # Integer inputs (e.g. Embedding)
    y = layer(x)
    assert y.dtype == jnp.bfloat16


def test_bf16_finiteness(layer_and_input):
    """bfloat16 outputs are finite."""
    layer, x = layer_and_input
    layer, x = _cast_bf16(layer, x)
    y = layer(x)
    if jnp.issubdtype(y.dtype, jnp.floating):
        assert jnp.all(jnp.isfinite(y))


def test_seq_bf16_output_dtype(seq_layer_and_input):
    """bfloat16 inputs produce bfloat16 outputs for sequence layers."""
    layer, x = seq_layer_and_input
    layer, x = _cast_bf16(layer, x)
    h0 = jax.tree.map(
        lambda a: jnp.zeros((x.shape[0], *a.shape), dtype=jnp.bfloat16),
        layer.cell.initial_state,
    )
    y, _ = layer(x, hx=h0)
    assert y.dtype == jnp.bfloat16


def test_seq_bf16_finiteness(seq_layer_and_input):
    """bfloat16 outputs are finite for sequence layers."""
    layer, x = seq_layer_and_input
    layer, x = _cast_bf16(layer, x)
    h0 = jax.tree.map(
        lambda a: jnp.zeros((x.shape[0], *a.shape), dtype=jnp.bfloat16),
        layer.cell.initial_state,
    )
    y, _ = layer(x, hx=h0)
    assert jnp.all(jnp.isfinite(y))


def test_cell_bf16_output_dtype(cell_and_input):
    """bfloat16 inputs produce bfloat16 outputs for RNN cells."""
    cell, x = cell_and_input
    cell, x = _cast_bf16(cell, x)
    h0 = jax.tree.map(lambda a: a.astype(jnp.bfloat16), cell.initial_state)
    result = cell(x, h0)
    for leaf in jax.tree.leaves(result):
        assert leaf.dtype == jnp.bfloat16


def test_cell_bf16_finiteness(cell_and_input):
    """bfloat16 outputs are finite for RNN cells."""
    cell, x = cell_and_input
    cell, x = _cast_bf16(cell, x)
    h0 = jax.tree.map(lambda a: a.astype(jnp.bfloat16), cell.initial_state)
    result = cell(x, h0)
    for leaf in jax.tree.leaves(result):
        if jnp.issubdtype(leaf.dtype, jnp.floating):
            assert jnp.all(jnp.isfinite(leaf))
