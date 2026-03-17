import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from ion import gnn


def _triangle_graph():
    """Undirected triangle (3 nodes, 6 edges) with self-loops."""
    senders = jnp.array([0, 1, 1, 2, 0, 2, 0, 1, 2])
    receivers = jnp.array([1, 0, 2, 1, 2, 0, 0, 1, 2])
    return senders, receivers


class TestGraphAttention:
    def test_output_shape(self):
        """Output shape is (num_nodes, out_dim)."""
        gat = gnn.GraphAttention(8, 16, key=jax.random.key(0))
        x = jnp.ones((5, 8))
        senders = jnp.array([0, 1, 2, 3])
        receivers = jnp.array([1, 2, 3, 4])
        y = gat(x, senders, receivers)
        assert y.shape == (5, 16)

    def test_output_shape_multi_head(self):
        """Various num_heads values all produce correct output shape."""
        for num_heads in [1, 2, 4, 8]:
            gat = gnn.GraphAttention(16, 16, num_heads=num_heads, key=jax.random.key(0))
            x = jnp.ones((3, 16))
            senders, receivers = _triangle_graph()
            y = gat(x, senders, receivers)
            assert y.shape == (3, 16)

    def test_no_bias(self):
        """No-bias mode: bias field is None, output still has correct shape."""
        gat = gnn.GraphAttention(8, 16, bias=False, key=jax.random.key(0))
        assert gat.b is None
        x = jnp.ones((3, 8))
        senders, receivers = _triangle_graph()
        y = gat(x, senders, receivers)
        assert y.shape == (3, 16)

    def test_glorot_init(self):
        """Glorot uniform initialization gives var(w) close to 2/(fan_in+fan_out)."""
        gat = gnn.GraphAttention(2048, 2048, key=jax.random.key(42))
        var = jnp.var(gat.w._value)
        # Glorot uniform: var = 2 / (fan_in + fan_out), but for (i, h, k) shape
        # the effective fan_in=2048, fan_out=2048
        expected_var = 2.0 / (2048 + 2048)
        npt.assert_allclose(var, expected_var, atol=0.05)

    def test_zero_bias_init(self):
        """Bias is initialized to all zeros."""
        gat = gnn.GraphAttention(8, 16, key=jax.random.key(0))
        assert jnp.all(gat.b == 0)

    def test_weight_dtype(self):
        """Weights match the requested dtype."""
        gat = gnn.GraphAttention(8, 16, dtype=jnp.float32, key=jax.random.key(0))
        assert gat.w.dtype == jnp.float32
        assert gat.att_sender.dtype == jnp.float32
        assert gat.att_receiver.dtype == jnp.float32

    def test_attention_changes_with_features(self):
        """Different node features produce different attention-weighted outputs."""
        gat = gnn.GraphAttention(4, 4, key=jax.random.key(0))
        senders, receivers = _triangle_graph()
        x1 = jax.random.normal(jax.random.key(1), (3, 4))
        x2 = jax.random.normal(jax.random.key(2), (3, 4))
        y1 = gat(x1, senders, receivers)
        y2 = gat(x2, senders, receivers)
        assert not jnp.allclose(y1, y2)

    def test_negative_slope(self):
        """Custom negative_slope is stored and used."""
        gat = gnn.GraphAttention(8, 8, negative_slope=0.1, key=jax.random.key(0))
        assert gat.negative_slope == 0.1

    def test_grad_wrt_input(self):
        """jax.grad w.r.t. node features produces finite gradients."""
        gat = gnn.GraphAttention(8, 16, num_heads=2, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 8))
        senders, receivers = _triangle_graph()
        g = jax.grad(lambda x: gat(x, senders, receivers).sum())(x)
        assert jnp.all(jnp.isfinite(g))

    def test_grad_wrt_params(self):
        """jax.grad w.r.t. model params produces finite gradients."""
        gat = gnn.GraphAttention(8, 16, num_heads=2, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 8))
        senders, receivers = _triangle_graph()
        grads = jax.grad(lambda m: m(x, senders, receivers).sum())(gat)
        for leaf in jax.tree.leaves(grads):
            if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.inexact):
                assert jnp.all(jnp.isfinite(leaf))

    def test_jit(self):
        """jax.jit produces the same output as eager execution."""
        gat = gnn.GraphAttention(8, 16, num_heads=2, key=jax.random.key(0))
        x = jnp.ones((3, 8))
        senders, receivers = _triangle_graph()
        expected = gat(x, senders, receivers)
        result = jax.jit(gat)(x, senders, receivers)
        npt.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_jit_grad(self):
        """Composing jax.jit with jax.grad produces finite gradients."""
        gat = gnn.GraphAttention(8, 16, num_heads=2, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 8))
        senders, receivers = _triangle_graph()
        g = jax.jit(jax.grad(lambda x: gat(x, senders, receivers).sum()))(x)
        assert jnp.all(jnp.isfinite(g))

    def test_frozen_params_get_zero_gradient(self):
        """Frozen GraphAttention layer produces zero gradients for its weights."""
        gat = gnn.GraphAttention(8, 16, num_heads=2, key=jax.random.key(0))
        gat = gat.freeze()
        x = jax.random.normal(jax.random.key(1), (3, 8))
        senders, receivers = _triangle_graph()
        grads = jax.grad(lambda m: m(x, senders, receivers).sum())(gat)
        npt.assert_allclose(grads.w._value, jnp.zeros_like(gat.w._value), atol=1e-7)
        npt.assert_allclose(
            grads.att_sender._value, jnp.zeros_like(gat.att_sender._value), atol=1e-7
        )
        npt.assert_allclose(
            grads.att_receiver._value, jnp.zeros_like(gat.att_receiver._value), atol=1e-7
        )

    def test_determinism(self):
        """Same inputs produce identical outputs across calls."""
        gat = gnn.GraphAttention(8, 16, num_heads=2, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 8))
        senders, receivers = _triangle_graph()
        y1 = gat(x, senders, receivers)
        y2 = gat(x, senders, receivers)
        npt.assert_allclose(y1, y2, rtol=0, atol=0)

    def test_different_graph_different_output(self):
        """Changing the graph topology changes the output."""
        gat = gnn.GraphAttention(4, 4, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (4, 4))
        s1 = jnp.array([0, 1])
        r1 = jnp.array([1, 0])
        s2 = jnp.array([0, 2])
        r2 = jnp.array([2, 0])
        y1 = gat(x, s1, r1)
        y2 = gat(x, s2, r2)
        assert not jnp.array_equal(y1, y2)

    def test_pytree_roundtrip(self):
        """Flatten then unflatten reconstructs the layer exactly."""
        gat = gnn.GraphAttention(8, 16, num_heads=2, key=jax.random.key(0))
        import jax.tree_util as jtu

        leaves, treedef = jtu.tree_flatten(gat)
        reconstructed = jtu.tree_unflatten(treedef, leaves)
        for a, b in zip(jtu.tree_leaves(gat), jtu.tree_leaves(reconstructed)):
            if isinstance(a, jnp.ndarray):
                npt.assert_allclose(a, b, rtol=0, atol=0)

    def test_single_node_self_loop(self):
        """Minimal graph: one node with a self-loop."""
        gat = gnn.GraphAttention(4, 8, num_heads=2, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (1, 4))
        senders = jnp.array([0])
        receivers = jnp.array([0])
        y = gat(x, senders, receivers)
        assert y.shape == (1, 8)
        assert jnp.all(jnp.isfinite(y))

    def test_neighbor_influence_via_jacobian(self):
        """Connected nodes influence each other, disconnected nodes do not."""
        gat = gnn.GraphAttention(4, 4, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 4))
        # Only edge: 0 -> 1 (node 2 is disconnected)
        senders = jnp.array([0])
        receivers = jnp.array([1])
        jac = jax.jacobian(lambda x: gat(x, senders, receivers))(x)
        # Node 2 should not influence node 1
        jac_nodes = jnp.sum(jnp.abs(jac), axis=(1, 3))  # (n, n)
        npt.assert_allclose(jac_nodes[1, 2], 0.0, atol=1e-5)
        # Node 0 should influence node 1
        assert jac_nodes[1, 0] > 1e-6


class TestGraphAttentionValidation:
    def test_out_dim_not_divisible_by_num_heads_raises(self):
        """out_dim must be divisible by num_heads."""
        with pytest.raises(ValueError, match="divisible"):
            gnn.GraphAttention(8, 7, num_heads=3, key=jax.random.key(0))
