import jax
import jax.numpy as jnp
import numpy.testing as npt

from ion import gnn


def _triangle_graph():
    """Undirected triangle (3 nodes, 6 edges) with self-loops."""
    senders = jnp.array([0, 1, 1, 2, 0, 2, 0, 1, 2])
    receivers = jnp.array([1, 0, 2, 1, 2, 0, 0, 1, 2])
    return senders, receivers


def _triangle_graph_no_self_loops():
    """Undirected triangle (3 nodes, 6 edges) without self-loops."""
    senders = jnp.array([0, 1, 1, 2, 0, 2])
    receivers = jnp.array([1, 0, 2, 1, 2, 0])
    return senders, receivers


class TestGraphConv:
    def test_output_shape(self):
        """Output shape is (num_nodes, out_dim)."""
        gcn = gnn.GraphConv(8, 16, key=jax.random.key(0))
        x = jnp.ones((5, 8))
        senders = jnp.array([0, 1, 2, 3])
        receivers = jnp.array([1, 2, 3, 4])
        y = gcn(x, senders, receivers)
        assert y.shape == (5, 16)

    def test_output_manual(self):
        """Output matches manual D^{-1/2} A D^{-1/2} (X W) + b computation."""
        gcn = gnn.GraphConv(2, 3, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 2))
        senders, receivers = _triangle_graph()

        # Manual computation
        h = x @ gcn.w

        # Build adjacency and degree from senders/receivers
        num_nodes = 3
        adj = jnp.zeros((num_nodes, num_nodes))
        for s, r in zip(senders, receivers):
            adj = adj.at[int(r), int(s)].add(1.0)
        deg = adj.sum(axis=1)
        deg_inv_sqrt = jnp.where(deg > 0, 1.0 / jnp.sqrt(deg), 0.0)
        norm_adj = deg_inv_sqrt[:, None] * adj * deg_inv_sqrt[None, :]
        expected = norm_adj @ h + gcn.b  # type: ignore[operator]

        y = gcn(x, senders, receivers)
        npt.assert_allclose(y, expected, rtol=1e-5, atol=1e-5)

    def test_no_bias(self):
        """No-bias mode: bias field is None, output still has correct shape."""
        gcn = gnn.GraphConv(8, 16, bias=False, key=jax.random.key(0))
        assert gcn.b is None
        x = jnp.ones((3, 8))
        senders, receivers = _triangle_graph()
        y = gcn(x, senders, receivers)
        assert y.shape == (3, 16)

    def test_he_normal_init(self):
        """He normal initialization gives var(w) close to 2/fan_in."""
        gcn = gnn.GraphConv(2048, 2048, key=jax.random.key(42))
        var = jnp.var(gcn.w._value)
        expected_var = 2.0 / 2048
        npt.assert_allclose(var, expected_var, atol=0.05)

    def test_zero_bias_init(self):
        """Bias is initialized to all zeros."""
        gcn = gnn.GraphConv(8, 16, key=jax.random.key(0))
        assert jnp.all(gcn.b == 0)

    def test_weight_dtype(self):
        """Weights match the requested dtype."""
        gcn = gnn.GraphConv(8, 16, dtype=jnp.float32, key=jax.random.key(0))
        assert gcn.w.dtype == jnp.float32

    def test_isolated_node_gets_only_bias(self):
        """A node with no incoming edges gets zero features (plus bias)."""
        gcn = gnn.GraphConv(4, 4, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 4))
        # Only edge: 0 -> 1 (node 2 is isolated)
        senders = jnp.array([0])
        receivers = jnp.array([1])
        y = gcn(x, senders, receivers)
        # Node 2 receives no messages, so output is just bias
        npt.assert_allclose(y[2], jnp.asarray(gcn.b), atol=1e-6)

    def test_self_loops_change_output(self):
        """Adding self-loops changes the layer output."""
        gcn = gnn.GraphConv(4, 4, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 4))

        senders_no_sl, receivers_no_sl = _triangle_graph_no_self_loops()
        senders_sl, receivers_sl = _triangle_graph()

        y_no_sl = gcn(x, senders_no_sl, receivers_no_sl)
        y_sl = gcn(x, senders_sl, receivers_sl)
        assert not jnp.allclose(y_no_sl, y_sl)

    def test_grad_wrt_input(self):
        """jax.grad w.r.t. node features produces finite gradients."""
        gcn = gnn.GraphConv(8, 16, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (4, 8))
        senders = jnp.array([0, 1, 2, 3, 0, 1, 2, 3])
        receivers = jnp.array([1, 0, 3, 2, 0, 1, 2, 3])
        g = jax.grad(lambda x: gcn(x, senders, receivers).sum())(x)
        assert jnp.all(jnp.isfinite(g))

    def test_grad_wrt_params(self):
        """jax.grad w.r.t. model params produces finite gradients."""
        gcn = gnn.GraphConv(8, 16, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (4, 8))
        senders = jnp.array([0, 1, 2, 3, 0, 1, 2, 3])
        receivers = jnp.array([1, 0, 3, 2, 0, 1, 2, 3])
        grads = jax.grad(lambda m: m(x, senders, receivers).sum())(gcn)
        for leaf in jax.tree.leaves(grads):
            if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.inexact):
                assert jnp.all(jnp.isfinite(leaf))

    def test_jit(self):
        """jax.jit produces the same output as eager execution."""
        gcn = gnn.GraphConv(8, 16, key=jax.random.key(0))
        x = jnp.ones((3, 8))
        senders, receivers = _triangle_graph()
        expected = gcn(x, senders, receivers)
        result = jax.jit(gcn)(x, senders, receivers)
        npt.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_jit_grad(self):
        """Composing jax.jit with jax.grad produces finite gradients."""
        gcn = gnn.GraphConv(8, 16, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 8))
        senders, receivers = _triangle_graph()
        g = jax.jit(jax.grad(lambda x: gcn(x, senders, receivers).sum()))(x)
        assert jnp.all(jnp.isfinite(g))

    def test_frozen_params_get_zero_gradient(self):
        """Frozen GraphConv layer produces zero gradients for its weights."""
        gcn = gnn.GraphConv(8, 16, key=jax.random.key(0))
        gcn = gcn.freeze()
        x = jax.random.normal(jax.random.key(1), (3, 8))
        senders, receivers = _triangle_graph()
        grads = jax.grad(lambda m: m(x, senders, receivers).sum())(gcn)
        npt.assert_allclose(grads.w._value, jnp.zeros_like(gcn.w._value), atol=1e-7)

    def test_determinism(self):
        """Same inputs produce identical outputs across calls."""
        gcn = gnn.GraphConv(8, 16, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 8))
        senders, receivers = _triangle_graph()
        y1 = gcn(x, senders, receivers)
        y2 = gcn(x, senders, receivers)
        npt.assert_allclose(y1, y2, rtol=0, atol=0)

    def test_different_graph_different_output(self):
        """Changing the graph topology changes the output."""
        gcn = gnn.GraphConv(4, 4, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (4, 4))
        s1 = jnp.array([0, 1])
        r1 = jnp.array([1, 0])
        s2 = jnp.array([0, 2])
        r2 = jnp.array([2, 0])
        y1 = gcn(x, s1, r1)
        y2 = gcn(x, s2, r2)
        assert not jnp.array_equal(y1, y2)

    def test_pytree_roundtrip(self):
        """Flatten then unflatten reconstructs the layer exactly."""
        gcn = gnn.GraphConv(8, 16, key=jax.random.key(0))
        import jax.tree_util as jtu

        leaves, treedef = jtu.tree_flatten(gcn)
        reconstructed = jtu.tree_unflatten(treedef, leaves)
        for a, b in zip(jtu.tree_leaves(gcn), jtu.tree_leaves(reconstructed)):
            if isinstance(a, jnp.ndarray):
                npt.assert_allclose(a, b, rtol=0, atol=0)

    def test_single_node_self_loop(self):
        """Minimal graph: one node with a self-loop."""
        gcn = gnn.GraphConv(4, 8, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (1, 4))
        senders = jnp.array([0])
        receivers = jnp.array([0])
        y = gcn(x, senders, receivers)
        assert y.shape == (1, 8)
        assert jnp.all(jnp.isfinite(y))
