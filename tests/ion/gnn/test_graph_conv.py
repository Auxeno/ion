import jax
import jax.numpy as jnp
import numpy.testing as npt

from ion import gnn


class TestGCNConv:
    def test_output_shape(self):
        """Output shape is (num_nodes, out_dim)."""
        gcn = gnn.GCNConv(8, 16, key=jax.random.key(0))
        x = jnp.ones((5, 8))
        senders = jnp.array([0, 1, 2, 3])
        receivers = jnp.array([1, 2, 3, 4])
        y = gcn(x, senders, receivers)
        assert y.shape == (5, 16)

    def test_output_manual(self, triangle_graph):
        """Output matches manual D^{-1/2} A D^{-1/2} (X W) + b computation."""
        gcn = gnn.GCNConv(2, 3, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 2))
        senders, receivers = triangle_graph

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

    def test_no_bias(self, triangle_graph):
        """No-bias mode: bias field is None, output still has correct shape."""
        gcn = gnn.GCNConv(8, 16, bias=False, key=jax.random.key(0))
        assert gcn.b is None
        x = jnp.ones((3, 8))
        senders, receivers = triangle_graph
        y = gcn(x, senders, receivers)
        assert y.shape == (3, 16)

    def test_he_normal_init(self):
        """He normal initialization gives var(w) close to 2/fan_in."""
        gcn = gnn.GCNConv(2048, 2048, key=jax.random.key(42))
        var = jnp.var(gcn.w._value)
        expected_var = 2.0 / 2048
        npt.assert_allclose(var, expected_var, atol=0.05)

    def test_zero_bias_init(self):
        """Bias is initialized to all zeros."""
        gcn = gnn.GCNConv(8, 16, key=jax.random.key(0))
        assert jnp.all(gcn.b == 0)

    def test_weight_dtype(self):
        """Weights match the requested dtype."""
        gcn = gnn.GCNConv(8, 16, dtype=jnp.float32, key=jax.random.key(0))
        assert gcn.w.dtype == jnp.float32

    def test_isolated_node_gets_only_bias(self):
        """A node with no incoming edges gets zero features (plus bias)."""
        gcn = gnn.GCNConv(4, 4, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 4))
        # Only edge: 0 -> 1 (node 2 is isolated)
        senders = jnp.array([0])
        receivers = jnp.array([1])
        y = gcn(x, senders, receivers)
        # Node 2 receives no messages, so output is just bias
        npt.assert_allclose(y[2], jnp.asarray(gcn.b), atol=1e-6)

    def test_self_loops_change_output(self, triangle_graph, triangle_graph_no_self_loops):
        """Adding self-loops changes the layer output."""
        gcn = gnn.GCNConv(4, 4, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 4))

        senders_no_sl, receivers_no_sl = triangle_graph_no_self_loops
        senders_sl, receivers_sl = triangle_graph

        y_no_sl = gcn(x, senders_no_sl, receivers_no_sl)
        y_sl = gcn(x, senders_sl, receivers_sl)
        assert not jnp.allclose(y_no_sl, y_sl)

    def test_single_node_self_loop(self):
        """Minimal graph: one node with a self-loop."""
        gcn = gnn.GCNConv(4, 8, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (1, 4))
        senders = jnp.array([0])
        receivers = jnp.array([0])
        y = gcn(x, senders, receivers)
        assert y.shape == (1, 8)
        assert jnp.all(jnp.isfinite(y))
