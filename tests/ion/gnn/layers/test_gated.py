from ion import gnn
import jax
import jax.numpy as jnp
import numpy.testing as npt


class TestGatedGCNConv:
    def test_output_manual(self):
        """Node and edge outputs match the gated update equations."""
        conv = gnn.GatedGCNConv(4, 6, edge_dim=2, eps=1e-5, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 4))
        x_edge = jax.random.normal(jax.random.key(2), (3, 2))
        senders = jnp.array([0, 1, 2])
        receivers = jnp.array([1, 2, 0])

        edge_expected = (
            x_edge @ conv.w_edge
            + x[senders] @ conv.w_sender
            + x[receivers] @ conv.w_receiver
            + conv.b_edge  # type: ignore[operator]
        )
        gates = jax.nn.sigmoid(edge_expected)
        gate_sum = gnn.segment_sum(gates, receivers, 3)
        messages = gates * (x[senders] @ conv.w_neigh)
        neigh = gnn.segment_sum(messages, receivers, 3) / (gate_sum + conv.eps)
        node_expected = x @ conv.w_self + neigh + conv.b_node  # type: ignore[operator]

        node_result, edge_result = conv(x, senders, receivers, x_edge=x_edge)

        npt.assert_allclose(node_result, node_expected, rtol=1e-5, atol=1e-5)
        npt.assert_allclose(edge_result, edge_expected, rtol=1e-5, atol=1e-5)

    def test_bipartite_output_manual(self):
        """Bipartite node and edge outputs match the gated update equations."""
        conv = gnn.GatedGCNConv((3, 5), 7, edge_dim=2, key=jax.random.key(0))
        x_src = jax.random.normal(jax.random.key(1), (4, 3))
        x_dst = jax.random.normal(jax.random.key(2), (2, 5))
        x_edge = jax.random.normal(jax.random.key(3), (3, 2))
        senders = jnp.array([0, 3, 1])
        receivers = jnp.array([0, 0, 1])

        edge_expected = (
            x_edge @ conv.w_edge
            + x_src[senders] @ conv.w_sender
            + x_dst[receivers] @ conv.w_receiver
            + conv.b_edge  # type: ignore[operator]
        )
        gates = jax.nn.sigmoid(edge_expected)
        gate_sum = gnn.segment_sum(gates, receivers, 2)
        messages = gates * (x_src[senders] @ conv.w_neigh)
        neigh = gnn.segment_sum(messages, receivers, 2) / (gate_sum + conv.eps)
        node_expected = x_dst @ conv.w_self + neigh + conv.b_node  # type: ignore[operator]

        node_result, edge_result = conv((x_src, x_dst), senders, receivers, x_edge=x_edge)

        npt.assert_allclose(node_result, node_expected, rtol=1e-5, atol=1e-5)
        npt.assert_allclose(edge_result, edge_expected, rtol=1e-5, atol=1e-5)

    def test_output_shapes(self):
        """Outputs have one row per node and edge at the shared output width."""
        conv = gnn.GatedGCNConv(4, 6, edge_dim=2, key=jax.random.key(0))
        x = jnp.ones((5, 4))
        x_edge = jnp.ones((3, 2))
        senders = jnp.array([0, 1, 2])
        receivers = jnp.array([1, 2, 3])

        x_out, x_edge_out = conv(x, senders, receivers, x_edge=x_edge)

        assert x_out.shape == (5, 6)
        assert x_edge_out.shape == (3, 6)

    def test_no_bias(self):
        """use_bias=False removes both output biases."""
        conv = gnn.GatedGCNConv(4, 6, edge_dim=2, use_bias=False, key=jax.random.key(0))

        assert conv.b_node is None
        assert conv.b_edge is None

    def test_zero_bias_init(self):
        """Node and edge biases are initialized to zeros."""
        conv = gnn.GatedGCNConv(4, 6, edge_dim=2, key=jax.random.key(0))

        assert jnp.all(conv.b_node == 0)
        assert jnp.all(conv.b_edge == 0)

    def test_edge_features_change_outputs(self):
        """Edge features affect both the learned edges and gated node messages."""
        conv = gnn.GatedGCNConv(4, 6, edge_dim=2, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 4))
        senders = jnp.array([0, 1, 2])
        receivers = jnp.array([2, 2, 0])

        node_zero, edge_zero = conv(x, senders, receivers, x_edge=jnp.zeros((3, 2)))
        x_edge = jnp.array([[1.0, -1.0], [-1.0, 1.0], [0.5, 2.0]])
        node_edge, edge_edge = conv(x, senders, receivers, x_edge=x_edge)

        assert not jnp.allclose(node_zero[2], node_edge[2])
        assert not jnp.allclose(edge_zero, edge_edge)

    def test_isolated_node_gets_root_and_bias(self):
        """A node with no incoming edges receives only its root term and bias."""
        conv = gnn.GatedGCNConv(4, 6, edge_dim=2, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 4))
        senders = jnp.array([0])
        receivers = jnp.array([1])
        x_edge = jax.random.normal(jax.random.key(2), (1, 2))

        x_out, _ = conv(x, senders, receivers, x_edge=x_edge)
        expected = x[2] @ conv.w_self + conv.b_node  # type: ignore[operator]

        npt.assert_allclose(x_out[2], expected, rtol=1e-5, atol=1e-5)

    def test_default_dtype(self):
        """Weights default to float32."""
        conv = gnn.GatedGCNConv(4, 6, edge_dim=2, key=jax.random.key(0))

        assert conv.w_self.dtype == jnp.float32
        assert conv.w_edge.dtype == jnp.float32
