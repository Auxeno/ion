import jax
import jax.numpy as jnp
import numpy.testing as npt

from ion import gnn


class TestGraphConv:
    def test_output_shape(self):
        """Output shape is (num_nodes, out_dim)."""
        conv = gnn.GraphConv(8, 16, key=jax.random.key(0))
        x = jnp.ones((5, 8))
        senders = jnp.array([0, 1, 2, 3])
        receivers = jnp.array([1, 2, 3, 4])
        y = conv(x, senders, receivers)
        assert y.shape == (5, 16)

    def test_output_manual(self, triangle_graph_no_self_loops):
        """Output matches sum(neighbours) @ w_neigh + x @ w_self + b."""
        conv = gnn.GraphConv(2, 3, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 2))
        senders, receivers = triangle_graph_no_self_loops

        neigh = jnp.stack([x[1] + x[2], x[0] + x[2], x[0] + x[1]])
        expected = neigh @ conv.w_neigh + x @ conv.w_self + conv.b  # type: ignore[operator]

        y = conv(x, senders, receivers)
        npt.assert_allclose(y, expected, rtol=1e-5, atol=1e-5)

    def test_edge_weight_manual(self):
        """Scalar edge weights scale sender features before aggregation."""
        conv = gnn.GraphConv(2, 3, key=jax.random.key(0))
        x = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        senders = jnp.array([0, 2, 1])
        receivers = jnp.array([1, 1, 2])
        edge_weight = jnp.array([0.5, 2.0, -1.0])

        neigh = jnp.stack(
            [
                jnp.zeros(2),
                0.5 * x[0] + 2.0 * x[2],
                -x[1],
            ]
        )
        expected = neigh @ conv.w_neigh + x @ conv.w_self + conv.b  # type: ignore[operator]

        y = conv(x, senders, receivers, edge_weight=edge_weight)
        npt.assert_allclose(y, expected, rtol=1e-5, atol=1e-5)

    def test_ones_edge_weight_matches_unweighted(self, triangle_graph_no_self_loops):
        """All-one edge weights reproduce ordinary sum aggregation."""
        conv = gnn.GraphConv(4, 8, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 4))
        senders, receivers = triangle_graph_no_self_loops
        edge_weight = jnp.ones(senders.shape)

        expected = conv(x, senders, receivers)
        y = conv(x, senders, receivers, edge_weight=edge_weight)
        npt.assert_array_equal(y, expected)

    def test_zero_edge_weight_leaves_root_and_bias(self, triangle_graph_no_self_loops):
        """Zero edge weights remove every neighbour contribution."""
        conv = gnn.GraphConv(4, 8, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 4))
        senders, receivers = triangle_graph_no_self_loops
        edge_weight = jnp.zeros(senders.shape)

        expected = x @ conv.w_self + conv.b  # type: ignore[operator]
        y = conv(x, senders, receivers, edge_weight=edge_weight)
        npt.assert_allclose(y, expected, rtol=1e-5, atol=1e-5)

    def test_no_bias(self, triangle_graph_no_self_loops):
        """No-bias mode has no bias parameter and keeps the output shape."""
        conv = gnn.GraphConv(8, 16, use_bias=False, key=jax.random.key(0))
        assert conv.b is None
        x = jnp.ones((3, 8))
        senders, receivers = triangle_graph_no_self_loops
        assert conv(x, senders, receivers).shape == (3, 16)

    def test_isolated_node_gets_root_and_bias(self):
        """A node with no incoming edges gets only its root term and bias."""
        conv = gnn.GraphConv(4, 4, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 4))
        senders = jnp.array([0])
        receivers = jnp.array([1])

        expected = x[2] @ conv.w_self + conv.b  # type: ignore[operator]
        y = conv(x, senders, receivers)
        npt.assert_allclose(y[2], expected, rtol=1e-5, atol=1e-6)

    def test_glorot_uniform_init(self):
        """Both weights use Glorot uniform initialization."""
        conv = gnn.GraphConv(2048, 2048, key=jax.random.key(42))
        expected_var = 2.0 / (2048 + 2048)
        npt.assert_allclose(jnp.var(conv.w_neigh._value), expected_var, rtol=0.05)
        npt.assert_allclose(jnp.var(conv.w_self._value), expected_var, rtol=0.05)

    def test_zero_bias_init(self):
        """Bias is initialized to all zeros."""
        conv = gnn.GraphConv(8, 16, key=jax.random.key(0))
        assert jnp.all(conv.b == 0)

    def test_default_dtype(self):
        """Parameters default to float32."""
        conv = gnn.GraphConv(8, 16, key=jax.random.key(0))
        assert conv.w_neigh.dtype == jnp.float32
        assert conv.w_self.dtype == jnp.float32

    def test_edge_weight_jit(self, triangle_graph_no_self_loops):
        """JIT compilation preserves weighted aggregation."""
        conv = gnn.GraphConv(4, 8, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 4))
        senders, receivers = triangle_graph_no_self_loops
        edge_weight = jax.random.normal(jax.random.key(2), senders.shape)

        expected = conv(x, senders, receivers, edge_weight=edge_weight)
        y = jax.jit(conv)(x, senders, receivers, edge_weight=edge_weight)
        npt.assert_allclose(y, expected, rtol=1e-5, atol=1e-5)

    def test_edge_weight_grad(self, triangle_graph_no_self_loops):
        """Gradients flow through scalar edge weights."""
        conv = gnn.GraphConv(4, 8, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 4))
        senders, receivers = triangle_graph_no_self_loops
        edge_weight = jax.random.normal(jax.random.key(2), senders.shape)

        grad = jax.grad(lambda weight: conv(x, senders, receivers, edge_weight=weight).sum())(
            edge_weight
        )
        assert jnp.all(jnp.isfinite(grad))
        assert jnp.any(grad != 0)
