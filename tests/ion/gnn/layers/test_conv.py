from ion import gnn
import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest


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
        gcn = gnn.GCNConv(8, 16, use_bias=False, key=jax.random.key(0))
        assert gcn.b is None
        x = jnp.ones((3, 8))
        senders, receivers = triangle_graph
        y = gcn(x, senders, receivers)
        assert y.shape == (3, 16)

    def test_glorot_uniform_init(self):
        """Glorot uniform initialization gives var(w) close to 2/(fan_in + fan_out)."""
        gcn = gnn.GCNConv(2048, 2048, key=jax.random.key(42))
        var = jnp.var(gcn.w._value)
        expected_var = 2.0 / (2048 + 2048)
        npt.assert_allclose(var, expected_var, rtol=0.05)

    def test_zero_bias_init(self):
        """Bias is initialized to all zeros."""
        gcn = gnn.GCNConv(8, 16, key=jax.random.key(0))
        assert jnp.all(gcn.b == 0)

    def test_default_dtype(self):
        """Weights default to float32."""
        gcn = gnn.GCNConv(8, 16, key=jax.random.key(0))
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
        """Output matches sum(neighbours) @ w_neigh + x @ w_root + b."""
        conv = gnn.GraphConv(2, 3, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 2))
        senders, receivers = triangle_graph_no_self_loops

        neigh = jnp.stack([x[1] + x[2], x[0] + x[2], x[0] + x[1]])
        expected = neigh @ conv.w_neigh + x @ conv.w_root + conv.b  # type: ignore[operator]

        y = conv(x, senders, receivers)
        npt.assert_allclose(y, expected, rtol=1e-5, atol=1e-5)

    def test_bipartite_output_manual(self):
        """Bipartite inputs aggregate sources into the destination node set."""
        conv = gnn.GraphConv((2, 4), 3, key=jax.random.key(0))
        x_src = jax.random.normal(jax.random.key(1), (3, 2))
        x_dst = jax.random.normal(jax.random.key(2), (2, 4))
        senders = jnp.array([0, 2, 1])
        receivers = jnp.array([0, 0, 1])

        neigh = jnp.stack([x_src[0] + x_src[2], x_src[1]])
        expected = neigh @ conv.w_neigh + x_dst @ conv.w_root + conv.b  # type: ignore[operator]
        y = conv((x_src, x_dst), senders, receivers)

        assert y.shape == (2, 3)
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
        expected = neigh @ conv.w_neigh + x @ conv.w_root + conv.b  # type: ignore[operator]

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

        expected = x @ conv.w_root + conv.b  # type: ignore[operator]
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

        expected = x[2] @ conv.w_root + conv.b  # type: ignore[operator]
        y = conv(x, senders, receivers)
        npt.assert_allclose(y[2], expected, rtol=1e-5, atol=1e-6)

    def test_glorot_uniform_init(self):
        """Both weights use Glorot uniform initialization."""
        conv = gnn.GraphConv(2048, 2048, key=jax.random.key(42))
        expected_var = 2.0 / (2048 + 2048)
        npt.assert_allclose(jnp.var(conv.w_neigh._value), expected_var, rtol=0.05)
        npt.assert_allclose(jnp.var(conv.w_root._value), expected_var, rtol=0.05)

    def test_zero_bias_init(self):
        """Bias is initialized to all zeros."""
        conv = gnn.GraphConv(8, 16, key=jax.random.key(0))
        assert jnp.all(conv.b == 0)

    def test_default_dtype(self):
        """Parameters default to float32."""
        conv = gnn.GraphConv(8, 16, key=jax.random.key(0))
        assert conv.w_neigh.dtype == jnp.float32
        assert conv.w_root.dtype == jnp.float32

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


def _neighbour_aggregate(x, senders, receivers, num_nodes, aggregate):
    """Reference neighbourhood pooling by dense scatter over receivers."""
    agg = []
    for r in range(num_nodes):
        neighbours = x[senders[receivers == r]]
        if neighbours.shape[0] == 0:
            agg.append(jnp.zeros(x.shape[1]))
        elif aggregate == "mean":
            agg.append(neighbours.mean(axis=0))
        elif aggregate == "sum":
            agg.append(neighbours.sum(axis=0))
        else:
            agg.append(neighbours.max(axis=0))
    return jnp.stack(agg)


class TestSAGEConv:
    def test_output_shape(self):
        """Output shape is (num_nodes, out_dim)."""
        sage = gnn.SAGEConv(8, 16, key=jax.random.key(0))
        x = jnp.ones((5, 8))
        senders = jnp.array([0, 1, 2, 3])
        receivers = jnp.array([1, 2, 3, 4])
        y = sage(x, senders, receivers)
        assert y.shape == (5, 16)

    @pytest.mark.parametrize("aggregate", ["mean", "max", "sum"])
    def test_output_manual(self, aggregate, triangle_graph_no_self_loops):
        """Output matches manual aggregation plus root transform and bias."""
        sage = gnn.SAGEConv(2, 3, aggregate=aggregate, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 2))
        senders, receivers = triangle_graph_no_self_loops

        neigh = _neighbour_aggregate(x, senders, receivers, 3, aggregate)
        expected = neigh @ sage.w_neigh + x @ sage.w_root + sage.b  # type: ignore[operator]

        y = sage(x, senders, receivers)
        npt.assert_allclose(y, expected, rtol=1e-5, atol=1e-5)

    def test_bipartite_output_manual(self):
        """Bipartite inputs pool sources into the destination node set."""
        sage = gnn.SAGEConv((2, 4), 3, key=jax.random.key(0))
        x_src = jax.random.normal(jax.random.key(1), (3, 2))
        x_dst = jax.random.normal(jax.random.key(2), (2, 4))
        senders = jnp.array([0, 2, 1])
        receivers = jnp.array([0, 0, 1])

        neigh = jnp.stack([(x_src[0] + x_src[2]) / 2, x_src[1]])
        expected = neigh @ sage.w_neigh + x_dst @ sage.w_root + sage.b  # type: ignore[operator]
        y = sage((x_src, x_dst), senders, receivers)

        assert y.shape == (2, 3)
        npt.assert_allclose(y, expected, rtol=1e-5, atol=1e-5)

    def test_no_bias(self, triangle_graph):
        """No-bias mode: bias field is None, output still has correct shape."""
        sage = gnn.SAGEConv(8, 16, use_bias=False, key=jax.random.key(0))
        assert sage.b is None
        x = jnp.ones((3, 8))
        senders, receivers = triangle_graph
        y = sage(x, senders, receivers)
        assert y.shape == (3, 16)

    def test_no_root_weight(self, triangle_graph_no_self_loops):
        """use_root_weight=False drops the root weight; output uses only neighbours."""
        sage = gnn.SAGEConv(2, 3, use_root_weight=False, key=jax.random.key(0))
        assert sage.w_root is None
        x = jax.random.normal(jax.random.key(1), (3, 2))
        senders, receivers = triangle_graph_no_self_loops

        neigh = _neighbour_aggregate(x, senders, receivers, 3, "mean")
        expected = neigh @ sage.w_neigh + sage.b  # type: ignore[operator]

        y = sage(x, senders, receivers)
        npt.assert_allclose(y, expected, rtol=1e-5, atol=1e-5)

    def test_normalize_unit_norm(self, triangle_graph):
        """normalize=True gives each node embedding unit L2 norm."""
        sage = gnn.SAGEConv(8, 16, normalize=True, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 8))
        senders, receivers = triangle_graph
        y = sage(x, senders, receivers)
        npt.assert_allclose(jnp.linalg.norm(y, axis=-1), jnp.ones(3), rtol=1e-5)

    def test_glorot_uniform_init(self):
        """Glorot uniform initialization gives var(w) close to 2/(fan_in + fan_out)."""
        sage = gnn.SAGEConv(2048, 2048, key=jax.random.key(42))
        var = jnp.var(sage.w_neigh._value)
        expected_var = 2.0 / (2048 + 2048)
        npt.assert_allclose(var, expected_var, rtol=0.05)

    def test_zero_bias_init(self):
        """Bias is initialized to all zeros."""
        sage = gnn.SAGEConv(8, 16, key=jax.random.key(0))
        assert jnp.all(sage.b == 0)

    def test_default_dtype(self):
        """Weights default to float32."""
        sage = gnn.SAGEConv(8, 16, key=jax.random.key(0))
        assert sage.w_neigh.dtype == jnp.float32

    def test_isolated_node_gets_root_and_bias(self, triangle_graph_no_self_loops):
        """A node with no incoming edges gets only its root term plus bias."""
        sage = gnn.SAGEConv(4, 4, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 4))
        # Only edge: 0 -> 1 (nodes 0 and 2 receive nothing)
        senders = jnp.array([0])
        receivers = jnp.array([1])
        y = sage(x, senders, receivers)
        expected = x[2] @ sage.w_root + sage.b  # type: ignore[operator]
        npt.assert_allclose(y[2], expected, rtol=1e-5, atol=1e-6)
