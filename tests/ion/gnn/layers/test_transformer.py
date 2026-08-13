import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from ion import gnn


def _manual_transformer(conv, x, senders, receivers, x_edge=None, edge_mask=None):
    """Reference implementation using the layer's parameters."""
    x_src, x_dst = x if isinstance(x, tuple) else (x, x)
    n_src, n_dst = x_src.shape[0], x_dst.shape[0]
    head_dim = conv.w_q.shape[-1] // conv.num_heads
    q = (x_dst @ conv.w_q).reshape(n_dst, conv.num_heads, head_dim)
    k = (x_src @ conv.w_k).reshape(n_src, conv.num_heads, head_dim)
    v = (x_src @ conv.w_v).reshape(n_src, conv.num_heads, head_dim)

    edge_k = k[senders]
    messages = v[senders]
    if x_edge is not None:
        e = (x_edge @ conv.w_edge).reshape(-1, conv.num_heads, head_dim)
        edge_k = edge_k + e
        messages = messages + e

    logits = jnp.sum(q[receivers] * edge_k, axis=-1) / jnp.sqrt(head_dim)
    if edge_mask is not None:
        logits = jnp.where(edge_mask[:, None], logits, -jnp.inf)
    attention = gnn.segment_softmax(logits, receivers, n_dst)
    out = gnn.segment_sum(messages * attention[..., None], receivers, n_dst).reshape(n_dst, -1)

    if conv.w_root is not None:
        root = x_dst @ conv.w_root
        if conv.w_beta is not None:
            gate_input = jnp.concatenate([root, out, root - out], axis=-1)
            beta = jax.nn.sigmoid(gate_input @ conv.w_beta)
            out = beta * root + (1 - beta) * out
        else:
            out = out + root
    if conv.b_out is not None:
        out = out + conv.b_out
    return out


class TestTransformerConv:
    def test_output_shape(self):
        """Output shape is (num_nodes, out_dim)."""
        conv = gnn.TransformerConv(8, 16, num_heads=2, key=jax.random.key(0))
        x = jnp.ones((5, 8))
        senders = jnp.array([0, 1, 2, 3])
        receivers = jnp.array([1, 2, 3, 4])
        assert conv(x, senders, receivers).shape == (5, 16)

    def test_output_shape_multi_head(self, triangle_graph):
        """All valid head counts preserve the requested output width."""
        x = jnp.ones((3, 16))
        senders, receivers = triangle_graph
        for num_heads in [1, 2, 4, 8]:
            conv = gnn.TransformerConv(16, 16, num_heads=num_heads, key=jax.random.key(0))
            assert conv(x, senders, receivers).shape == (3, 16)

    def test_output_manual(self, triangle_graph_no_self_loops):
        """Output matches scaled dot-product attention plus the root transform."""
        conv = gnn.TransformerConv(4, 8, num_heads=2, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 4))
        senders, receivers = triangle_graph_no_self_loops
        expected = _manual_transformer(conv, x, senders, receivers)
        y = conv(x, senders, receivers)
        npt.assert_allclose(y, expected, rtol=1e-5, atol=1e-5)

    def test_bipartite_output_manual(self):
        """Bipartite receivers query source keys and values."""
        conv = gnn.TransformerConv((3, 5), 8, num_heads=2, key=jax.random.key(0))
        x_src = jax.random.normal(jax.random.key(1), (4, 3))
        x_dst = jax.random.normal(jax.random.key(2), (2, 5))
        senders = jnp.array([0, 3, 1])
        receivers = jnp.array([0, 0, 1])

        x = (x_src, x_dst)
        expected = _manual_transformer(conv, x, senders, receivers)
        y = conv(x, senders, receivers)

        assert y.shape == (2, 8)
        npt.assert_allclose(y, expected, rtol=1e-5, atol=1e-5)

    def test_no_bias(self, triangle_graph_no_self_loops):
        """No-bias mode creates no bias parameter."""
        conv = gnn.TransformerConv(4, 8, use_bias=False, key=jax.random.key(0))
        assert conv.b_out is None
        x = jnp.ones((3, 4))
        senders, receivers = triangle_graph_no_self_loops
        assert conv(x, senders, receivers).shape == (3, 8)

    def test_no_root_weight(self, triangle_graph_no_self_loops):
        """use_root_weight=False removes the central-node projection."""
        conv = gnn.TransformerConv(4, 8, use_root_weight=False, key=jax.random.key(0))
        assert conv.w_root is None
        x = jax.random.normal(jax.random.key(1), (3, 4))
        senders, receivers = triangle_graph_no_self_loops
        expected = _manual_transformer(conv, x, senders, receivers)
        npt.assert_allclose(conv(x, senders, receivers), expected, rtol=1e-5, atol=1e-5)

    def test_beta_gate_manual(self, triangle_graph_no_self_loops):
        """use_beta=True gates between root and aggregated neighbourhood features."""
        conv = gnn.TransformerConv(4, 8, num_heads=2, use_beta=True, key=jax.random.key(0))
        assert conv.w_beta is not None
        x = jax.random.normal(jax.random.key(1), (3, 4))
        senders, receivers = triangle_graph_no_self_loops
        expected = _manual_transformer(conv, x, senders, receivers)
        npt.assert_allclose(conv(x, senders, receivers), expected, rtol=1e-5, atol=1e-5)

    def test_isolated_node_gets_root_and_bias(self):
        """A node without incoming edges receives only its root term and bias."""
        conv = gnn.TransformerConv(4, 8, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 4))
        senders = jnp.array([0])
        receivers = jnp.array([1])
        expected = x[2] @ conv.w_root + conv.b_out  # type: ignore[operator]
        npt.assert_allclose(conv(x, senders, receivers)[2], expected, rtol=1e-5, atol=1e-6)

    def test_single_node_self_loop(self):
        """A single node with a self-loop produces finite output."""
        conv = gnn.TransformerConv(4, 8, num_heads=2, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (1, 4))
        y = conv(x, jnp.array([0]), jnp.array([0]))
        assert y.shape == (1, 8)
        assert jnp.all(jnp.isfinite(y))

    def test_glorot_init(self):
        """Flat Q/K/V projections use the intended Glorot fans."""
        conv = gnn.TransformerConv(256, 128, num_heads=4, key=jax.random.key(42))
        expected_var = 2.0 / (256 + 128)
        npt.assert_allclose(jnp.var(conv.w_q._value), expected_var, rtol=0.1)
        npt.assert_allclose(jnp.var(conv.w_k._value), expected_var, rtol=0.1)
        npt.assert_allclose(jnp.var(conv.w_v._value), expected_var, rtol=0.1)

    def test_default_dtype(self):
        """Parameters default to float32."""
        conv = gnn.TransformerConv(8, 16, num_heads=2, use_beta=True, key=jax.random.key(0))
        assert conv.w_q.dtype == jnp.float32
        assert conv.w_k.dtype == jnp.float32
        assert conv.w_v.dtype == jnp.float32
        assert conv.w_root.dtype == jnp.float32  # type: ignore[union-attr]
        assert conv.w_beta.dtype == jnp.float32  # type: ignore[union-attr]

    def test_projection_shapes(self):
        """Projection parameters stay flat; heads exist only in activations."""
        conv = gnn.TransformerConv(
            8, 16, num_heads=4, edge_dim=3, use_beta=True, key=jax.random.key(0)
        )
        assert conv.w_q.shape == (8, 16)
        assert conv.w_k.shape == (8, 16)
        assert conv.w_v.shape == (8, 16)
        assert conv.w_edge.shape == (3, 16)  # type: ignore[union-attr]
        assert conv.w_root.shape == (8, 16)  # type: ignore[union-attr]
        assert conv.w_beta.shape == (48, 1)  # type: ignore[union-attr]


class TestTransformerConvEdgeFeatures:
    def test_output_manual(self, triangle_graph):
        """Edge features enter both keys and values."""
        conv = gnn.TransformerConv(4, 8, num_heads=2, edge_dim=3, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 4))
        senders, receivers = triangle_graph
        x_edge = jax.random.normal(jax.random.key(2), (senders.shape[0], 3))
        expected = _manual_transformer(conv, x, senders, receivers, x_edge=x_edge)
        y = conv(x, senders, receivers, x_edge=x_edge)
        npt.assert_allclose(y, expected, rtol=1e-5, atol=1e-5)

    def test_edge_features_change_output(self, triangle_graph):
        """Changing edge features changes attention and message values."""
        conv = gnn.TransformerConv(4, 8, num_heads=2, edge_dim=3, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 4))
        senders, receivers = triangle_graph
        x_edge_1 = jax.random.normal(jax.random.key(2), (senders.shape[0], 3))
        x_edge_2 = jax.random.normal(jax.random.key(3), (senders.shape[0], 3))
        y_1 = conv(x, senders, receivers, x_edge=x_edge_1)
        y_2 = conv(x, senders, receivers, x_edge=x_edge_2)
        assert not jnp.allclose(y_1, y_2)

    def test_edge_dim_without_x_edge_raises(self, triangle_graph):
        """Constructed edge dimensions require edge features at call time."""
        conv = gnn.TransformerConv(4, 8, edge_dim=3, key=jax.random.key(0))
        senders, receivers = triangle_graph
        with pytest.raises(ValueError, match="no x_edge"):
            conv(jnp.ones((3, 4)), senders, receivers)

    def test_x_edge_without_edge_dim_raises(self, triangle_graph):
        """Edge features cannot be passed without an edge projection."""
        conv = gnn.TransformerConv(4, 8, key=jax.random.key(0))
        senders, receivers = triangle_graph
        with pytest.raises(ValueError, match="edge_dim not set"):
            conv(jnp.ones((3, 4)), senders, receivers, x_edge=jnp.ones((9, 3)))


class TestTransformerConvEdgeMask:
    def test_all_true_matches_no_mask(self, triangle_graph_no_self_loops):
        """An all-True mask does not change the output."""
        conv = gnn.TransformerConv(4, 8, num_heads=2, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 4))
        senders, receivers = triangle_graph_no_self_loops
        mask = jnp.ones(senders.shape, dtype=bool)
        npt.assert_allclose(
            conv(x, senders, receivers, edge_mask=mask),
            conv(x, senders, receivers),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_all_false_removes_messages(self):
        """An all-False mask produces zero without a root term or bias."""
        conv = gnn.TransformerConv(
            4, 8, use_root_weight=False, use_bias=False, key=jax.random.key(0)
        )
        x = jax.random.normal(jax.random.key(1), (3, 4))
        senders = jnp.array([0, 1, 2])
        receivers = jnp.array([1, 2, 0])
        y = conv(x, senders, receivers, edge_mask=jnp.zeros(3, dtype=bool))
        npt.assert_allclose(y, jnp.zeros_like(y), atol=1e-6)

    def test_masked_output_manual(self, triangle_graph):
        """Masked attention matches the reference computation."""
        conv = gnn.TransformerConv(4, 8, num_heads=2, edge_dim=3, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 4))
        senders, receivers = triangle_graph
        x_edge = jax.random.normal(jax.random.key(2), (senders.shape[0], 3))
        mask = jnp.ones(senders.shape, dtype=bool).at[jnp.array([0, 3])].set(False)
        expected = _manual_transformer(conv, x, senders, receivers, x_edge=x_edge, edge_mask=mask)
        y = conv(x, senders, receivers, x_edge=x_edge, edge_mask=mask)
        npt.assert_allclose(y, expected, rtol=1e-5, atol=1e-5)

    def test_fully_masked_receiver_zero(self):
        """A receiver whose incoming edges are all masked contributes zero."""
        conv = gnn.TransformerConv(
            4, 4, use_root_weight=False, use_bias=False, key=jax.random.key(0)
        )
        x = jax.random.normal(jax.random.key(1), (4, 4))
        senders = jnp.array([0, 2, 1, 3])
        receivers = jnp.array([1, 1, 2, 2])
        mask = jnp.array([False, False, True, True])
        y = conv(x, senders, receivers, edge_mask=mask)
        npt.assert_allclose(y[1], 0.0, atol=1e-6)

    def test_fully_masked_receiver_other_nodes_unchanged(self):
        """Receivers with valid edges are unaffected by a fully masked neighbourhood."""
        conv = gnn.TransformerConv(
            4, 4, use_root_weight=False, use_bias=False, key=jax.random.key(0)
        )
        x = jax.random.normal(jax.random.key(1), (4, 4))
        senders = jnp.array([0, 2, 1, 3])
        receivers = jnp.array([1, 1, 2, 2])
        mask = jnp.array([False, False, True, True])
        y = conv(x, senders, receivers, edge_mask=mask)
        y_full = conv(x, senders, receivers)
        npt.assert_allclose(y[2], y_full[2], atol=1e-6)

    def test_fully_masked_receiver_with_edge_features(self):
        """Edge features do not leak into a fully masked neighbourhood."""
        conv = gnn.TransformerConv(
            4, 4, edge_dim=3, use_root_weight=False, use_bias=False, key=jax.random.key(0)
        )
        x = jax.random.normal(jax.random.key(1), (4, 4))
        senders = jnp.array([0, 2, 1, 3])
        receivers = jnp.array([1, 1, 2, 2])
        x_edge = jax.random.normal(jax.random.key(2), (4, 3))
        mask = jnp.array([False, False, True, True])
        y = conv(x, senders, receivers, x_edge=x_edge, edge_mask=mask)
        npt.assert_allclose(y[1], 0.0, atol=1e-6)

    def test_fully_masked_receiver_gradients_finite(self):
        """A fully masked neighbourhood has zero input gradient and no NaNs."""
        conv = gnn.TransformerConv(
            4, 4, use_root_weight=False, use_bias=False, key=jax.random.key(0)
        )
        x = jax.random.normal(jax.random.key(1), (4, 4))
        senders = jnp.array([0, 2, 1, 3])
        receivers = jnp.array([1, 1, 2, 2])
        mask = jnp.array([False, False, True, True])
        grads = jax.grad(lambda x: conv(x, senders, receivers, edge_mask=mask)[1].sum())(x)
        npt.assert_array_equal(grads, 0.0)


class TestTransformerConvValidation:
    def test_out_dim_not_divisible_by_num_heads_raises(self):
        """out_dim must split evenly across attention heads."""
        with pytest.raises(ValueError, match="divisible"):
            gnn.TransformerConv(8, 7, num_heads=3, key=jax.random.key(0))

    def test_beta_without_root_raises(self):
        """The beta gate requires a root representation."""
        with pytest.raises(ValueError, match="root_weight"):
            gnn.TransformerConv(8, 16, use_beta=True, use_root_weight=False, key=jax.random.key(0))
