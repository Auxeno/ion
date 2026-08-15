from ion import gnn
import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest


class TestGATConv:
    def test_output_shape(self):
        """Output shape is (num_nodes, out_dim)."""
        gat = gnn.GATConv(8, 16, key=jax.random.key(0))
        x = jnp.ones((5, 8))
        senders = jnp.array([0, 1, 2, 3])
        receivers = jnp.array([1, 2, 3, 4])
        y = gat(x, senders, receivers)
        assert y.shape == (5, 16)

    def test_bipartite_output_shape(self):
        """Bipartite inputs can have distinct node counts and feature widths."""
        gat = gnn.GATConv((3, 5), 8, num_heads=2, key=jax.random.key(0))
        x_src = jax.random.normal(jax.random.key(1), (4, 3))
        x_dst = jax.random.normal(jax.random.key(2), (2, 5))
        senders = jnp.array([0, 3, 1])
        receivers = jnp.array([0, 0, 1])

        y = gat((x_src, x_dst), senders, receivers)

        assert y.shape == (2, 8)
        assert gat.w_sender.shape == (3, 8)
        assert gat.w_receiver is not None
        assert gat.w_receiver.shape == (5, 8)

    def test_output_shape_multi_head(self, triangle_graph):
        """Various num_heads values all produce correct output shape."""
        senders, receivers = triangle_graph
        for num_heads in [1, 2, 4, 8]:
            gat = gnn.GATConv(16, 16, num_heads=num_heads, key=jax.random.key(0))
            x = jnp.ones((3, 16))
            y = gat(x, senders, receivers)
            assert y.shape == (3, 16)

    def test_no_bias(self, triangle_graph):
        """No-bias mode: bias field is None, output still has correct shape."""
        gat = gnn.GATConv(8, 16, use_bias=False, key=jax.random.key(0))
        assert gat.b_out is None
        x = jnp.ones((3, 8))
        senders, receivers = triangle_graph
        y = gat(x, senders, receivers)
        assert y.shape == (3, 16)

    def test_glorot_init(self):
        """Glorot fans come from the flat (in_dim, out_dim) projection, including multi-head."""
        for num_heads in [1, 4]:
            gat = gnn.GATConv(256, 128, num_heads=num_heads, key=jax.random.key(42))
            var = jnp.var(gat.w_sender._value)
            expected_var = 2.0 / (256 + 128)
            npt.assert_allclose(var, expected_var, rtol=0.1)

    def test_zero_bias_init(self):
        """Bias is initialized to all zeros."""
        gat = gnn.GATConv(8, 16, key=jax.random.key(0))
        assert jnp.all(gat.b_out == 0)

    def test_default_dtype(self):
        """Weights default to float32."""
        gat = gnn.GATConv(8, 16, key=jax.random.key(0))
        assert gat.w_sender.dtype == jnp.float32
        assert gat.att_sender.dtype == jnp.float32
        assert gat.att_receiver.dtype == jnp.float32

    def test_projection_shapes(self):
        """Projection parameters stay flat; only activations gain a head axis."""
        gat = gnn.GATConv(8, 16, num_heads=4, edge_dim=3, key=jax.random.key(0))
        assert gat.w_sender.shape == (8, 16)
        assert gat.w_edge is not None
        assert gat.w_edge.shape == (3, 16)

    def test_attention_changes_with_features(self, triangle_graph):
        """Different node features produce different attention-weighted outputs."""
        gat = gnn.GATConv(4, 4, key=jax.random.key(0))
        senders, receivers = triangle_graph
        x1 = jax.random.normal(jax.random.key(1), (3, 4))
        x2 = jax.random.normal(jax.random.key(2), (3, 4))
        y1 = gat(x1, senders, receivers)
        y2 = gat(x2, senders, receivers)
        assert not jnp.allclose(y1, y2)

    def test_negative_slope(self):
        """Custom negative_slope is stored and used."""
        gat = gnn.GATConv(8, 8, negative_slope=0.1, key=jax.random.key(0))
        assert gat.negative_slope == 0.1

    def test_neighbour_influence_via_jacobian(self):
        """Connected nodes influence each other, disconnected nodes do not."""
        gat = gnn.GATConv(4, 4, key=jax.random.key(0))
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

    def test_single_node_self_loop(self):
        """Minimal graph: one node with a self-loop."""
        gat = gnn.GATConv(4, 8, num_heads=2, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (1, 4))
        senders = jnp.array([0])
        receivers = jnp.array([0])
        y = gat(x, senders, receivers)
        assert y.shape == (1, 8)
        assert jnp.all(jnp.isfinite(y))


class TestGATConvEdgeFeatures:
    def test_edge_dim_output_shape(self, triangle_graph):
        """With edge features, output shape is still (num_nodes, out_dim)."""
        senders, receivers = triangle_graph
        num_edges = senders.shape[0]
        gat = gnn.GATConv(8, 16, num_heads=2, edge_dim=4, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 8))
        x_edge = jax.random.normal(jax.random.key(2), (num_edges, 4))
        y = gat(x, senders, receivers, x_edge=x_edge)
        assert y.shape == (3, 16)

    def test_edge_dim_none_matches_no_edge(self, triangle_graph):
        """edge_dim=None produces identical output to omitting x_edge."""
        senders, receivers = triangle_graph
        gat = gnn.GATConv(8, 16, num_heads=2, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 8))
        y1 = gat(x, senders, receivers)
        y2 = gat(x, senders, receivers, x_edge=None)
        npt.assert_array_equal(y1, y2)

    def test_edge_features_change_output(self, triangle_graph):
        """Providing different edge features changes the output."""
        senders, receivers = triangle_graph
        num_edges = senders.shape[0]
        gat = gnn.GATConv(8, 16, num_heads=2, edge_dim=4, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 8))
        x_edge1 = jax.random.normal(jax.random.key(2), (num_edges, 4))
        x_edge2 = jax.random.normal(jax.random.key(3), (num_edges, 4))
        y1 = gat(x, senders, receivers, x_edge=x_edge1)
        y2 = gat(x, senders, receivers, x_edge=x_edge2)
        assert not jnp.allclose(y1, y2)

    def test_edge_dim_grad(self, triangle_graph):
        """Gradients flow through edge params."""
        senders, receivers = triangle_graph
        num_edges = senders.shape[0]
        gat = gnn.GATConv(8, 16, num_heads=2, edge_dim=4, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 8))
        x_edge = jax.random.normal(jax.random.key(2), (num_edges, 4))

        grads = jax.grad(lambda m: m(x, senders, receivers, x_edge=x_edge).sum())(gat)
        assert jnp.all(jnp.isfinite(grads.w_edge._value))
        assert jnp.all(jnp.isfinite(grads.att_edge._value))
        assert jnp.any(grads.w_edge._value != 0)
        assert jnp.any(grads.att_edge._value != 0)

    def test_edge_dim_frozen(self, triangle_graph):
        """Frozen edge params get zero gradients."""
        senders, receivers = triangle_graph
        num_edges = senders.shape[0]
        gat = gnn.GATConv(8, 16, num_heads=2, edge_dim=4, key=jax.random.key(0))
        frozen = gat.freeze()
        x = jax.random.normal(jax.random.key(1), (3, 8))
        x_edge = jax.random.normal(jax.random.key(2), (num_edges, 4))

        grads = jax.grad(lambda m: m(x, senders, receivers, x_edge=x_edge).sum())(frozen)
        npt.assert_allclose(grads.w_edge._value, jnp.zeros_like(grads.w_edge._value), atol=1e-7)
        npt.assert_allclose(grads.att_edge._value, jnp.zeros_like(grads.att_edge._value), atol=1e-7)

    def test_edge_dim_without_x_edge_raises(self, triangle_graph):
        """edge_dim set but x_edge omitted raises instead of silently skipping edge features."""
        senders, receivers = triangle_graph
        gat = gnn.GATConv(8, 16, num_heads=2, edge_dim=4, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 8))
        with pytest.raises(ValueError):
            gat(x, senders, receivers)

    def test_x_edge_without_edge_dim_raises(self, triangle_graph):
        """Passing x_edge to a layer without edge_dim raises."""
        senders, receivers = triangle_graph
        num_edges = senders.shape[0]
        gat = gnn.GATConv(8, 16, num_heads=2, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 8))
        x_edge = jax.random.normal(jax.random.key(2), (num_edges, 4))
        with pytest.raises(ValueError):
            gat(x, senders, receivers, x_edge=x_edge)

    def test_edge_jit(self, triangle_graph):
        """jax.jit with edge features produces the same output as eager."""
        senders, receivers = triangle_graph
        num_edges = senders.shape[0]
        gat = gnn.GATConv(8, 16, num_heads=2, edge_dim=4, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 8))
        x_edge = jax.random.normal(jax.random.key(2), (num_edges, 4))
        expected = gat(x, senders, receivers, x_edge=x_edge)
        result = jax.jit(gat)(x, senders, receivers, x_edge=x_edge)
        npt.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_edge_determinism(self, triangle_graph):
        """Same inputs with edge features produce identical outputs."""
        senders, receivers = triangle_graph
        num_edges = senders.shape[0]
        gat = gnn.GATConv(8, 16, num_heads=2, edge_dim=4, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 8))
        x_edge = jax.random.normal(jax.random.key(2), (num_edges, 4))
        y1 = gat(x, senders, receivers, x_edge=x_edge)
        y2 = gat(x, senders, receivers, x_edge=x_edge)
        npt.assert_array_equal(y1, y2)


class TestGATConvEdgeMask:
    def test_all_true_matches_no_mask(self, triangle_graph):
        """All-True mask produces the same output as no mask."""
        senders, receivers = triangle_graph
        gat = gnn.GATConv(8, 16, num_heads=2, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 8))
        mask = jnp.ones(senders.shape[0], dtype=bool)
        y_no_mask = gat(x, senders, receivers)
        y_masked = gat(x, senders, receivers, edge_mask=mask)
        npt.assert_allclose(y_masked, y_no_mask, rtol=1e-5, atol=1e-5)

    def test_masked_edge_no_influence(self):
        """Masked edges have zero influence on the output (verified via Jacobian)."""
        gat = gnn.GATConv(4, 4, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 4))
        # Edges: 0->1, 2->1. Mask out 2->1.
        senders = jnp.array([0, 2])
        receivers = jnp.array([1, 1])
        mask = jnp.array([True, False])
        jac = jax.jacobian(lambda x: gat(x, senders, receivers, edge_mask=mask))(x)
        jac_nodes = jnp.sum(jnp.abs(jac), axis=(1, 3))  # (n, n)
        # Node 2's edge is masked, so it should not influence node 1
        npt.assert_allclose(jac_nodes[1, 2], 0.0, atol=1e-5)
        # Node 0's edge is unmasked, so it should influence node 1
        assert jac_nodes[1, 0] > 1e-6

    def test_all_false_produces_zero(self):
        """All-False mask zeroes out all messages, producing zero output (no bias)."""
        gat = gnn.GATConv(4, 4, use_bias=False, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 4))
        senders = jnp.array([0, 1, 2])
        receivers = jnp.array([1, 2, 0])
        mask = jnp.zeros(3, dtype=bool)
        y = gat(x, senders, receivers, edge_mask=mask)
        npt.assert_allclose(y, jnp.zeros_like(y), atol=1e-6)

    def test_edge_mask_with_edge_features(self, triangle_graph):
        """Edge mask zeroes out edge feature contributions for masked edges."""
        senders, receivers = triangle_graph
        num_edges = senders.shape[0]
        gat = gnn.GATConv(8, 16, num_heads=2, edge_dim=4, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 8))
        x_edge = jax.random.normal(jax.random.key(2), (num_edges, 4))
        mask = jnp.ones(num_edges, dtype=bool).at[0].set(False)
        y_masked = gat(x, senders, receivers, x_edge=x_edge, edge_mask=mask)
        assert y_masked.shape == (3, 16)
        assert jnp.all(jnp.isfinite(y_masked))

    def test_masked_nonfinite_edge_features_do_not_leak(self):
        """Masked non-finite edge features cannot contaminate the output."""
        gat = gnn.GATConv(4, 4, edge_dim=2, use_bias=False, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 4))
        senders = jnp.array([0, 2])
        receivers = jnp.array([1, 1])
        x_edge = jnp.array([[1.0, -1.0], [jnp.inf, jnp.nan]])
        mask = jnp.array([True, False])

        result = gat(x, senders, receivers, x_edge=x_edge, edge_mask=mask)
        expected = gat(
            x,
            senders[:1],
            receivers[:1],
            x_edge=x_edge[:1],
            edge_mask=mask[:1],
        )

        assert jnp.all(jnp.isfinite(result))
        npt.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_edge_mask_grad(self, triangle_graph):
        """Gradients flow through edge_mask without NaN."""
        senders, receivers = triangle_graph
        gat = gnn.GATConv(8, 16, num_heads=2, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 8))
        mask = jnp.ones(senders.shape[0], dtype=bool).at[0].set(False)
        grads = jax.grad(lambda m: m(x, senders, receivers, edge_mask=mask).sum())(gat)
        assert jnp.all(jnp.isfinite(grads.w_sender._value))

    def test_edge_mask_jit(self, triangle_graph):
        """jax.jit with edge_mask produces same output as eager."""
        senders, receivers = triangle_graph
        gat = gnn.GATConv(8, 16, num_heads=2, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 8))
        mask = jnp.ones(senders.shape[0], dtype=bool).at[0].set(False)
        expected = gat(x, senders, receivers, edge_mask=mask)
        result = jax.jit(gat)(x, senders, receivers, edge_mask=mask)
        npt.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_fully_masked_receiver_zero(self):
        """A receiver whose incoming edges are all masked contributes zero."""
        conv = gnn.GATConv(4, 4, use_bias=False, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (4, 4))
        senders = jnp.array([0, 2, 1, 3])
        receivers = jnp.array([1, 1, 2, 2])
        mask = jnp.array([False, False, True, True])
        y = conv(x, senders, receivers, edge_mask=mask)
        npt.assert_allclose(y[1], 0.0, atol=1e-6)

    def test_fully_masked_receiver_other_nodes_unchanged(self):
        """Receivers with valid edges are unaffected by a fully masked neighbourhood."""
        conv = gnn.GATConv(4, 4, use_bias=False, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (4, 4))
        senders = jnp.array([0, 2, 1, 3])
        receivers = jnp.array([1, 1, 2, 2])
        mask = jnp.array([False, False, True, True])
        y = conv(x, senders, receivers, edge_mask=mask)
        y_full = conv(x, senders, receivers)
        npt.assert_allclose(y[2], y_full[2], atol=1e-6)

    def test_fully_masked_receiver_with_edge_features(self):
        """Edge features do not leak into a fully masked neighbourhood."""
        conv = gnn.GATConv(4, 4, edge_dim=3, use_bias=False, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (4, 4))
        senders = jnp.array([0, 2, 1, 3])
        receivers = jnp.array([1, 1, 2, 2])
        x_edge = jax.random.normal(jax.random.key(2), (4, 3))
        mask = jnp.array([False, False, True, True])
        y = conv(x, senders, receivers, x_edge=x_edge, edge_mask=mask)
        npt.assert_allclose(y[1], 0.0, atol=1e-6)

    def test_fully_masked_receiver_gradients_finite(self):
        """A fully masked neighbourhood has zero input gradient and no NaNs."""
        conv = gnn.GATConv(4, 4, use_bias=False, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (4, 4))
        senders = jnp.array([0, 2, 1, 3])
        receivers = jnp.array([1, 1, 2, 2])
        mask = jnp.array([False, False, True, True])
        grads = jax.grad(lambda x: conv(x, senders, receivers, edge_mask=mask)[1].sum())(x)
        npt.assert_array_equal(grads, 0.0)


class TestGATConvValidation:
    def test_out_dim_not_divisible_by_num_heads_raises(self):
        """out_dim must be divisible by num_heads."""
        with pytest.raises(ValueError, match="divisible"):
            gnn.GATConv(8, 7, num_heads=3, key=jax.random.key(0))


class TestGATv2Conv:
    def test_output_shape(self):
        """Output shape is (num_nodes, out_dim)."""
        gat = gnn.GATv2Conv(8, 16, key=jax.random.key(0))
        x = jnp.ones((5, 8))
        senders = jnp.array([0, 1, 2, 3])
        receivers = jnp.array([1, 2, 3, 4])
        y = gat(x, senders, receivers)
        assert y.shape == (5, 16)

    def test_bipartite_output_shape(self):
        """Bipartite inputs can have distinct node counts and feature widths."""
        gat = gnn.GATv2Conv((3, 5), 8, num_heads=2, key=jax.random.key(0))
        x_src = jax.random.normal(jax.random.key(1), (4, 3))
        x_dst = jax.random.normal(jax.random.key(2), (2, 5))
        senders = jnp.array([0, 3, 1])
        receivers = jnp.array([0, 0, 1])

        y = gat((x_src, x_dst), senders, receivers)

        assert y.shape == (2, 8)
        assert gat.w_sender.shape == (3, 8)
        assert gat.w_receiver.shape == (5, 8)

    def test_output_shape_multi_head(self, triangle_graph):
        """Various num_heads values all produce correct output shape."""
        senders, receivers = triangle_graph
        for num_heads in [1, 2, 4, 8]:
            gat = gnn.GATv2Conv(16, 16, num_heads=num_heads, key=jax.random.key(0))
            x = jnp.ones((3, 16))
            y = gat(x, senders, receivers)
            assert y.shape == (3, 16)

    def test_no_bias(self, triangle_graph):
        """No-bias mode: bias field is None, output still has correct shape."""
        gat = gnn.GATv2Conv(8, 16, use_bias=False, key=jax.random.key(0))
        assert gat.b_out is None
        x = jnp.ones((3, 8))
        senders, receivers = triangle_graph
        y = gat(x, senders, receivers)
        assert y.shape == (3, 16)

    def test_default_dtype(self):
        """Weights default to float32."""
        gat = gnn.GATv2Conv(8, 16, key=jax.random.key(0))
        assert gat.w_sender.dtype == jnp.float32
        assert gat.w_receiver.dtype == jnp.float32
        assert gat.att.dtype == jnp.float32

    def test_projection_shapes(self):
        """Projection parameters stay flat; only activations gain a head axis."""
        gat = gnn.GATv2Conv(8, 16, num_heads=4, edge_dim=3, key=jax.random.key(0))
        assert gat.w_sender.shape == (8, 16)
        assert gat.w_receiver.shape == (8, 16)
        assert gat.w_edge is not None
        assert gat.w_edge.shape == (3, 16)

    def test_glorot_init(self):
        """Glorot fans come from the flat (in_dim, out_dim) projection, including multi-head."""
        for num_heads in [1, 4]:
            gat = gnn.GATv2Conv(256, 128, num_heads=num_heads, key=jax.random.key(42))
            expected_var = 2.0 / (256 + 128)
            npt.assert_allclose(jnp.var(gat.w_sender._value), expected_var, rtol=0.1)
            npt.assert_allclose(jnp.var(gat.w_receiver._value), expected_var, rtol=0.1)

    def test_attention_changes_with_features(self, triangle_graph):
        """Different node features produce different attention-weighted outputs."""
        gat = gnn.GATv2Conv(4, 4, key=jax.random.key(0))
        senders, receivers = triangle_graph
        x1 = jax.random.normal(jax.random.key(1), (3, 4))
        x2 = jax.random.normal(jax.random.key(2), (3, 4))
        y1 = gat(x1, senders, receivers)
        y2 = gat(x2, senders, receivers)
        assert not jnp.allclose(y1, y2)

    def test_negative_slope(self):
        """Custom negative_slope is stored and used."""
        gat = gnn.GATv2Conv(8, 8, negative_slope=0.1, key=jax.random.key(0))
        assert gat.negative_slope == 0.1

    def test_single_node_self_loop(self):
        """Minimal graph: one node with a self-loop."""
        gat = gnn.GATv2Conv(4, 8, num_heads=2, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (1, 4))
        senders = jnp.array([0])
        receivers = jnp.array([0])
        y = gat(x, senders, receivers)
        assert y.shape == (1, 8)
        assert jnp.all(jnp.isfinite(y))

    def test_differs_from_gatv1(self, triangle_graph):
        """GATv2Conv produces different output than GATConv (dynamic vs static)."""
        senders, receivers = triangle_graph
        x = jax.random.normal(jax.random.key(1), (3, 8))
        v1 = gnn.GATConv(8, 16, num_heads=2, key=jax.random.key(0))
        v2 = gnn.GATv2Conv(8, 16, num_heads=2, key=jax.random.key(0))
        y1 = v1(x, senders, receivers)
        y2 = v2(x, senders, receivers)
        assert not jnp.allclose(y1, y2)


class TestGATv2ConvEdgeFeatures:
    def test_edge_dim_output_shape(self, triangle_graph):
        """With edge features, output shape is still (num_nodes, out_dim)."""
        senders, receivers = triangle_graph
        num_edges = senders.shape[0]
        gat = gnn.GATv2Conv(8, 16, num_heads=2, edge_dim=4, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 8))
        x_edge = jax.random.normal(jax.random.key(2), (num_edges, 4))
        y = gat(x, senders, receivers, x_edge=x_edge)
        assert y.shape == (3, 16)

    def test_edge_features_change_output(self, triangle_graph):
        """Providing different edge features changes the output."""
        senders, receivers = triangle_graph
        num_edges = senders.shape[0]
        gat = gnn.GATv2Conv(8, 16, num_heads=2, edge_dim=4, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 8))
        x_edge1 = jax.random.normal(jax.random.key(2), (num_edges, 4))
        x_edge2 = jax.random.normal(jax.random.key(3), (num_edges, 4))
        y1 = gat(x, senders, receivers, x_edge=x_edge1)
        y2 = gat(x, senders, receivers, x_edge=x_edge2)
        assert not jnp.allclose(y1, y2)

    def test_edge_dim_grad(self, triangle_graph):
        """Gradients flow through edge params."""
        senders, receivers = triangle_graph
        num_edges = senders.shape[0]
        gat = gnn.GATv2Conv(8, 16, num_heads=2, edge_dim=4, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 8))
        x_edge = jax.random.normal(jax.random.key(2), (num_edges, 4))

        grads = jax.grad(lambda m: m(x, senders, receivers, x_edge=x_edge).sum())(gat)
        assert jnp.all(jnp.isfinite(grads.w_edge._value))
        assert jnp.any(grads.w_edge._value != 0)

    def test_edge_dim_frozen(self, triangle_graph):
        """Frozen edge params get zero gradients."""
        senders, receivers = triangle_graph
        num_edges = senders.shape[0]
        gat = gnn.GATv2Conv(8, 16, num_heads=2, edge_dim=4, key=jax.random.key(0))
        frozen = gat.freeze()
        x = jax.random.normal(jax.random.key(1), (3, 8))
        x_edge = jax.random.normal(jax.random.key(2), (num_edges, 4))

        grads = jax.grad(lambda m: m(x, senders, receivers, x_edge=x_edge).sum())(frozen)
        npt.assert_allclose(grads.w_edge._value, jnp.zeros_like(grads.w_edge._value), atol=1e-7)

    def test_edge_dim_without_x_edge_raises(self, triangle_graph):
        """edge_dim set but x_edge omitted raises instead of silently skipping edge features."""
        senders, receivers = triangle_graph
        gat = gnn.GATv2Conv(8, 16, num_heads=2, edge_dim=4, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 8))
        with pytest.raises(ValueError):
            gat(x, senders, receivers)

    def test_x_edge_without_edge_dim_raises(self, triangle_graph):
        """Passing x_edge to a layer without edge_dim raises."""
        senders, receivers = triangle_graph
        num_edges = senders.shape[0]
        gat = gnn.GATv2Conv(8, 16, num_heads=2, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 8))
        x_edge = jax.random.normal(jax.random.key(2), (num_edges, 4))
        with pytest.raises(ValueError):
            gat(x, senders, receivers, x_edge=x_edge)

    def test_edge_jit(self, triangle_graph):
        """jax.jit with edge features produces the same output as eager."""
        senders, receivers = triangle_graph
        num_edges = senders.shape[0]
        gat = gnn.GATv2Conv(8, 16, num_heads=2, edge_dim=4, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 8))
        x_edge = jax.random.normal(jax.random.key(2), (num_edges, 4))
        expected = gat(x, senders, receivers, x_edge=x_edge)
        result = jax.jit(gat)(x, senders, receivers, x_edge=x_edge)
        npt.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_edge_determinism(self, triangle_graph):
        """Same inputs with edge features produce identical outputs."""
        senders, receivers = triangle_graph
        num_edges = senders.shape[0]
        gat = gnn.GATv2Conv(8, 16, num_heads=2, edge_dim=4, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 8))
        x_edge = jax.random.normal(jax.random.key(2), (num_edges, 4))
        y1 = gat(x, senders, receivers, x_edge=x_edge)
        y2 = gat(x, senders, receivers, x_edge=x_edge)
        npt.assert_array_equal(y1, y2)


class TestGATv2ConvEdgeMask:
    def test_all_true_matches_no_mask(self, triangle_graph):
        """All-True mask produces the same output as no mask."""
        senders, receivers = triangle_graph
        gat = gnn.GATv2Conv(8, 16, num_heads=2, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 8))
        mask = jnp.ones(senders.shape[0], dtype=bool)
        y_no_mask = gat(x, senders, receivers)
        y_masked = gat(x, senders, receivers, edge_mask=mask)
        npt.assert_allclose(y_masked, y_no_mask, rtol=1e-5, atol=1e-5)

    def test_masked_edge_no_influence(self):
        """Masked edges have zero influence on the output (verified via Jacobian)."""
        gat = gnn.GATv2Conv(4, 4, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 4))
        senders = jnp.array([0, 2])
        receivers = jnp.array([1, 1])
        mask = jnp.array([True, False])
        jac = jax.jacobian(lambda x: gat(x, senders, receivers, edge_mask=mask))(x)
        jac_nodes = jnp.sum(jnp.abs(jac), axis=(1, 3))
        npt.assert_allclose(jac_nodes[1, 2], 0.0, atol=1e-5)
        assert jac_nodes[1, 0] > 1e-6

    def test_all_false_produces_zero(self):
        """All-False mask zeroes out all messages, producing zero output (no bias)."""
        gat = gnn.GATv2Conv(4, 4, use_bias=False, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 4))
        senders = jnp.array([0, 1, 2])
        receivers = jnp.array([1, 2, 0])
        mask = jnp.zeros(3, dtype=bool)
        y = gat(x, senders, receivers, edge_mask=mask)
        npt.assert_allclose(y, jnp.zeros_like(y), atol=1e-6)

    def test_edge_mask_with_edge_features(self, triangle_graph):
        """Edge mask zeroes out edge feature contributions for masked edges."""
        senders, receivers = triangle_graph
        num_edges = senders.shape[0]
        gat = gnn.GATv2Conv(8, 16, num_heads=2, edge_dim=4, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 8))
        x_edge = jax.random.normal(jax.random.key(2), (num_edges, 4))
        mask = jnp.ones(num_edges, dtype=bool).at[0].set(False)
        y_masked = gat(x, senders, receivers, x_edge=x_edge, edge_mask=mask)
        assert y_masked.shape == (3, 16)
        assert jnp.all(jnp.isfinite(y_masked))

    def test_masked_nonfinite_edge_features_do_not_leak(self):
        """Masked non-finite edge features cannot contaminate the output."""
        gat = gnn.GATv2Conv(4, 4, edge_dim=2, use_bias=False, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 4))
        senders = jnp.array([0, 2])
        receivers = jnp.array([1, 1])
        x_edge = jnp.array([[1.0, -1.0], [jnp.inf, jnp.nan]])
        mask = jnp.array([True, False])

        result = gat(x, senders, receivers, x_edge=x_edge, edge_mask=mask)
        expected = gat(
            x,
            senders[:1],
            receivers[:1],
            x_edge=x_edge[:1],
            edge_mask=mask[:1],
        )

        assert jnp.all(jnp.isfinite(result))
        npt.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_edge_mask_grad(self, triangle_graph):
        """Gradients flow through edge_mask without NaN."""
        senders, receivers = triangle_graph
        gat = gnn.GATv2Conv(8, 16, num_heads=2, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 8))
        mask = jnp.ones(senders.shape[0], dtype=bool).at[0].set(False)
        grads = jax.grad(lambda m: m(x, senders, receivers, edge_mask=mask).sum())(gat)
        assert jnp.all(jnp.isfinite(grads.w_sender._value))

    def test_edge_mask_jit(self, triangle_graph):
        """jax.jit with edge_mask produces same output as eager."""
        senders, receivers = triangle_graph
        gat = gnn.GATv2Conv(8, 16, num_heads=2, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 8))
        mask = jnp.ones(senders.shape[0], dtype=bool).at[0].set(False)
        expected = gat(x, senders, receivers, edge_mask=mask)
        result = jax.jit(gat)(x, senders, receivers, edge_mask=mask)
        npt.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_fully_masked_receiver_zero(self):
        """A receiver whose incoming edges are all masked contributes zero."""
        conv = gnn.GATv2Conv(4, 4, use_bias=False, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (4, 4))
        senders = jnp.array([0, 2, 1, 3])
        receivers = jnp.array([1, 1, 2, 2])
        mask = jnp.array([False, False, True, True])
        y = conv(x, senders, receivers, edge_mask=mask)
        npt.assert_allclose(y[1], 0.0, atol=1e-6)

    def test_fully_masked_receiver_other_nodes_unchanged(self):
        """Receivers with valid edges are unaffected by a fully masked neighbourhood."""
        conv = gnn.GATv2Conv(4, 4, use_bias=False, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (4, 4))
        senders = jnp.array([0, 2, 1, 3])
        receivers = jnp.array([1, 1, 2, 2])
        mask = jnp.array([False, False, True, True])
        y = conv(x, senders, receivers, edge_mask=mask)
        y_full = conv(x, senders, receivers)
        npt.assert_allclose(y[2], y_full[2], atol=1e-6)

    def test_fully_masked_receiver_with_edge_features(self):
        """Edge features do not leak into a fully masked neighbourhood."""
        conv = gnn.GATv2Conv(4, 4, edge_dim=3, use_bias=False, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (4, 4))
        senders = jnp.array([0, 2, 1, 3])
        receivers = jnp.array([1, 1, 2, 2])
        x_edge = jax.random.normal(jax.random.key(2), (4, 3))
        mask = jnp.array([False, False, True, True])
        y = conv(x, senders, receivers, x_edge=x_edge, edge_mask=mask)
        npt.assert_allclose(y[1], 0.0, atol=1e-6)

    def test_fully_masked_receiver_gradients_finite(self):
        """A fully masked neighbourhood has zero input gradient and no NaNs."""
        conv = gnn.GATv2Conv(4, 4, use_bias=False, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (4, 4))
        senders = jnp.array([0, 2, 1, 3])
        receivers = jnp.array([1, 1, 2, 2])
        mask = jnp.array([False, False, True, True])
        grads = jax.grad(lambda x: conv(x, senders, receivers, edge_mask=mask)[1].sum())(x)
        npt.assert_array_equal(grads, 0.0)


class TestGATv2ConvValidation:
    def test_out_dim_not_divisible_by_num_heads_raises(self):
        """out_dim must be divisible by num_heads."""
        with pytest.raises(ValueError, match="divisible"):
            gnn.GATv2Conv(8, 7, num_heads=3, key=jax.random.key(0))


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
    x_out = gnn.segment_sum(messages * attention[..., None], receivers, n_dst).reshape(n_dst, -1)

    if conv.w_root is not None:
        root = x_dst @ conv.w_root
        if conv.w_beta is not None:
            gate_input = jnp.concatenate([root, x_out, root - x_out], axis=-1)
            beta = jax.nn.sigmoid(gate_input @ conv.w_beta)
            x_out = beta * root + (1 - beta) * x_out
        else:
            x_out = x_out + root
    if conv.b_out is not None:
        x_out = x_out + conv.b_out
    return x_out


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
