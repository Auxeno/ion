import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from ion import gnn


class TestGATConv:
    def test_output_shape(self):
        """Output shape is (num_nodes, out_dim)."""
        gat = gnn.GATConv(8, 16, key=jax.random.key(0))
        x = jnp.ones((5, 8))
        senders = jnp.array([0, 1, 2, 3])
        receivers = jnp.array([1, 2, 3, 4])
        y = gat(x, senders, receivers)
        assert y.shape == (5, 16)

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
        gat = gnn.GATConv(8, 16, bias=False, key=jax.random.key(0))
        assert gat.b is None
        x = jnp.ones((3, 8))
        senders, receivers = triangle_graph
        y = gat(x, senders, receivers)
        assert y.shape == (3, 16)

    def test_glorot_init(self):
        """Glorot uniform initialization gives var(w) close to 2/(fan_in+fan_out)."""
        gat = gnn.GATConv(2048, 2048, key=jax.random.key(42))
        var = jnp.var(gat.w._value)
        # Glorot uniform: var = 2 / (fan_in + fan_out), but for (i, h, k) shape
        # the effective fan_in=2048, fan_out=2048
        expected_var = 2.0 / (2048 + 2048)
        npt.assert_allclose(var, expected_var, atol=0.05)

    def test_zero_bias_init(self):
        """Bias is initialized to all zeros."""
        gat = gnn.GATConv(8, 16, key=jax.random.key(0))
        assert jnp.all(gat.b == 0)

    def test_weight_dtype(self):
        """Weights match the requested dtype."""
        gat = gnn.GATConv(8, 16, dtype=jnp.float32, key=jax.random.key(0))
        assert gat.w.dtype == jnp.float32
        assert gat.att_sender.dtype == jnp.float32
        assert gat.att_receiver.dtype == jnp.float32

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

    def test_neighbor_influence_via_jacobian(self):
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
        y = gat(x, senders, receivers, x_edge)
        assert y.shape == (3, 16)

    def test_edge_dim_none_matches_no_edge(self, triangle_graph):
        """edge_dim=None produces identical output to omitting x_edge."""
        senders, receivers = triangle_graph
        gat = gnn.GATConv(8, 16, num_heads=2, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 8))
        y1 = gat(x, senders, receivers)
        y2 = gat(x, senders, receivers, x_edge=None)
        npt.assert_allclose(y1, y2, rtol=0, atol=0)

    def test_edge_features_change_output(self, triangle_graph):
        """Providing different edge features changes the output."""
        senders, receivers = triangle_graph
        num_edges = senders.shape[0]
        gat = gnn.GATConv(8, 16, num_heads=2, edge_dim=4, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 8))
        x_edge1 = jax.random.normal(jax.random.key(2), (num_edges, 4))
        x_edge2 = jax.random.normal(jax.random.key(3), (num_edges, 4))
        y1 = gat(x, senders, receivers, x_edge1)
        y2 = gat(x, senders, receivers, x_edge2)
        assert not jnp.allclose(y1, y2)

    def test_edge_dim_grad(self, triangle_graph):
        """Gradients flow through edge params."""
        senders, receivers = triangle_graph
        num_edges = senders.shape[0]
        gat = gnn.GATConv(8, 16, num_heads=2, edge_dim=4, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 8))
        x_edge = jax.random.normal(jax.random.key(2), (num_edges, 4))

        grads = jax.grad(lambda m: m(x, senders, receivers, x_edge).sum())(gat)
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

        grads = jax.grad(lambda m: m(x, senders, receivers, x_edge).sum())(frozen)
        npt.assert_allclose(grads.w_edge._value, jnp.zeros_like(grads.w_edge._value), atol=1e-7)
        npt.assert_allclose(grads.att_edge._value, jnp.zeros_like(grads.att_edge._value), atol=1e-7)

    def test_edge_dim_without_x_edge(self, triangle_graph):
        """edge_dim set but x_edge omitted still produces valid output (ignores edge path)."""
        senders, receivers = triangle_graph
        gat = gnn.GATConv(8, 16, num_heads=2, edge_dim=4, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 8))
        y = gat(x, senders, receivers)
        assert y.shape == (3, 16)
        assert jnp.all(jnp.isfinite(y))

    def test_x_edge_without_edge_dim_raises(self, triangle_graph):
        """Passing x_edge to a layer without edge_dim raises."""
        senders, receivers = triangle_graph
        num_edges = senders.shape[0]
        gat = gnn.GATConv(8, 16, num_heads=2, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 8))
        x_edge = jax.random.normal(jax.random.key(2), (num_edges, 4))
        with pytest.raises(Exception):
            gat(x, senders, receivers, x_edge)

    def test_edge_jit(self, triangle_graph):
        """jax.jit with edge features produces the same output as eager."""
        senders, receivers = triangle_graph
        num_edges = senders.shape[0]
        gat = gnn.GATConv(8, 16, num_heads=2, edge_dim=4, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 8))
        x_edge = jax.random.normal(jax.random.key(2), (num_edges, 4))
        expected = gat(x, senders, receivers, x_edge)
        result = jax.jit(gat)(x, senders, receivers, x_edge)
        npt.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_edge_determinism(self, triangle_graph):
        """Same inputs with edge features produce identical outputs."""
        senders, receivers = triangle_graph
        num_edges = senders.shape[0]
        gat = gnn.GATConv(8, 16, num_heads=2, edge_dim=4, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 8))
        x_edge = jax.random.normal(jax.random.key(2), (num_edges, 4))
        y1 = gat(x, senders, receivers, x_edge)
        y2 = gat(x, senders, receivers, x_edge)
        npt.assert_allclose(y1, y2, rtol=0, atol=0)


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
        gat = gnn.GATConv(4, 4, bias=False, key=jax.random.key(0))
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
        y_masked = gat(x, senders, receivers, x_edge, edge_mask=mask)
        assert y_masked.shape == (3, 16)
        assert jnp.all(jnp.isfinite(y_masked))

    def test_edge_mask_grad(self, triangle_graph):
        """Gradients flow through edge_mask without NaN."""
        senders, receivers = triangle_graph
        gat = gnn.GATConv(8, 16, num_heads=2, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 8))
        mask = jnp.ones(senders.shape[0], dtype=bool).at[0].set(False)
        grads = jax.grad(lambda m: m(x, senders, receivers, edge_mask=mask).sum())(gat)
        assert jnp.all(jnp.isfinite(grads.w._value))

    def test_edge_mask_jit(self, triangle_graph):
        """jax.jit with edge_mask produces same output as eager."""
        senders, receivers = triangle_graph
        gat = gnn.GATConv(8, 16, num_heads=2, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 8))
        mask = jnp.ones(senders.shape[0], dtype=bool).at[0].set(False)
        expected = gat(x, senders, receivers, edge_mask=mask)
        result = jax.jit(gat)(x, senders, receivers, edge_mask=mask)
        npt.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


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
        gat = gnn.GATv2Conv(8, 16, bias=False, key=jax.random.key(0))
        assert gat.b is None
        x = jnp.ones((3, 8))
        senders, receivers = triangle_graph
        y = gat(x, senders, receivers)
        assert y.shape == (3, 16)

    def test_weight_dtype(self):
        """Weights match the requested dtype."""
        gat = gnn.GATv2Conv(8, 16, dtype=jnp.float32, key=jax.random.key(0))
        assert gat.w_sender.dtype == jnp.float32
        assert gat.w_receiver.dtype == jnp.float32
        assert gat.att.dtype == jnp.float32

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
        y = gat(x, senders, receivers, x_edge)
        assert y.shape == (3, 16)

    def test_edge_features_change_output(self, triangle_graph):
        """Providing different edge features changes the output."""
        senders, receivers = triangle_graph
        num_edges = senders.shape[0]
        gat = gnn.GATv2Conv(8, 16, num_heads=2, edge_dim=4, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 8))
        x_edge1 = jax.random.normal(jax.random.key(2), (num_edges, 4))
        x_edge2 = jax.random.normal(jax.random.key(3), (num_edges, 4))
        y1 = gat(x, senders, receivers, x_edge1)
        y2 = gat(x, senders, receivers, x_edge2)
        assert not jnp.allclose(y1, y2)

    def test_edge_dim_grad(self, triangle_graph):
        """Gradients flow through edge params."""
        senders, receivers = triangle_graph
        num_edges = senders.shape[0]
        gat = gnn.GATv2Conv(8, 16, num_heads=2, edge_dim=4, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 8))
        x_edge = jax.random.normal(jax.random.key(2), (num_edges, 4))

        grads = jax.grad(lambda m: m(x, senders, receivers, x_edge).sum())(gat)
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

        grads = jax.grad(lambda m: m(x, senders, receivers, x_edge).sum())(frozen)
        npt.assert_allclose(grads.w_edge._value, jnp.zeros_like(grads.w_edge._value), atol=1e-7)

    def test_edge_dim_without_x_edge(self, triangle_graph):
        """edge_dim set but x_edge omitted still produces valid output."""
        senders, receivers = triangle_graph
        gat = gnn.GATv2Conv(8, 16, num_heads=2, edge_dim=4, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 8))
        y = gat(x, senders, receivers)
        assert y.shape == (3, 16)
        assert jnp.all(jnp.isfinite(y))

    def test_x_edge_without_edge_dim_raises(self, triangle_graph):
        """Passing x_edge to a layer without edge_dim raises."""
        senders, receivers = triangle_graph
        num_edges = senders.shape[0]
        gat = gnn.GATv2Conv(8, 16, num_heads=2, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 8))
        x_edge = jax.random.normal(jax.random.key(2), (num_edges, 4))
        with pytest.raises(Exception):
            gat(x, senders, receivers, x_edge)

    def test_edge_jit(self, triangle_graph):
        """jax.jit with edge features produces the same output as eager."""
        senders, receivers = triangle_graph
        num_edges = senders.shape[0]
        gat = gnn.GATv2Conv(8, 16, num_heads=2, edge_dim=4, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 8))
        x_edge = jax.random.normal(jax.random.key(2), (num_edges, 4))
        expected = gat(x, senders, receivers, x_edge)
        result = jax.jit(gat)(x, senders, receivers, x_edge)
        npt.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_edge_determinism(self, triangle_graph):
        """Same inputs with edge features produce identical outputs."""
        senders, receivers = triangle_graph
        num_edges = senders.shape[0]
        gat = gnn.GATv2Conv(8, 16, num_heads=2, edge_dim=4, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 8))
        x_edge = jax.random.normal(jax.random.key(2), (num_edges, 4))
        y1 = gat(x, senders, receivers, x_edge)
        y2 = gat(x, senders, receivers, x_edge)
        npt.assert_allclose(y1, y2, rtol=0, atol=0)


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
        gat = gnn.GATv2Conv(4, 4, bias=False, key=jax.random.key(0))
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
        y_masked = gat(x, senders, receivers, x_edge, edge_mask=mask)
        assert y_masked.shape == (3, 16)
        assert jnp.all(jnp.isfinite(y_masked))

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


class TestGATv2ConvValidation:
    def test_out_dim_not_divisible_by_num_heads_raises(self):
        """out_dim must be divisible by num_heads."""
        with pytest.raises(ValueError, match="divisible"):
            gnn.GATv2Conv(8, 7, num_heads=3, key=jax.random.key(0))
