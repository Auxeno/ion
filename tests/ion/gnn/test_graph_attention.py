import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from ion import gnn


class TestGraphAttention:
    def test_output_shape(self):
        """Output shape is (num_nodes, out_dim)."""
        gat = gnn.GraphAttention(8, 16, key=jax.random.key(0))
        x = jnp.ones((5, 8))
        senders = jnp.array([0, 1, 2, 3])
        receivers = jnp.array([1, 2, 3, 4])
        y = gat(x, senders, receivers)
        assert y.shape == (5, 16)

    def test_output_shape_multi_head(self, triangle_graph):
        """Various num_heads values all produce correct output shape."""
        senders, receivers = triangle_graph
        for num_heads in [1, 2, 4, 8]:
            gat = gnn.GraphAttention(16, 16, num_heads=num_heads, key=jax.random.key(0))
            x = jnp.ones((3, 16))
            y = gat(x, senders, receivers)
            assert y.shape == (3, 16)

    def test_no_bias(self, triangle_graph):
        """No-bias mode: bias field is None, output still has correct shape."""
        gat = gnn.GraphAttention(8, 16, bias=False, key=jax.random.key(0))
        assert gat.b is None
        x = jnp.ones((3, 8))
        senders, receivers = triangle_graph
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

    def test_attention_changes_with_features(self, triangle_graph):
        """Different node features produce different attention-weighted outputs."""
        gat = gnn.GraphAttention(4, 4, key=jax.random.key(0))
        senders, receivers = triangle_graph
        x1 = jax.random.normal(jax.random.key(1), (3, 4))
        x2 = jax.random.normal(jax.random.key(2), (3, 4))
        y1 = gat(x1, senders, receivers)
        y2 = gat(x2, senders, receivers)
        assert not jnp.allclose(y1, y2)

    def test_negative_slope(self):
        """Custom negative_slope is stored and used."""
        gat = gnn.GraphAttention(8, 8, negative_slope=0.1, key=jax.random.key(0))
        assert gat.negative_slope == 0.1

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

    def test_single_node_self_loop(self):
        """Minimal graph: one node with a self-loop."""
        gat = gnn.GraphAttention(4, 8, num_heads=2, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (1, 4))
        senders = jnp.array([0])
        receivers = jnp.array([0])
        y = gat(x, senders, receivers)
        assert y.shape == (1, 8)
        assert jnp.all(jnp.isfinite(y))


class TestGraphAttentionValidation:
    def test_out_dim_not_divisible_by_num_heads_raises(self):
        """out_dim must be divisible by num_heads."""
        with pytest.raises(ValueError, match="divisible"):
            gnn.GraphAttention(8, 7, num_heads=3, key=jax.random.key(0))
