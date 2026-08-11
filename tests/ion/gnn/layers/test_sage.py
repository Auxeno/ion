import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from ion import gnn


def _neighbor_agg(x, senders, receivers, num_nodes, aggregator):
    """Reference neighborhood pooling by dense scatter over receivers."""
    agg = []
    for r in range(num_nodes):
        neighbors = x[senders[receivers == r]]
        if neighbors.shape[0] == 0:
            agg.append(jnp.zeros(x.shape[1]))
        elif aggregator == "mean":
            agg.append(neighbors.mean(axis=0))
        elif aggregator == "sum":
            agg.append(neighbors.sum(axis=0))
        else:
            agg.append(neighbors.max(axis=0))
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

    @pytest.mark.parametrize("aggregator", ["mean", "max", "sum"])
    def test_output_manual(self, aggregator, triangle_graph_no_self_loops):
        """Output matches manual agg(neighbors) @ w_neigh + x @ w_self + b."""
        sage = gnn.SAGEConv(2, 3, aggregator=aggregator, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 2))
        senders, receivers = triangle_graph_no_self_loops

        neigh = _neighbor_agg(x, senders, receivers, 3, aggregator)
        expected = neigh @ sage.w_neigh + x @ sage.w_self + sage.b  # type: ignore[operator]

        y = sage(x, senders, receivers)
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
        """use_root_weight=False drops the self weight; output uses only neighbors."""
        sage = gnn.SAGEConv(2, 3, use_root_weight=False, key=jax.random.key(0))
        assert sage.w_self is None
        x = jax.random.normal(jax.random.key(1), (3, 2))
        senders, receivers = triangle_graph_no_self_loops

        neigh = _neighbor_agg(x, senders, receivers, 3, "mean")
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
        expected = x[2] @ sage.w_self + sage.b  # type: ignore[operator]
        npt.assert_allclose(y[2], expected, rtol=1e-5, atol=1e-6)
