import jax
import jax.numpy as jnp
import numpy.testing as npt

from ion import gnn


class TestMeanPool:
    def test_output_manual(self):
        """Output matches per-graph mean computed manually."""
        x = jax.random.normal(jax.random.key(0), (5, 3))
        graph_ids = jnp.array([0, 0, 0, 1, 1])
        result = gnn.mean_pool(x, graph_ids, num_graphs=2)
        assert result.shape == (2, 3)
        npt.assert_allclose(result[0], x[:3].mean(axis=0), rtol=1e-5, atol=1e-5)
        npt.assert_allclose(result[1], x[3:].mean(axis=0), rtol=1e-5, atol=1e-5)

    def test_grad_finite(self):
        """Gradients through mean_pool are finite."""
        x = jax.random.normal(jax.random.key(0), (5, 3))
        graph_ids = jnp.array([0, 0, 0, 1, 1])
        g = jax.grad(lambda x: gnn.mean_pool(x, graph_ids, 2).sum())(x)
        assert jnp.all(jnp.isfinite(g))


class TestSumPool:
    def test_output_manual(self):
        """Output matches per-graph sum computed manually."""
        x = jax.random.normal(jax.random.key(0), (5, 3))
        graph_ids = jnp.array([0, 0, 0, 1, 1])
        result = gnn.sum_pool(x, graph_ids, num_graphs=2)
        npt.assert_allclose(result[0], x[:3].sum(axis=0), rtol=1e-5, atol=1e-5)
        npt.assert_allclose(result[1], x[3:].sum(axis=0), rtol=1e-5, atol=1e-5)


class TestMaxPool:
    def test_output_manual(self):
        """Output matches per-graph max computed manually."""
        x = jax.random.normal(jax.random.key(0), (5, 3))
        graph_ids = jnp.array([0, 0, 0, 1, 1])
        result = gnn.max_pool(x, graph_ids, num_graphs=2)
        npt.assert_allclose(result[0], x[:3].max(axis=0), rtol=1e-5, atol=1e-5)
        npt.assert_allclose(result[1], x[3:].max(axis=0), rtol=1e-5, atol=1e-5)

    def test_empty_graph(self):
        """Graphs with no nodes give zeros, not -inf."""
        x = jnp.array([[1.0, -2.0], [3.0, -4.0]])
        graph_ids = jnp.array([0, 0])
        result = gnn.max_pool(x, graph_ids, num_graphs=2)
        npt.assert_allclose(result[1], jnp.zeros(2), atol=1e-6)
        assert jnp.all(jnp.isfinite(result))


class TestMinPool:
    def test_output_manual(self):
        """Output matches per-graph min computed manually."""
        x = jax.random.normal(jax.random.key(0), (5, 3))
        graph_ids = jnp.array([0, 0, 0, 1, 1])
        result = gnn.min_pool(x, graph_ids, num_graphs=2)
        npt.assert_allclose(result[0], x[:3].min(axis=0), rtol=1e-5, atol=1e-5)
        npt.assert_allclose(result[1], x[3:].min(axis=0), rtol=1e-5, atol=1e-5)

    def test_empty_graph(self):
        """Graphs with no nodes give zeros, not +inf."""
        x = jnp.array([[1.0, -2.0], [3.0, -4.0]])
        graph_ids = jnp.array([0, 0])
        result = gnn.min_pool(x, graph_ids, num_graphs=2)
        npt.assert_allclose(result[1], jnp.zeros(2), atol=1e-6)
        assert jnp.all(jnp.isfinite(result))

