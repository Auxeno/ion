import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from ion import gnn, nn


class TestGraphNorm:
    def test_single_graph(self):
        """Omitting graph metadata normalizes all nodes as one graph."""
        norm = gnn.GraphNorm(3)
        x = jax.random.normal(jax.random.key(0), (6, 3))

        expected = norm(x, jnp.zeros(6, dtype=jnp.int32), 1)
        result = norm(x)

        npt.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)
        npt.assert_allclose(jax.jit(norm)(x), result, rtol=1e-6, atol=1e-6)

    def test_output_manual(self):
        """Output matches normalization by graph with a learned mean scale."""
        norm = gnn.GraphNorm(2, eps=1e-5)
        norm = norm.at.mean_scale.set(nn.Param(jnp.array([0.5, 1.5])))
        norm = norm.at.scale.set(nn.Param(jnp.array([2.0, 3.0])))
        norm = norm.at.b.set(nn.Param(jnp.array([1.0, -1.0])))
        assert norm.b is not None
        x = jnp.array([[1.0, 2.0], [3.0, 6.0], [4.0, 8.0], [10.0, 14.0], [16.0, 20.0]])
        graph_ids = jnp.array([0, 0, 1, 1, 1])

        means = jnp.stack([jnp.mean(x[:2], axis=0), jnp.mean(x[2:], axis=0)])
        shifted = x - norm.mean_scale * means[graph_ids]
        variances = jnp.stack(
            [jnp.mean(jnp.square(shifted[:2]), axis=0), jnp.mean(jnp.square(shifted[2:]), axis=0)]
        )
        expected = shifted / jnp.sqrt(variances[graph_ids] + norm.eps)
        expected = expected * norm.scale + norm.b

        npt.assert_allclose(norm(x, graph_ids, 2), expected, rtol=1e-6, atol=1e-6)

    def test_normalizes_each_graph(self):
        """Default mean scale gives zero mean and unit variance per graph."""
        norm = gnn.GraphNorm(3, eps=1e-8)
        x = jax.random.normal(jax.random.key(0), (12, 3))
        graph_ids = jnp.repeat(jnp.arange(3), 4)
        y = norm(x, graph_ids, 3)

        for graph in range(3):
            values = y[graph_ids == graph]
            npt.assert_allclose(jnp.mean(values, axis=0), 0.0, atol=1e-6)
            npt.assert_allclose(jnp.var(values, axis=0), 1.0, atol=1e-5)

    def test_parameter_initialization(self):
        """Affine and mean scales use their standard initial values."""
        norm = gnn.GraphNorm(4)
        assert norm.b is not None

        npt.assert_array_equal(norm.scale.value, jnp.ones(4))
        npt.assert_array_equal(norm.b.value, jnp.zeros(4))
        npt.assert_array_equal(norm.mean_scale.value, jnp.ones(4))
        assert norm.num_params == 12

    def test_no_bias(self):
        """use_bias=False removes the output bias without changing normalization."""
        norm = gnn.GraphNorm(3, use_bias=False)
        x = jax.random.normal(jax.random.key(0), (6, 3))

        assert norm.b is None
        npt.assert_allclose(norm(x), gnn.GraphNorm(3)(x), rtol=1e-6, atol=1e-6)
        assert norm.num_params == 6

    def test_graphs_are_independent(self):
        """Changing one graph does not change another graph's output."""
        norm = gnn.GraphNorm(2)
        x = jax.random.normal(jax.random.key(0), (6, 2))
        graph_ids = jnp.array([0, 0, 0, 1, 1, 1])
        changed = x.at[:3].multiply(10.0).at[:3].add(4.0)

        expected = norm(x, graph_ids, 2)
        result = norm(changed, graph_ids, 2)
        npt.assert_allclose(result[3:], expected[3:], rtol=1e-6, atol=1e-6)

    def test_permutation_equivariant(self):
        """Reordering packed nodes reorders the normalized output."""
        norm = gnn.GraphNorm(3)
        x = jax.random.normal(jax.random.key(0), (7, 3))
        graph_ids = jnp.array([0, 1, 0, 1, 1, 0, 1])
        order = jnp.array([4, 2, 0, 6, 5, 1, 3])

        expected = norm(x, graph_ids, 2)
        result = norm(x[order], graph_ids[order], 2)
        npt.assert_allclose(result, expected[order], rtol=1e-6, atol=1e-6)

    def test_single_node_graph(self):
        """A single-node graph normalizes to the initialized bias."""
        norm = gnn.GraphNorm(2)
        x = jnp.array([[2.0, -3.0], [1.0, 4.0], [5.0, 8.0]])
        graph_ids = jnp.array([0, 1, 1])

        npt.assert_allclose(norm(x, graph_ids, 2)[0], jnp.zeros(2), atol=1e-6)

    def test_graph_metadata_is_supplied_together(self):
        """Packed graph IDs and their graph count must be supplied together."""
        norm = gnn.GraphNorm(2)
        x = jnp.ones((3, 2))

        with pytest.raises(ValueError, match="num_graphs is required"):
            norm(x, jnp.zeros(3, dtype=jnp.int32))
        with pytest.raises(ValueError, match="num_graphs requires graph_ids"):
            norm(x, num_graphs=1)

    def test_jit_and_grad(self):
        """Graph normalization composes with JIT and differentiation."""
        norm = gnn.GraphNorm(3)
        x = jax.random.normal(jax.random.key(0), (6, 3))
        graph_ids = jnp.array([0, 0, 0, 1, 1, 1])

        expected = norm(x, graph_ids, 2)
        result = jax.jit(norm, static_argnums=2)(x, graph_ids, 2)
        model_grad = jax.grad(lambda model: jnp.square(model(x, graph_ids, 2)).sum())(norm)
        input_grad = jax.grad(lambda value: jnp.square(norm(value, graph_ids, 2)).sum())(x)

        npt.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)
        assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(model_grad))
        assert jnp.all(jnp.isfinite(input_grad))

    @pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
    def test_mixed_precision(self, dtype):
        """Statistics use float32 while preserving the input dtype."""
        norm = gnn.GraphNorm(3).astype(dtype)
        x = (100 * jax.random.normal(jax.random.key(0), (64, 3))).astype(dtype)
        graph_ids = jnp.repeat(jnp.arange(4), 16)

        result = norm(x, graph_ids, 4)
        expected = gnn.GraphNorm(3)(x.astype(jnp.float32), graph_ids, 4).astype(dtype)

        assert result.dtype == dtype
        assert jnp.all(jnp.isfinite(result))
        npt.assert_allclose(result, expected, rtol=1e-2, atol=1e-2)
