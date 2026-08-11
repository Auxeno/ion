import jax
import jax.numpy as jnp
from jax.nn.initializers import zeros
import numpy.testing as npt
import pytest

from ion import gnn, nn


class TestGlobalAttentionPool:
    def test_output_manual(self):
        """Output matches a per-graph softmax and weighted sum."""
        score = nn.Linear(2, 1, key=jax.random.key(0))
        pool = gnn.GlobalAttentionPool(score)
        x = jnp.array([[1.0, 0.0], [3.0, 2.0], [0.0, 4.0], [2.0, 1.0]])
        graph_ids = jnp.array([0, 0, 1, 1])

        y = pool(x, graph_ids, num_graphs=2)

        logits = score(x)
        weights_0 = jax.nn.softmax(logits[:2], axis=0)
        weights_1 = jax.nn.softmax(logits[2:], axis=0)
        expected = jnp.stack(
            [(weights_0 * x[:2]).sum(axis=0), (weights_1 * x[2:]).sum(axis=0)]
        )
        npt.assert_allclose(y, expected, rtol=1e-6, atol=1e-6)

    def test_uniform_attention_matches_mean_pool(self):
        """A zero score assigns uniform weight within each graph."""
        score = nn.Linear(3, 1, w_init=zeros, b_init=zeros, key=jax.random.key(0))
        pool = gnn.GlobalAttentionPool(score)
        x = jax.random.normal(jax.random.key(1), (5, 3))
        graph_ids = jnp.array([0, 0, 1, 1, 1])

        expected = gnn.mean_pool(x, graph_ids, num_graphs=2)
        npt.assert_allclose(pool(x, graph_ids, 2), expected, rtol=1e-6, atol=1e-6)

    def test_value(self):
        """The optional value module supplies the features being pooled."""
        key_score, key_value = jax.random.split(jax.random.key(0))
        score = nn.Linear(2, 1, key=key_score)
        value = nn.Linear(2, 4, key=key_value)
        pool = gnn.GlobalAttentionPool(score, value=value)
        x = jax.random.normal(jax.random.key(1), (5, 2))
        graph_ids = jnp.array([0, 0, 1, 1, 1])

        attention = gnn.segment_softmax(score(x), graph_ids, 2)
        expected = gnn.segment_sum(attention * value(x), graph_ids, 2)
        result = pool(x, graph_ids, 2)

        assert result.shape == (2, 4)
        npt.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)

    def test_empty_graph(self):
        """Graphs with no nodes produce zero rows."""
        pool = gnn.GlobalAttentionPool(nn.Linear(2, 1, key=jax.random.key(0)))
        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        graph_ids = jnp.array([0, 2])
        result = pool(x, graph_ids, num_graphs=4)

        npt.assert_array_equal(result[1], jnp.zeros(2))
        npt.assert_array_equal(result[3], jnp.zeros(2))

    def test_permutation_invariant(self):
        """Reordering packed nodes does not change graph representations."""
        pool = gnn.GlobalAttentionPool(nn.Linear(3, 1, key=jax.random.key(0)))
        x = jax.random.normal(jax.random.key(1), (6, 3))
        graph_ids = jnp.array([0, 1, 0, 1, 1, 0])
        order = jnp.array([4, 2, 0, 5, 1, 3])

        expected = pool(x, graph_ids, 2)
        result = pool(x[order], graph_ids[order], 2)
        npt.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)

    def test_score_requires_scalar_output(self):
        """The score must produce one attention logit per node."""
        pool = gnn.GlobalAttentionPool(nn.Linear(3, 2, key=jax.random.key(0)))
        x = jnp.ones((4, 3))

        with pytest.raises(ValueError, match=r"score must return shape \(4, 1\), got \(4, 2\)"):
            pool(x, jnp.array([0, 0, 1, 1]), 2)

    def test_children_supply_all_params(self):
        """The pool creates no parameters beyond its child modules."""
        key_score, key_value = jax.random.split(jax.random.key(0))
        score = nn.Linear(3, 1, key=key_score)
        value = nn.Linear(3, 5, key=key_value)
        pool = gnn.GlobalAttentionPool(score, value=value)

        assert pool.num_params == score.num_params + value.num_params

    def test_jit_and_grad(self):
        """The readout composes with JIT and differentiation."""
        pool = gnn.GlobalAttentionPool(nn.Linear(3, 1, key=jax.random.key(0)))
        x = jax.random.normal(jax.random.key(1), (5, 3))
        graph_ids = jnp.array([0, 0, 1, 1, 1])

        expected = pool(x, graph_ids, 2)
        result = jax.jit(pool, static_argnums=2)(x, graph_ids, 2)
        grads = jax.grad(lambda model: model(x, graph_ids, 2).sum())(pool)

        npt.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)
        assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(grads))

    def test_bfloat16(self):
        """Reduced-precision inputs preserve their dtype and stay finite."""
        pool = gnn.GlobalAttentionPool(nn.Linear(3, 1, key=jax.random.key(0))).astype(jnp.bfloat16)
        x = jax.random.normal(jax.random.key(1), (5, 3), dtype=jnp.bfloat16)
        graph_ids = jnp.array([0, 0, 1, 1, 1])
        result = pool(x, graph_ids, 2)

        assert result.dtype == jnp.bfloat16
        assert jnp.all(jnp.isfinite(result))
