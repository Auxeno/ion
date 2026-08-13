import jax
import jax.numpy as jnp
import numpy.testing as npt

from ion import gnn, nn


class TestEdgeUpdate:
    def test_output_manual(self):
        """Output is edge_model(concat(sender, receiver, edge))."""
        edge_model = nn.MLP([10, 8, 6], key=jax.random.key(0))
        update = gnn.EdgeUpdate(edge_model)
        x = jax.random.normal(jax.random.key(1), (3, 4))
        x_edge = jax.random.normal(jax.random.key(2), (3, 2))
        senders = jnp.array([0, 1, 2])
        receivers = jnp.array([1, 2, 0])

        expected = edge_model(jnp.concatenate((x[senders], x[receivers], x_edge), axis=-1))
        result = update(x, senders, receivers, x_edge=x_edge)

        npt.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_bipartite_output_manual(self):
        """Bipartite edges gather endpoints from separate node sets."""
        edge_model = nn.MLP([10, 7], key=jax.random.key(0))
        update = gnn.EdgeUpdate(edge_model)
        x_src = jax.random.normal(jax.random.key(1), (4, 3))
        x_dst = jax.random.normal(jax.random.key(2), (2, 5))
        x_edge = jax.random.normal(jax.random.key(3), (3, 2))
        senders = jnp.array([0, 3, 1])
        receivers = jnp.array([0, 0, 1])

        edge_inputs = jnp.concatenate((x_src[senders], x_dst[receivers], x_edge), axis=-1)
        expected = edge_model(edge_inputs)
        y = update((x_src, x_dst), senders, receivers, x_edge=x_edge)

        npt.assert_allclose(y, expected, rtol=1e-5, atol=1e-5)

    def test_output_shape(self):
        """Output has one row per edge and the edge model's output dimension."""
        update = gnn.EdgeUpdate(nn.MLP([10, 6], key=jax.random.key(0)))
        x = jnp.ones((5, 4))
        x_edge = jnp.ones((3, 2))
        senders = jnp.array([0, 1, 2])
        receivers = jnp.array([1, 2, 3])

        assert update(x, senders, receivers, x_edge=x_edge).shape == (3, 6)

    def test_creates_no_params(self):
        """All parameters belong to the caller-supplied edge model."""
        edge_model = nn.MLP([10, 8, 6], key=jax.random.key(0))
        update = gnn.EdgeUpdate(edge_model)

        assert update.num_params == edge_model.num_params
