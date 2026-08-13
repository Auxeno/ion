import jax
import jax.numpy as jnp
import numpy.testing as npt

from ion import gnn, nn


class TestRGCNConv:
    def test_output_manual(self):
        """Output matches per-relation mean aggregation plus the root transform."""
        conv = gnn.RGCNConv(4, 6, 2, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 4))
        senders = jnp.array([0, 1, 2])
        receivers = jnp.array([1, 2, 0])
        edge_type = jnp.array([0, 1, 0])

        expected = x @ conv.w_self + conv.b  # type: ignore[operator]
        for relation in range(2):
            mask = edge_type == relation
            neigh = (x @ conv.w_neigh[relation])[senders] * mask[:, None]
            count = gnn.segment_sum(mask.astype(x.dtype), receivers, 3)
            expected = (
                expected
                + gnn.segment_sum(neigh, receivers, 3) / jnp.where(count > 0, count, 1)[:, None]
            )

        y = conv(x, senders, receivers, edge_type=edge_type)

        npt.assert_allclose(y, expected, rtol=1e-5, atol=1e-5)

    def test_single_relation_matches_sage_mean(self):
        """One relation reduces to mean aggregation, matching SAGEConv."""
        sage = gnn.SAGEConv(4, 6, key=jax.random.key(0))
        conv = gnn.RGCNConv(4, 6, 1, key=jax.random.key(1))
        conv = conv.at.w_neigh.set(nn.Param(sage.w_neigh.value[None]))
        conv = conv.at.w_self.set(sage.w_self)
        conv = conv.at.b.set(sage.b)
        x = jax.random.normal(jax.random.key(2), (3, 4))
        senders = jnp.array([0, 1, 2])
        receivers = jnp.array([1, 2, 0])

        y = conv(x, senders, receivers, edge_type=jnp.zeros(3, dtype=jnp.int32))

        npt.assert_allclose(y, sage(x, senders, receivers), rtol=1e-5, atol=1e-5)

    def test_output_shape(self):
        """Output shape is (num_nodes, out_dim)."""
        conv = gnn.RGCNConv(8, 16, 3, key=jax.random.key(0))
        x = jnp.ones((5, 8))
        senders = jnp.array([0, 1, 2, 3])
        receivers = jnp.array([1, 2, 3, 4])
        edge_type = jnp.array([0, 1, 2, 0])

        y = conv(x, senders, receivers, edge_type=edge_type)

        assert y.shape == (5, 16)

    def test_relations_use_separate_transforms(self):
        """Relabelling an edge's relation changes the output it contributes to."""
        conv = gnn.RGCNConv(4, 6, 2, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 4))
        senders = jnp.array([0])
        receivers = jnp.array([1])

        first = conv(x, senders, receivers, edge_type=jnp.array([0]))
        second = conv(x, senders, receivers, edge_type=jnp.array([1]))

        assert not jnp.allclose(first[1], second[1])

    def test_basis_decomposition_shapes(self):
        """num_bases shares matrices across relations through mixing coefficients."""
        conv = gnn.RGCNConv(4, 6, 12, num_bases=2, key=jax.random.key(0))

        assert conv.w_neigh.shape == (2, 4, 6)
        assert conv.w_coeff is not None
        assert conv.w_coeff.shape == (12, 2)

    def test_no_basis_decomposition(self):
        """Without num_bases each relation holds its own transform."""
        conv = gnn.RGCNConv(4, 6, 3, key=jax.random.key(0))

        assert conv.w_neigh.shape == (3, 4, 6)
        assert conv.w_coeff is None

    def test_no_bias(self):
        """use_bias=False removes the bias."""
        conv = gnn.RGCNConv(4, 6, 2, use_bias=False, key=jax.random.key(0))

        assert conv.b is None

    def test_isolated_node_gets_root_and_bias(self):
        """A node with no incoming edges receives only its root term and bias."""
        conv = gnn.RGCNConv(4, 6, 2, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 4))
        senders = jnp.array([0])
        receivers = jnp.array([1])

        y = conv(x, senders, receivers, edge_type=jnp.array([0]))
        expected = x[2] @ conv.w_self + conv.b  # type: ignore[operator]

        npt.assert_allclose(y[2], expected, rtol=1e-5, atol=1e-5)

    def test_jit_and_grad(self):
        """The layer traces under jit and produces finite parameter gradients."""
        conv = gnn.RGCNConv(4, 6, 2, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (3, 4))
        senders = jnp.array([0, 1, 2])
        receivers = jnp.array([1, 2, 0])
        edge_type = jnp.array([0, 1, 0])

        y = conv(x, senders, receivers, edge_type=edge_type)
        jitted = jax.jit(conv)(x, senders, receivers, edge_type=edge_type)
        grads = jax.grad(lambda m: m(x, senders, receivers, edge_type=edge_type).sum())(conv)

        npt.assert_allclose(jitted, y, rtol=1e-5, atol=1e-5)
        assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(grads))

    def test_default_dtype(self):
        """Weights default to float32."""
        conv = gnn.RGCNConv(4, 6, 2, key=jax.random.key(0))

        assert conv.w_neigh.dtype == jnp.float32
        assert conv.w_self.dtype == jnp.float32
