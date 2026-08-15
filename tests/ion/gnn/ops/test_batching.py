import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from ion import gnn


class TestBatchGraphs:
    def test_edge_offsets(self):
        """Edge indices of later graphs are offset by cumulative node counts."""
        xs = [jnp.ones((3, 2)), jnp.ones((2, 2))]
        senders_list = [jnp.array([0, 1]), jnp.array([0, 1])]
        receivers_list = [jnp.array([1, 2]), jnp.array([1, 0])]
        x, senders, receivers, graph_ids = gnn.batch_graphs(xs, senders_list, receivers_list)
        assert x.shape == (5, 2)
        npt.assert_array_equal(senders, jnp.array([0, 1, 3, 4]))
        npt.assert_array_equal(receivers, jnp.array([1, 2, 4, 3]))

    def test_graph_ids(self):
        """graph_ids maps each node to its source graph."""
        xs = [jnp.ones((3, 2)), jnp.ones((2, 2))]
        senders_list = [jnp.array([0]), jnp.array([0])]
        receivers_list = [jnp.array([1]), jnp.array([1])]
        _, _, _, graph_ids = gnn.batch_graphs(xs, senders_list, receivers_list)
        npt.assert_array_equal(graph_ids, jnp.array([0, 0, 0, 1, 1]))

    def test_single_graph_identity(self):
        """Batching one graph leaves features and edges unchanged."""
        x_in = jax.random.normal(jax.random.key(0), (4, 3))
        senders_in = jnp.array([0, 1, 2])
        receivers_in = jnp.array([1, 2, 3])
        x, senders, receivers, graph_ids = gnn.batch_graphs([x_in], [senders_in], [receivers_in])
        npt.assert_array_equal(x, x_in)
        npt.assert_array_equal(senders, senders_in)
        npt.assert_array_equal(receivers, receivers_in)
        npt.assert_array_equal(graph_ids, jnp.zeros(4, dtype=jnp.int32))

    def test_batched_conv_matches_per_graph(self):
        """Conv + mean_pool over a batch matches running each graph separately."""
        conv = gnn.GCNConv(4, 8, key=jax.random.key(0))
        keys = jax.random.split(jax.random.key(1), 2)
        xs = [jax.random.normal(keys[0], (3, 4)), jax.random.normal(keys[1], (5, 4))]
        senders_list = [jnp.array([0, 1, 2]), jnp.array([0, 1, 2, 3, 4])]
        receivers_list = [jnp.array([1, 2, 0]), jnp.array([1, 2, 3, 4, 0])]
        x, senders, receivers, graph_ids = gnn.batch_graphs(xs, senders_list, receivers_list)
        batched = gnn.mean_pool(conv(x, senders, receivers), graph_ids, num_graphs=2)
        separate = jnp.stack(
            [
                conv(x_g, s_g, r_g).mean(axis=0)
                for x_g, s_g, r_g in zip(xs, senders_list, receivers_list)
            ]
        )
        npt.assert_allclose(batched, separate, rtol=1e-5, atol=1e-5)

    def test_numpy_inputs(self):
        """NumPy graphs batch to the same arrays as JAX ones."""
        xs = [np.ones((3, 2), np.float32), np.ones((2, 2), np.float32)]
        senders_list = [np.array([0, 1]), np.array([0, 1])]
        receivers_list = [np.array([1, 2]), np.array([1, 0])]
        from_numpy = gnn.batch_graphs(xs, senders_list, receivers_list)
        from_jax = gnn.batch_graphs(
            [jnp.asarray(x) for x in xs],
            [jnp.asarray(s) for s in senders_list],
            [jnp.asarray(r) for r in receivers_list],
        )
        for numpy_array, jax_array in zip(from_numpy, from_jax):
            assert isinstance(numpy_array, jax.Array)
            assert numpy_array.dtype == jax_array.dtype
            npt.assert_array_equal(numpy_array, jax_array)


class TestPadGraphs:
    def test_capacity_and_sentinels(self):
        """Padding fills to capacity with indices one past the last node and graph."""
        x = jnp.ones((5, 2))
        senders, receivers = jnp.array([0, 1]), jnp.array([1, 2])
        graph_ids = jnp.array([0, 0, 0, 1, 1])
        x, senders, receivers, graph_ids = gnn.pad_graphs(
            x,
            senders,
            receivers,
            graph_ids,
            node_capacity=8,
            edge_capacity=4,
            num_graphs=2,
        )
        assert (x.shape, senders.shape, graph_ids.shape) == ((8, 2), (4,), (8,))
        npt.assert_array_equal(x[5:], jnp.zeros((3, 2)))
        npt.assert_array_equal(senders[2:], jnp.array([8, 8]))
        npt.assert_array_equal(receivers[2:], jnp.array([8, 8]))
        npt.assert_array_equal(graph_ids[5:], jnp.array([2, 2, 2]))

    def test_exact_capacity_is_identity(self):
        """Padding to the current size changes nothing."""
        x = jax.random.normal(jax.random.key(0), (4, 3))
        senders, receivers = jnp.array([0, 1, 2]), jnp.array([1, 2, 3])
        graph_ids = jnp.zeros(4, dtype=jnp.int32)
        padded = gnn.pad_graphs(x, senders, receivers, graph_ids, 4, 3, 1)
        for original, result in zip((x, senders, receivers, graph_ids), padded):
            npt.assert_array_equal(result, original)

    def test_overflowing_capacity_raises(self):
        """A batch larger than its capacity fails rather than truncating."""
        with pytest.raises(ValueError):
            gnn.pad_graphs(
                jnp.ones((5, 2)), jnp.array([0]), jnp.array([1]), jnp.zeros(5, jnp.int32), 4, 1, 1
            )

    def test_conv_ignores_padding(self):
        """Convolutions give the same real node features with and without padding."""
        conv = gnn.GATv2Conv(4, 8, num_heads=2, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (5, 4))
        senders, receivers = jnp.array([0, 1, 2, 3]), jnp.array([1, 2, 3, 4])
        graph_ids = jnp.array([0, 0, 0, 1, 1])
        padded = gnn.pad_graphs(x, senders, receivers, graph_ids, 9, 7, 2)
        npt.assert_allclose(
            conv(x, senders, receivers),
            conv(padded[0], padded[1], padded[2])[:5],
            rtol=1e-5,
            atol=1e-5,
        )

    def test_pool_drops_padding(self):
        """Pooling a padded batch returns one row per real graph."""
        x = jax.random.normal(jax.random.key(0), (5, 3))
        senders, receivers = jnp.array([0, 1]), jnp.array([1, 2])
        graph_ids = jnp.array([0, 0, 0, 1, 1])
        padded = gnn.pad_graphs(x, senders, receivers, graph_ids, 9, 5, 2)
        for pool in (gnn.mean_pool, gnn.sum_pool, gnn.max_pool, gnn.min_pool):
            pooled = pool(padded[0], padded[3], num_graphs=2)
            assert pooled.shape == (2, 3)
            npt.assert_allclose(pooled, pool(x, graph_ids, num_graphs=2), rtol=1e-5, atol=1e-5)

    def test_gradients_ignore_padding(self):
        """Padding contributes no gradient to the model."""
        conv = gnn.GCNConv(4, 6, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (5, 4))
        senders, receivers = jnp.array([0, 1, 2, 3]), jnp.array([1, 2, 3, 4])
        graph_ids = jnp.array([0, 0, 0, 1, 1])
        padded = gnn.pad_graphs(x, senders, receivers, graph_ids, 9, 7, 2)

        def loss(conv, x, senders, receivers, graph_ids):
            return gnn.mean_pool(conv(x, senders, receivers), graph_ids, num_graphs=2).sum()

        unpadded_grads = jax.grad(loss)(conv, x, senders, receivers, graph_ids)
        padded_grads = jax.grad(loss)(conv, *padded)
        for unpadded_leaf, padded_leaf in zip(
            jax.tree.leaves(unpadded_grads), jax.tree.leaves(padded_grads)
        ):
            npt.assert_allclose(unpadded_leaf, padded_leaf, rtol=1e-5, atol=1e-5)


class TestUnbatchGraphs:
    def test_round_trip(self):
        """Unbatching a batched graph recovers the original graphs."""
        keys = jax.random.split(jax.random.key(0), 3)
        xs = [jax.random.normal(key, (n, 3)) for key, n in zip(keys, (4, 1, 2))]
        senders_list = [jnp.array([0, 2]), jnp.array([], dtype=jnp.int32), jnp.array([1])]
        receivers_list = [jnp.array([1, 3]), jnp.array([], dtype=jnp.int32), jnp.array([0])]
        batched = gnn.batch_graphs(xs, senders_list, receivers_list)
        out_xs, out_senders, out_receivers = gnn.unbatch_graphs(*batched)

        for original, restored in zip(xs, out_xs):
            npt.assert_array_equal(restored, original)
        for original, restored in zip(senders_list, out_senders):
            npt.assert_array_equal(restored, original)
        for original, restored in zip(receivers_list, out_receivers):
            npt.assert_array_equal(restored, original)

    def test_edge_offsets_removed(self):
        """Edge indices are shifted back to be local to each graph."""
        x = jnp.ones((5, 2))
        senders = jnp.array([0, 1, 3, 4])
        receivers = jnp.array([1, 2, 4, 3])
        graph_ids = jnp.array([0, 0, 0, 1, 1])
        _, out_senders, out_receivers = gnn.unbatch_graphs(x, senders, receivers, graph_ids)
        npt.assert_array_equal(out_senders[1], jnp.array([0, 1]))
        npt.assert_array_equal(out_receivers[1], jnp.array([1, 0]))

    def test_single_graph_identity(self):
        """Unbatching one graph leaves features and edges unchanged."""
        x = jax.random.normal(jax.random.key(0), (4, 3))
        senders = jnp.array([0, 1, 2])
        receivers = jnp.array([1, 2, 3])
        out_xs, out_senders, out_receivers = gnn.unbatch_graphs(
            x, senders, receivers, jnp.zeros(4, dtype=jnp.int32)
        )
        assert len(out_xs) == 1
        npt.assert_array_equal(out_xs[0], x)
        npt.assert_array_equal(out_senders[0], senders)
        npt.assert_array_equal(out_receivers[0], receivers)

