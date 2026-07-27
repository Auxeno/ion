import jax
import jax.numpy as jnp
import numpy.testing as npt

from ion import gnn
from ion.gnn import ops


class TestSegmentSoftmax:
    def test_sums_to_one_per_segment(self):
        """Each segment's weights sum to 1 after normalization."""
        data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        segment_ids = jnp.array([0, 0, 0, 1, 1])
        result = gnn.segment_softmax(data, segment_ids, num_segments=2)
        seg_0_sum = result[:3].sum()
        seg_1_sum = result[3:].sum()
        npt.assert_allclose(seg_0_sum, 1.0, atol=1e-5)
        npt.assert_allclose(seg_1_sum, 1.0, atol=1e-5)

    def test_single_segment_matches_softmax(self):
        """With one segment, result matches regular softmax."""
        data = jnp.array([1.0, 2.0, 3.0])
        segment_ids = jnp.array([0, 0, 0])
        result = gnn.segment_softmax(data, segment_ids, num_segments=1)
        expected = jax.nn.softmax(data)
        npt.assert_allclose(result, expected, atol=1e-5)

    def test_preserves_relative_order(self):
        """Larger values get larger weights within each segment."""
        data = jnp.array([1.0, 3.0, 2.0])
        segment_ids = jnp.array([0, 0, 0])
        result = gnn.segment_softmax(data, segment_ids, num_segments=1)
        assert result[1] > result[2] > result[0]

    def test_large_values_stable(self):
        """Large input values produce finite output (no overflow)."""
        data = jnp.array([1000.0, 1001.0, 999.0])
        segment_ids = jnp.array([0, 0, 0])
        result = gnn.segment_softmax(data, segment_ids, num_segments=1)
        assert jnp.all(jnp.isfinite(result))
        npt.assert_allclose(result.sum(), 1.0, atol=1e-5)

    def test_multidimensional_data(self):
        """Works with (e, h) shaped data for multi-head attention."""
        data = jax.random.normal(jax.random.key(0), (6, 4))
        segment_ids = jnp.array([0, 0, 0, 1, 1, 1])
        result = gnn.segment_softmax(data, segment_ids, num_segments=2)
        # Each head in each segment sums to 1
        for head in range(4):
            npt.assert_allclose(result[:3, head].sum(), 1.0, atol=1e-5)
            npt.assert_allclose(result[3:, head].sum(), 1.0, atol=1e-5)

    def test_sums_to_one_exact(self):
        """Per-segment sums hit 1.0 to float32 roundoff; the old +1e-6 epsilon biased them low."""
        data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        segment_ids = jnp.array([0, 0, 0, 1, 1])
        result = gnn.segment_softmax(data, segment_ids, num_segments=2)
        npt.assert_allclose(result[:3].sum(), 1.0, atol=2e-7)
        npt.assert_allclose(result[3:].sum(), 1.0, atol=2e-7)

    def test_empty_segment(self):
        """Segments with no members produce no NaNs and leave other segments exact."""
        data = jnp.array([1.0, 2.0])
        segment_ids = jnp.array([0, 0])
        result = gnn.segment_softmax(data, segment_ids, num_segments=3)
        assert jnp.all(jnp.isfinite(result))
        npt.assert_allclose(result.sum(), 1.0, atol=2e-7)

    def test_jit_compatible(self):
        """segment_softmax works under jax.jit."""
        data = jnp.array([1.0, 2.0, 3.0])
        segment_ids = jnp.array([0, 0, 0])
        eager = gnn.segment_softmax(data, segment_ids, 1)
        jitted = jax.jit(gnn.segment_softmax, static_argnums=2)(data, segment_ids, 1)
        npt.assert_allclose(eager, jitted, atol=1e-6)


class TestAddSelfLoops:
    def test_output_length(self):
        """Output has num_nodes extra edges appended."""
        senders = jnp.array([0, 1])
        receivers = jnp.array([1, 0])
        s, r = gnn.add_self_loops(senders, receivers, num_nodes=3)
        assert s.shape[0] == 2 + 3
        assert r.shape[0] == 2 + 3

    def test_self_loop_content(self):
        """Appended edges are (0->0), (1->1), ..., (n-1->n-1)."""
        senders = jnp.array([0, 1])
        receivers = jnp.array([1, 0])
        s, r = gnn.add_self_loops(senders, receivers, num_nodes=3)
        # Last 3 edges are self-loops
        npt.assert_array_equal(s[2:], jnp.array([0, 1, 2]))
        npt.assert_array_equal(r[2:], jnp.array([0, 1, 2]))

    def test_preserves_original_edges(self):
        """Original edges are unchanged at the start of the array."""
        senders = jnp.array([0, 2, 1])
        receivers = jnp.array([1, 0, 2])
        s, r = gnn.add_self_loops(senders, receivers, num_nodes=3)
        npt.assert_array_equal(s[:3], senders)
        npt.assert_array_equal(r[:3], receivers)

    def test_empty_graph(self):
        """Works on a graph with no edges (self-loops only)."""
        senders = jnp.array([], dtype=jnp.int32)
        receivers = jnp.array([], dtype=jnp.int32)
        s, r = gnn.add_self_loops(senders, receivers, num_nodes=4)
        assert s.shape[0] == 4
        npt.assert_array_equal(s, jnp.arange(4))
        npt.assert_array_equal(r, jnp.arange(4))


class TestReexports:
    def test_aliases_jax_ops(self):
        """Re-exported segment ops are the jax.ops functions themselves."""
        assert ops.segment_sum is jax.ops.segment_sum
        assert ops.segment_max is jax.ops.segment_max
        assert ops.segment_min is jax.ops.segment_min
        assert ops.segment_prod is jax.ops.segment_prod
        assert gnn.segment_sum is ops.segment_sum
        assert gnn.segment_max is ops.segment_max
        assert gnn.segment_min is ops.segment_min
        assert gnn.segment_prod is ops.segment_prod


class TestSegmentMean:
    def test_output_manual(self):
        """Output matches per-segment mean computed manually."""
        data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        segment_ids = jnp.array([0, 0, 0, 1, 1])
        result = gnn.segment_mean(data, segment_ids, num_segments=2)
        npt.assert_allclose(result, jnp.array([2.0, 4.5]), rtol=1e-5, atol=1e-5)

    def test_multidimensional_data(self):
        """Works with (e, d) shaped data."""
        data = jax.random.normal(jax.random.key(0), (6, 4))
        segment_ids = jnp.array([0, 0, 0, 1, 1, 1])
        result = gnn.segment_mean(data, segment_ids, num_segments=2)
        npt.assert_allclose(result[0], data[:3].mean(axis=0), rtol=1e-5, atol=1e-5)
        npt.assert_allclose(result[1], data[3:].mean(axis=0), rtol=1e-5, atol=1e-5)

    def test_empty_segment(self):
        """Segments with no members give zeros, not NaN."""
        data = jnp.array([1.0, 2.0])
        segment_ids = jnp.array([0, 0])
        result = gnn.segment_mean(data, segment_ids, num_segments=3)
        npt.assert_allclose(result, jnp.array([1.5, 0.0, 0.0]), rtol=1e-5, atol=1e-5)

    def test_jit_compatible(self):
        """segment_mean works under jax.jit."""
        data = jnp.array([1.0, 2.0, 3.0])
        segment_ids = jnp.array([0, 0, 1])
        eager = gnn.segment_mean(data, segment_ids, 2)
        jitted = jax.jit(gnn.segment_mean, static_argnums=2)(data, segment_ids, 2)
        npt.assert_allclose(eager, jitted, atol=1e-6)


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
