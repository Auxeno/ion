import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

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

    @pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
    def test_mixed_precision(self, dtype):
        """Normalization uses float32 while preserving the input dtype."""
        data = jnp.zeros(4096, dtype=dtype)
        segment_ids = jnp.zeros(4096, dtype=jnp.int32)
        result = gnn.segment_softmax(data, segment_ids, num_segments=1)

        assert result.dtype == dtype
        npt.assert_array_equal(result, jnp.full(4096, 1 / 4096, dtype=dtype))


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


class TestRemoveSelfLoops:
    def test_drops_self_loops(self):
        """Edges with matching sender and receiver are dropped."""
        senders = jnp.array([0, 1, 1])
        receivers = jnp.array([1, 1, 2])
        s, r = gnn.remove_self_loops(senders, receivers)
        npt.assert_array_equal(s, jnp.array([0, 1]))
        npt.assert_array_equal(r, jnp.array([1, 2]))

    def test_preserves_order(self):
        """Surviving edges keep their original relative order."""
        senders = jnp.array([2, 0, 1, 1])
        receivers = jnp.array([2, 1, 1, 0])
        s, r = gnn.remove_self_loops(senders, receivers)
        npt.assert_array_equal(s, jnp.array([0, 1]))
        npt.assert_array_equal(r, jnp.array([1, 0]))

    def test_no_self_loops_is_identity(self):
        """A graph without self-loops passes through unchanged."""
        senders = jnp.array([0, 2, 1])
        receivers = jnp.array([1, 0, 2])
        s, r = gnn.remove_self_loops(senders, receivers)
        npt.assert_array_equal(s, senders)
        npt.assert_array_equal(r, receivers)

    def test_all_self_loops(self):
        """Removing every edge yields empty arrays."""
        senders = jnp.array([0, 1, 2])
        receivers = jnp.array([0, 1, 2])
        s, r = gnn.remove_self_loops(senders, receivers)
        assert s.shape[0] == 0
        assert r.shape[0] == 0

    def test_empty_graph(self):
        """Works on a graph with no edges."""
        senders = jnp.array([], dtype=jnp.int32)
        receivers = jnp.array([], dtype=jnp.int32)
        s, r = gnn.remove_self_loops(senders, receivers)
        assert s.shape[0] == 0
        assert r.shape[0] == 0

    def test_inverts_add_self_loops(self):
        """Removing after adding recovers the original edges."""
        senders = jnp.array([0, 2, 1])
        receivers = jnp.array([1, 0, 2])
        s, r = gnn.add_self_loops(senders, receivers, num_nodes=3)
        s, r = gnn.remove_self_loops(s, r)
        npt.assert_array_equal(s, senders)
        npt.assert_array_equal(r, receivers)


class TestDegree:
    def test_out_degree(self):
        """Counting senders gives the number of edges leaving each node."""
        senders = jnp.array([0, 0, 1])
        npt.assert_array_equal(gnn.degree(senders, num_nodes=3), jnp.array([2, 1, 0]))

    def test_in_degree(self):
        """Counting receivers gives the number of edges arriving at each node."""
        receivers = jnp.array([1, 2, 2])
        npt.assert_array_equal(gnn.degree(receivers, num_nodes=3), jnp.array([0, 1, 2]))

    def test_length_matches_num_nodes(self):
        """Isolated trailing nodes are still counted as zero."""
        senders = jnp.array([0, 1])
        assert gnn.degree(senders, num_nodes=6).shape == (6,)

    def test_total_equals_edge_count(self):
        """Degrees sum to the number of edges."""
        senders = jnp.array([0, 2, 1, 1])
        assert int(gnn.degree(senders, num_nodes=3).sum()) == 4

    def test_empty_graph(self):
        """A graph with no edges has zero degree everywhere."""
        senders = jnp.array([], dtype=jnp.int32)
        npt.assert_array_equal(gnn.degree(senders, num_nodes=3), jnp.zeros(3, dtype=int))

    def test_jittable(self):
        """Output shape is set by num_nodes, so the call traces."""
        fn = jax.jit(gnn.degree, static_argnums=1)
        npt.assert_array_equal(fn(jnp.array([0, 0, 2]), 3), jnp.array([2, 0, 1]))


class TestCoalesce:
    def test_sorts_and_deduplicates(self):
        """Edges come back sorted by (sender, receiver) with duplicates dropped."""
        senders = jnp.array([2, 0, 2, 1])
        receivers = jnp.array([0, 1, 0, 2])
        s, r, kept = gnn.coalesce(senders, receivers, num_nodes=3)
        npt.assert_array_equal(s, jnp.array([0, 1, 2]))
        npt.assert_array_equal(r, jnp.array([1, 2, 0]))
        npt.assert_array_equal(kept, jnp.array([1, 3, 0]))

    def test_kept_indexes_original_rows(self):
        """The kept indices select the surviving rows of the input."""
        senders = jnp.array([2, 0, 2, 1])
        receivers = jnp.array([0, 1, 0, 2])
        s, r, kept = gnn.coalesce(senders, receivers, num_nodes=3)
        npt.assert_array_equal(senders[kept], s)
        npt.assert_array_equal(receivers[kept], r)

    def test_keeps_first_duplicate(self):
        """The earliest occurrence of a repeated edge is the one kept."""
        senders = jnp.array([1, 1, 1])
        receivers = jnp.array([2, 2, 2])
        _, _, kept = gnn.coalesce(senders, receivers, num_nodes=3)
        npt.assert_array_equal(kept, jnp.array([0]))

    def test_idempotent(self):
        """Coalescing an already-canonical edge list changes nothing."""
        senders = jnp.array([2, 0, 2, 1])
        receivers = jnp.array([0, 1, 0, 2])
        s, r, _ = gnn.coalesce(senders, receivers, num_nodes=3)
        s2, r2, _ = gnn.coalesce(s, r, num_nodes=3)
        npt.assert_array_equal(s2, s)
        npt.assert_array_equal(r2, r)

    def test_preserves_self_loops(self):
        """Self-loops are ordinary edges and survive coalescing."""
        senders = jnp.array([1, 0, 1])
        receivers = jnp.array([1, 1, 1])
        s, r, _ = gnn.coalesce(senders, receivers, num_nodes=2)
        npt.assert_array_equal(s, jnp.array([0, 1]))
        npt.assert_array_equal(r, jnp.array([1, 1]))

    def test_distinguishes_direction(self):
        """(i, j) and (j, i) are different edges and both survive."""
        senders = jnp.array([0, 1])
        receivers = jnp.array([1, 0])
        s, r, _ = gnn.coalesce(senders, receivers, num_nodes=2)
        npt.assert_array_equal(s, jnp.array([0, 1]))
        npt.assert_array_equal(r, jnp.array([1, 0]))

    def test_empty_graph(self):
        """Works on a graph with no edges."""
        empty = jnp.array([], dtype=jnp.int32)
        s, r, kept = gnn.coalesce(empty, empty, num_nodes=3)
        assert s.shape[0] == 0
        assert r.shape[0] == 0
        assert kept.shape[0] == 0


class TestToUndirected:
    def test_adds_missing_reverses(self):
        """Edges without a reverse gain one; existing pairs are left alone."""
        senders = jnp.array([0, 1, 1])
        receivers = jnp.array([1, 0, 2])
        s, r, kept = gnn.to_undirected(senders, receivers, num_nodes=3)
        npt.assert_array_equal(s, jnp.array([0, 1, 1, 2]))
        npt.assert_array_equal(r, jnp.array([1, 0, 2, 1]))
        npt.assert_array_equal(kept, jnp.array([0, 1, 2, 5]))

    def test_output_is_symmetric(self):
        """Every edge in the result has its reverse in the result."""
        senders = jnp.array([0, 1, 3, 2])
        receivers = jnp.array([1, 2, 0, 3])
        s, r, _ = gnn.to_undirected(senders, receivers, num_nodes=4)
        forward = {(int(a), int(b)) for a, b in zip(s, r)}
        assert forward == {(b, a) for a, b in forward}

    def test_idempotent(self):
        """Coalescing means a symmetric graph passes through unchanged."""
        senders = jnp.array([0, 1, 1])
        receivers = jnp.array([1, 0, 2])
        s, r, _ = gnn.to_undirected(senders, receivers, num_nodes=3)
        s2, r2, _ = gnn.to_undirected(s, r, num_nodes=3)
        npt.assert_array_equal(s2, s)
        npt.assert_array_equal(r2, r)

    def test_discards_direction(self):
        """Opposite orientations of the same edge give the same result."""
        forward = gnn.to_undirected(jnp.array([0]), jnp.array([1]), num_nodes=2)
        backward = gnn.to_undirected(jnp.array([1]), jnp.array([0]), num_nodes=2)
        npt.assert_array_equal(forward[0], backward[0])
        npt.assert_array_equal(forward[1], backward[1])

    def test_self_loop_appears_once(self):
        """A self-loop is its own reverse and is not duplicated."""
        senders = jnp.array([1])
        receivers = jnp.array([1])
        s, r, _ = gnn.to_undirected(senders, receivers, num_nodes=2)
        npt.assert_array_equal(s, jnp.array([1]))
        npt.assert_array_equal(r, jnp.array([1]))

    def test_copies_edge_features(self):
        """Symmetric features are duplicated onto the reverse edge."""
        senders = jnp.array([0, 1, 1])
        receivers = jnp.array([1, 0, 2])
        x_edge = jnp.array([[10.0], [20.0], [30.0]])
        _, _, kept = gnn.to_undirected(senders, receivers, num_nodes=3)
        x_edge = jnp.concatenate([x_edge, x_edge])[kept]
        npt.assert_allclose(x_edge.ravel(), jnp.array([10.0, 20.0, 30.0, 30.0]))

    def test_negates_directional_features(self):
        """Direction-dependent features flip sign on the reverse edge."""
        senders = jnp.array([0, 1, 1])
        receivers = jnp.array([1, 0, 2])
        x_edge = jnp.array([[10.0], [20.0], [30.0]])
        _, _, kept = gnn.to_undirected(senders, receivers, num_nodes=3)
        x_edge = jnp.concatenate([x_edge, -x_edge])[kept]
        npt.assert_allclose(x_edge.ravel(), jnp.array([10.0, 20.0, 30.0, -30.0]))

    def test_empty_graph(self):
        """Works on a graph with no edges."""
        empty = jnp.array([], dtype=jnp.int32)
        s, r, kept = gnn.to_undirected(empty, empty, num_nodes=3)
        assert s.shape[0] == 0
        assert kept.shape[0] == 0


class TestReexports:
    def test_aliases_jax_ops(self):
        """Unwrapped segment ops are the jax.ops functions themselves."""
        assert ops.segment_sum is not jax.ops.segment_sum
        assert ops.segment_max is jax.ops.segment_max
        assert ops.segment_min is jax.ops.segment_min
        assert ops.segment_prod is jax.ops.segment_prod
        assert gnn.segment_sum is ops.segment_sum
        assert gnn.segment_max is ops.segment_max
        assert gnn.segment_min is ops.segment_min
        assert gnn.segment_prod is ops.segment_prod


class TestSegmentSum:
    @pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
    def test_mixed_precision(self, dtype):
        """Floating-point sums accumulate in float32 and return the input dtype."""
        data = jnp.ones(4096, dtype=dtype)
        segment_ids = jnp.zeros(4096, dtype=jnp.int32)
        result = gnn.segment_sum(data, segment_ids, num_segments=1)

        assert result.dtype == dtype
        npt.assert_array_equal(result, jnp.array([4096], dtype=dtype))

    def test_integer_data_is_not_cast_to_float32(self):
        """Integer segment sums retain exact integer behavior."""
        data = jnp.array([2**24, 1], dtype=jnp.int32)
        result = gnn.segment_sum(data, jnp.array([0, 0]), num_segments=1)
        npt.assert_array_equal(result, jnp.array([2**24 + 1], dtype=jnp.int32))


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

    @pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
    def test_mixed_precision(self, dtype):
        """Sums and counts use float32 while preserving the input dtype."""
        data = jnp.concatenate((jnp.zeros(2048, dtype=dtype), jnp.ones(2048, dtype=dtype)))
        segment_ids = jnp.zeros(4096, dtype=jnp.int32)
        result = gnn.segment_mean(data, segment_ids, num_segments=1)

        assert result.dtype == dtype
        npt.assert_array_equal(result, jnp.array([0.5], dtype=dtype))


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
