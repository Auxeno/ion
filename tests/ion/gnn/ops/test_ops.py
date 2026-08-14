import jax
import jax.numpy as jnp
import numpy as np
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

    def test_fully_masked_segment(self):
        """A segment of all -inf logits gives zero weights, not NaN."""
        data = jnp.array([-jnp.inf, -jnp.inf, 1.0])
        segment_ids = jnp.array([0, 0, 1])
        result = gnn.segment_softmax(data, segment_ids, num_segments=2)
        npt.assert_allclose(result, jnp.array([0.0, 0.0, 1.0]), atol=2e-7)

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
        s, r, kept = gnn.coalesce(senders, receivers)
        npt.assert_array_equal(s, jnp.array([0, 1, 2]))
        npt.assert_array_equal(r, jnp.array([1, 2, 0]))
        npt.assert_array_equal(kept, jnp.array([1, 3, 0]))

    def test_kept_indexes_original_rows(self):
        """The kept indices select the surviving rows of the input."""
        senders = jnp.array([2, 0, 2, 1])
        receivers = jnp.array([0, 1, 0, 2])
        s, r, kept = gnn.coalesce(senders, receivers)
        npt.assert_array_equal(senders[kept], s)
        npt.assert_array_equal(receivers[kept], r)

    def test_keeps_first_duplicate(self):
        """The earliest occurrence of a repeated edge is the one kept."""
        senders = jnp.array([1, 1, 1])
        receivers = jnp.array([2, 2, 2])
        _, _, kept = gnn.coalesce(senders, receivers)
        npt.assert_array_equal(kept, jnp.array([0]))

    def test_idempotent(self):
        """Coalescing an already-canonical edge list changes nothing."""
        senders = jnp.array([2, 0, 2, 1])
        receivers = jnp.array([0, 1, 0, 2])
        s, r, _ = gnn.coalesce(senders, receivers)
        s2, r2, _ = gnn.coalesce(s, r)
        npt.assert_array_equal(s2, s)
        npt.assert_array_equal(r2, r)

    def test_preserves_self_loops(self):
        """Self-loops are ordinary edges and survive coalescing."""
        senders = jnp.array([1, 0, 1])
        receivers = jnp.array([1, 1, 1])
        s, r, _ = gnn.coalesce(senders, receivers)
        npt.assert_array_equal(s, jnp.array([0, 1]))
        npt.assert_array_equal(r, jnp.array([1, 1]))

    def test_distinguishes_direction(self):
        """(i, j) and (j, i) are different edges and both survive."""
        senders = jnp.array([0, 1])
        receivers = jnp.array([1, 0])
        s, r, _ = gnn.coalesce(senders, receivers)
        npt.assert_array_equal(s, jnp.array([0, 1]))
        npt.assert_array_equal(r, jnp.array([1, 0]))

    def test_empty_graph(self):
        """Works on a graph with no edges."""
        empty = jnp.array([], dtype=jnp.int32)
        s, r, kept = gnn.coalesce(empty, empty)
        assert s.shape[0] == 0
        assert r.shape[0] == 0
        assert kept.shape[0] == 0


class TestToUndirected:
    def test_adds_missing_reverses(self):
        """Edges without a reverse gain one; existing pairs are left alone."""
        senders = jnp.array([0, 1, 1])
        receivers = jnp.array([1, 0, 2])
        s, r, kept = gnn.to_undirected(senders, receivers)
        npt.assert_array_equal(s, jnp.array([0, 1, 1, 2]))
        npt.assert_array_equal(r, jnp.array([1, 0, 2, 1]))
        npt.assert_array_equal(kept, jnp.array([0, 1, 2, 5]))

    def test_output_is_symmetric(self):
        """Every edge in the result has its reverse in the result."""
        senders = jnp.array([0, 1, 3, 2])
        receivers = jnp.array([1, 2, 0, 3])
        s, r, _ = gnn.to_undirected(senders, receivers)
        forward = {(int(a), int(b)) for a, b in zip(s, r)}
        assert forward == {(b, a) for a, b in forward}

    def test_idempotent(self):
        """Coalescing means a symmetric graph passes through unchanged."""
        senders = jnp.array([0, 1, 1])
        receivers = jnp.array([1, 0, 2])
        s, r, _ = gnn.to_undirected(senders, receivers)
        s2, r2, _ = gnn.to_undirected(s, r)
        npt.assert_array_equal(s2, s)
        npt.assert_array_equal(r2, r)

    def test_discards_direction(self):
        """Opposite orientations of the same edge give the same result."""
        forward = gnn.to_undirected(jnp.array([0]), jnp.array([1]))
        backward = gnn.to_undirected(jnp.array([1]), jnp.array([0]))
        npt.assert_array_equal(forward[0], backward[0])
        npt.assert_array_equal(forward[1], backward[1])

    def test_self_loop_appears_once(self):
        """A self-loop is its own reverse and is not duplicated."""
        senders = jnp.array([1])
        receivers = jnp.array([1])
        s, r, _ = gnn.to_undirected(senders, receivers)
        npt.assert_array_equal(s, jnp.array([1]))
        npt.assert_array_equal(r, jnp.array([1]))

    def test_copies_edge_features(self):
        """Symmetric features are duplicated onto the reverse edge."""
        senders = jnp.array([0, 1, 1])
        receivers = jnp.array([1, 0, 2])
        x_edge = jnp.array([[10.0], [20.0], [30.0]])
        _, _, kept = gnn.to_undirected(senders, receivers)
        x_edge = jnp.concatenate([x_edge, x_edge])[kept]
        npt.assert_allclose(x_edge.ravel(), jnp.array([10.0, 20.0, 30.0, 30.0]))

    def test_negates_directional_features(self):
        """Direction-dependent features flip sign on the reverse edge."""
        senders = jnp.array([0, 1, 1])
        receivers = jnp.array([1, 0, 2])
        x_edge = jnp.array([[10.0], [20.0], [30.0]])
        _, _, kept = gnn.to_undirected(senders, receivers)
        x_edge = jnp.concatenate([x_edge, -x_edge])[kept]
        npt.assert_allclose(x_edge.ravel(), jnp.array([10.0, 20.0, 30.0, -30.0]))

    def test_empty_graph(self):
        """Works on a graph with no edges."""
        empty = jnp.array([], dtype=jnp.int32)
        s, r, kept = gnn.to_undirected(empty, empty)
        assert s.shape[0] == 0
        assert kept.shape[0] == 0


class TestLineGraph:
    def test_path_and_fork(self):
        """An edge connects to every edge leaving the node it lands on."""
        senders = jnp.array([0, 1, 1])
        receivers = jnp.array([1, 2, 3])
        ls, lr, shared = gnn.line_graph(senders, receivers)
        npt.assert_array_equal(ls, jnp.array([0, 0]))
        npt.assert_array_equal(lr, jnp.array([1, 2]))
        npt.assert_array_equal(shared, jnp.array([1, 1]))

    def test_matches_join_definition(self):
        """Pairs are exactly those where one edge ends where the other begins."""
        senders = jnp.array([0, 1, 2, 1])
        receivers = jnp.array([1, 2, 0, 0])
        ls, lr, _ = gnn.line_graph(senders, receivers, non_backtracking=False)
        pairs = sorted(zip(ls.tolist(), lr.tolist()))
        expected = sorted((a, b) for a in range(4) for b in range(4) if receivers[a] == senders[b])
        assert pairs == expected

    def test_non_backtracking_drops_reverse_pairs(self):
        """The i -> v -> i pairs are removed, and only those."""
        senders = jnp.array([0, 1])
        receivers = jnp.array([1, 0])
        ls, lr, _ = gnn.line_graph(senders, receivers, non_backtracking=False)
        assert sorted(zip(ls.tolist(), lr.tolist())) == [(0, 1), (1, 0)]
        ls, lr, _ = gnn.line_graph(senders, receivers, non_backtracking=True)
        assert ls.shape[0] == 0

    def test_shared_is_the_pivot_node(self):
        """Each pair meets at the returned node."""
        senders = jnp.array([0, 1, 2, 1])
        receivers = jnp.array([1, 2, 0, 0])
        ls, lr, shared = gnn.line_graph(senders, receivers)
        npt.assert_array_equal(receivers[ls], shared)
        npt.assert_array_equal(senders[lr], shared)

    def test_triangle_splits_by_orientation(self):
        """A symmetric triangle gives two disjoint 3-cycles, one per orientation."""
        senders = jnp.array([0, 1, 2])
        receivers = jnp.array([1, 2, 0])
        s, r, _ = gnn.to_undirected(senders, receivers)
        ls, lr, _ = gnn.line_graph(s, r)
        assert s.shape[0] == 6
        assert ls.shape[0] == 6

    def test_size_matches_degree_product(self):
        """Backtracking pairs number sum(indeg * outdeg) over nodes."""
        senders = jnp.array([0, 1, 2, 1, 0])
        receivers = jnp.array([1, 2, 0, 0, 2])
        ls, _, _ = gnn.line_graph(senders, receivers, non_backtracking=False)
        in_degree = gnn.degree(receivers, 3)
        out_degree = gnn.degree(senders, 3)
        assert ls.shape[0] == int((in_degree * out_degree).sum())

    def test_isolated_and_leaf_edges(self):
        """Edges landing on a node with no outgoing edges connect to nothing."""
        senders = jnp.array([0, 2])
        receivers = jnp.array([1, 3])
        ls, lr, shared = gnn.line_graph(senders, receivers)
        assert ls.shape[0] == 0
        assert lr.shape[0] == 0
        assert shared.shape[0] == 0

    def test_edge_features_become_node_features(self):
        """Line-graph nodes are edge rows, so x_edge passes through unchanged."""
        senders = jnp.array([0, 1, 1])
        receivers = jnp.array([1, 2, 3])
        x_edge = jnp.array([[1.0], [2.0], [3.0]])
        ls, lr, _ = gnn.line_graph(senders, receivers)
        out = gnn.segment_sum(x_edge[ls], lr, x_edge.shape[0])
        npt.assert_allclose(out.ravel(), jnp.array([0.0, 1.0, 1.0]))


class TestToAdjacency:
    def test_marks_present_edges(self):
        """Entry (i, j) is one exactly when the edge i -> j exists."""
        senders = jnp.array([0, 1, 2])
        receivers = jnp.array([1, 2, 0])
        adjacency = gnn.to_adjacency(senders, receivers, 3)
        expected = jnp.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        npt.assert_allclose(adjacency, expected)

    def test_isolated_nodes_give_empty_rows(self):
        """A node with no edges contributes an all-zero row and column."""
        senders = jnp.array([0])
        receivers = jnp.array([1])
        adjacency = gnn.to_adjacency(senders, receivers, 3)
        npt.assert_allclose(adjacency[2], jnp.zeros(3))
        npt.assert_allclose(adjacency[:, 2], jnp.zeros(3))

    def test_duplicate_edges_collapse(self):
        """Repeated edges set the same entry, so the matrix stays binary."""
        senders = jnp.array([0, 0, 0])
        receivers = jnp.array([1, 1, 1])
        adjacency = gnn.to_adjacency(senders, receivers, 2)
        npt.assert_allclose(adjacency, jnp.array([[0.0, 1.0], [0.0, 0.0]]))

    def test_direction_is_sender_to_receiver(self):
        """A one-way edge fills one triangle only."""
        adjacency = gnn.to_adjacency(jnp.array([0]), jnp.array([1]), 2)
        assert adjacency[0, 1] == 1.0
        assert adjacency[1, 0] == 0.0

    def test_undirected_input_is_symmetric(self):
        """Adding reverse edges first gives a symmetric matrix."""
        senders, receivers, _ = gnn.to_undirected(jnp.array([0, 1]), jnp.array([1, 2]))
        adjacency = gnn.to_adjacency(senders, receivers, 3)
        npt.assert_allclose(adjacency, adjacency.T)

    def test_degree_matches_row_and_column_sums(self):
        """Row sums are out-degree and column sums are in-degree."""
        senders = jnp.array([0, 0, 1])
        receivers = jnp.array([1, 2, 2])
        adjacency = gnn.to_adjacency(senders, receivers, 3)
        npt.assert_allclose(adjacency.sum(1), gnn.degree(senders, 3))
        npt.assert_allclose(adjacency.sum(0), gnn.degree(receivers, 3))

    def test_jits(self):
        """Output shape comes from num_nodes, so the op traces."""
        senders = jnp.array([0, 1])
        receivers = jnp.array([1, 2])
        adjacency = jax.jit(gnn.to_adjacency, static_argnums=2)(senders, receivers, 3)
        assert adjacency.shape == (3, 3)


class TestFromAdjacency:
    def test_recovers_edges(self):
        """Nonzero entries come back as edges, sorted by (sender, receiver)."""
        adjacency = jnp.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        senders, receivers = gnn.from_adjacency(adjacency)
        npt.assert_array_equal(senders, jnp.array([0, 1, 2]))
        npt.assert_array_equal(receivers, jnp.array([1, 2, 0]))

    def test_round_trips_coalesced_edges(self):
        """Canonical edge lists survive a trip through the dense form."""
        senders, receivers, _ = gnn.coalesce(jnp.array([2, 0, 1]), jnp.array([0, 1, 2]))
        s, r = gnn.from_adjacency(gnn.to_adjacency(senders, receivers, 3))
        npt.assert_array_equal(s, senders)
        npt.assert_array_equal(r, receivers)

    def test_empty_graph(self):
        """A zero matrix yields no edges."""
        senders, receivers = gnn.from_adjacency(jnp.zeros((3, 3)))
        assert senders.shape[0] == 0
        assert receivers.shape[0] == 0

    def test_num_edges_pads_out_of_range(self):
        """Spare slots hold the index num_nodes, one past the last node."""
        adjacency = jnp.array([[0.0, 1.0], [0.0, 0.0]])
        senders, receivers = gnn.from_adjacency(adjacency, 3)
        npt.assert_array_equal(senders, jnp.array([0, 2, 2]))
        npt.assert_array_equal(receivers, jnp.array([1, 2, 2]))

    def test_num_edges_truncates(self):
        """Edges beyond the requested count are dropped."""
        adjacency = jnp.ones((2, 2))
        senders, receivers = gnn.from_adjacency(adjacency, 2)
        npt.assert_array_equal(senders, jnp.array([0, 0]))
        npt.assert_array_equal(receivers, jnp.array([0, 1]))

    def test_padding_drops_out_of_segment_reductions(self):
        """Padded edges scatter nowhere, so aggregation ignores them."""
        adjacency = jnp.array([[0.0, 1.0], [0.0, 0.0]])
        senders, receivers = gnn.from_adjacency(adjacency, 4)
        x_node = jnp.array([[1.0], [2.0]])
        out = gnn.segment_sum(x_node[senders], receivers, 2)
        npt.assert_allclose(out, jnp.array([[0.0], [1.0]]))

    def test_jits_with_num_edges(self):
        """A static edge count makes the op traceable."""
        adjacency = jnp.array([[0.0, 1.0], [1.0, 0.0]])
        senders, receivers = jax.jit(gnn.from_adjacency, static_argnums=1)(adjacency, 2)
        npt.assert_array_equal(senders, jnp.array([0, 1]))
        npt.assert_array_equal(receivers, jnp.array([1, 0]))

    def test_any_nonzero_counts_as_an_edge(self):
        """Weighted matrices give the same topology as binary ones."""
        adjacency = jnp.array([[0.0, 0.5], [2.0, 0.0]])
        senders, receivers = gnn.from_adjacency(adjacency)
        npt.assert_array_equal(senders, jnp.array([0, 1]))
        npt.assert_array_equal(receivers, jnp.array([1, 0]))


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


class TestSegmentVar:
    def test_matches_jnp_var(self):
        """Output matches jnp.var applied per segment."""
        data = jnp.array([1.0, 2.0, 3.0, 4.0, 6.0])
        segment_ids = jnp.array([0, 0, 0, 1, 1])
        result = gnn.segment_var(data, segment_ids, num_segments=2)
        expected = jnp.array([jnp.var(data[:3]), jnp.var(data[3:])])
        npt.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_multidimensional_data(self):
        """Works with (e, d) shaped data."""
        data = jax.random.normal(jax.random.key(0), (6, 4))
        segment_ids = jnp.array([0, 0, 0, 1, 1, 1])
        result = gnn.segment_var(data, segment_ids, num_segments=2)
        npt.assert_allclose(result[0], data[:3].var(axis=0), rtol=1e-5, atol=1e-5)
        npt.assert_allclose(result[1], data[3:].var(axis=0), rtol=1e-5, atol=1e-5)

    def test_empty_and_singleton_segments(self):
        """Segments with no members or one member give zeros, not NaN."""
        data = jnp.array([1.0, 2.0, 7.0])
        segment_ids = jnp.array([0, 0, 2])
        result = gnn.segment_var(data, segment_ids, num_segments=3)
        npt.assert_allclose(result, jnp.array([0.25, 0.0, 0.0]), rtol=1e-5, atol=1e-5)

    def test_large_offset_stable(self):
        """A large constant offset leaves the variance unchanged."""
        data = jnp.array([1.0, 2.0, 3.0]) + 1e6
        segment_ids = jnp.zeros(3, dtype=jnp.int32)
        result = gnn.segment_var(data, segment_ids, num_segments=1)
        npt.assert_allclose(result, jnp.array([2.0 / 3.0]), rtol=1e-4, atol=1e-4)

    def test_jit_compatible(self):
        """segment_var works under jax.jit."""
        data = jnp.array([1.0, 2.0, 3.0])
        segment_ids = jnp.array([0, 0, 1])
        eager = gnn.segment_var(data, segment_ids, 2)
        jitted = jax.jit(gnn.segment_var, static_argnums=2)(data, segment_ids, 2)
        npt.assert_allclose(eager, jitted, atol=1e-6)

    @pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
    def test_mixed_precision(self, dtype):
        """Accumulates in float32 while preserving the input dtype."""
        data = jnp.concatenate((jnp.zeros(2048, dtype=dtype), jnp.ones(2048, dtype=dtype)))
        segment_ids = jnp.zeros(4096, dtype=jnp.int32)
        result = gnn.segment_var(data, segment_ids, num_segments=1)

        assert result.dtype == dtype
        npt.assert_array_equal(result, jnp.array([0.25], dtype=dtype))


class TestSegmentStd:
    def test_matches_jnp_std(self):
        """Output matches jnp.std applied per segment."""
        data = jnp.array([1.0, 2.0, 3.0, 4.0, 6.0])
        segment_ids = jnp.array([0, 0, 0, 1, 1])
        result = gnn.segment_std(data, segment_ids, num_segments=2)
        expected = jnp.array([jnp.std(data[:3]), jnp.std(data[3:])])
        npt.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_is_sqrt_of_var(self):
        """Standard deviation is the square root of the variance."""
        data = jax.random.normal(jax.random.key(0), (6, 4))
        segment_ids = jnp.array([0, 0, 0, 1, 1, 1])
        std = gnn.segment_std(data, segment_ids, num_segments=2)
        var = gnn.segment_var(data, segment_ids, num_segments=2)
        npt.assert_allclose(std, jnp.sqrt(var), rtol=1e-5, atol=1e-5)

    def test_identical_values_finite(self):
        """Segments whose values are all equal give zero, not NaN."""
        data = jnp.array([3.0, 3.0, 3.0, 5.0])
        segment_ids = jnp.array([0, 0, 0, 2])
        result = gnn.segment_std(data, segment_ids, num_segments=3)
        assert jnp.all(jnp.isfinite(result))
        npt.assert_allclose(result, jnp.zeros(3), atol=1e-6)

    def test_jit_compatible(self):
        """segment_std works under jax.jit."""
        data = jnp.array([1.0, 2.0, 3.0])
        segment_ids = jnp.array([0, 0, 1])
        eager = gnn.segment_std(data, segment_ids, 2)
        jitted = jax.jit(gnn.segment_std, static_argnums=2)(data, segment_ids, 2)
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
        from_numpy = gnn.batch_graphs(xs, senders_list, receivers_list)  # pyright: ignore[reportArgumentType]
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
            x, senders, receivers, graph_ids, num_nodes=8, num_edges=4, num_graphs=2
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
