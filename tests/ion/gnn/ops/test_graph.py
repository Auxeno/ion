import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from ion import gnn


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
        s, r, kept = gnn.remove_self_loops(senders, receivers)
        npt.assert_array_equal(s, jnp.array([0, 1]))
        npt.assert_array_equal(r, jnp.array([1, 2]))
        npt.assert_array_equal(kept, jnp.array([0, 2]))

    def test_preserves_order(self):
        """Surviving edges keep their original relative order."""
        senders = jnp.array([2, 0, 1, 1])
        receivers = jnp.array([2, 1, 1, 0])
        s, r, kept = gnn.remove_self_loops(senders, receivers)
        npt.assert_array_equal(s, jnp.array([0, 1]))
        npt.assert_array_equal(r, jnp.array([1, 0]))
        npt.assert_array_equal(kept, jnp.array([1, 3]))

    def test_no_self_loops_is_identity(self):
        """A graph without self-loops passes through unchanged."""
        senders = jnp.array([0, 2, 1])
        receivers = jnp.array([1, 0, 2])
        s, r, kept = gnn.remove_self_loops(senders, receivers)
        npt.assert_array_equal(s, senders)
        npt.assert_array_equal(r, receivers)
        npt.assert_array_equal(kept, jnp.arange(3))

    def test_all_self_loops(self):
        """Removing every edge yields empty arrays."""
        senders = jnp.array([0, 1, 2])
        receivers = jnp.array([0, 1, 2])
        s, r, kept = gnn.remove_self_loops(senders, receivers)
        assert s.shape[0] == 0
        assert r.shape[0] == 0
        assert kept.shape[0] == 0

    def test_empty_graph(self):
        """Works on a graph with no edges."""
        senders = jnp.array([], dtype=jnp.int32)
        receivers = jnp.array([], dtype=jnp.int32)
        s, r, kept = gnn.remove_self_loops(senders, receivers)
        assert s.shape[0] == 0
        assert r.shape[0] == 0
        assert kept.shape[0] == 0

    def test_inverts_add_self_loops(self):
        """Removing after adding recovers the original edges."""
        senders = jnp.array([0, 2, 1])
        receivers = jnp.array([1, 0, 2])
        s, r = gnn.add_self_loops(senders, receivers, num_nodes=3)
        s, r, _ = gnn.remove_self_loops(s, r)
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


class TestInducedSubgraph:
    def test_induced_edges_and_relabelling(self):
        """Edges between selected nodes are retained and relabelled by node order."""
        senders = jnp.array([0, 1, 2, 2, 3, 4, 3])
        receivers = jnp.array([1, 2, 0, 3, 4, 2, 0])

        s, r, node_ids, edge_ids = gnn.induced_subgraph(
            senders, receivers, jnp.array([2, 0, 3]), num_nodes=5
        )

        npt.assert_array_equal(s, jnp.array([0, 0, 2]))
        npt.assert_array_equal(r, jnp.array([1, 2, 1]))
        npt.assert_array_equal(node_ids, jnp.array([2, 0, 3]))
        npt.assert_array_equal(edge_ids, jnp.array([2, 3, 6]))

    def test_indices_align_features(self):
        """Returned original indices select the matching node and edge features."""
        senders = jnp.array([0, 1, 2, 2, 3, 4, 3])
        receivers = jnp.array([1, 2, 0, 3, 4, 2, 0])
        x = jnp.arange(10).reshape(5, 2)
        x_edge = 10 * jnp.arange(7)

        _, _, node_ids, edge_ids = gnn.induced_subgraph(
            senders, receivers, jnp.array([2, 0, 3]), 5
        )

        npt.assert_array_equal(x[node_ids], x[jnp.array([2, 0, 3])])
        npt.assert_array_equal(x_edge[edge_ids], jnp.array([20, 30, 60]))

    def test_isolated_node_is_preserved(self):
        """Selected nodes remain present even when no selected edge touches them."""
        s, r, node_ids, edge_ids = gnn.induced_subgraph(
            jnp.array([0, 1]), jnp.array([1, 2]), jnp.array([3]), 4
        )

        npt.assert_array_equal(node_ids, jnp.array([3]))
        assert s.shape == r.shape == edge_ids.shape == (0,)

    def test_empty_selection(self):
        """Selecting no nodes produces an empty subgraph."""
        empty = jnp.array([], dtype=jnp.int32)
        s, r, node_ids, edge_ids = gnn.induced_subgraph(
            jnp.array([0, 1]), jnp.array([1, 2]), empty, 3
        )

        assert s.shape == r.shape == node_ids.shape == edge_ids.shape == (0,)

    def test_node_ids_must_be_unique(self):
        """Repeated node IDs would make relabelling ambiguous."""
        senders = jnp.array([0])
        receivers = jnp.array([1])

        with pytest.raises(ValueError, match="duplicates"):
            gnn.induced_subgraph(senders, receivers, jnp.array([0, 0]), 2)

    def test_numpy_inputs_preserve_topology_dtype(self):
        """Host arrays return JAX arrays without changing the topology dtype."""
        senders = np.array([0, 1, 2], dtype=np.int32)
        receivers = np.array([1, 2, 0], dtype=np.int32)
        node_ids = np.array([2, 0], dtype=np.int32)

        result = gnn.induced_subgraph(senders, receivers, node_ids, 3)

        assert all(isinstance(array, jax.Array) for array in result)
        assert result[0].dtype == result[1].dtype == jnp.int32
        npt.assert_array_equal(result[2], jnp.array([2, 0]))

class TestKHopSubgraph:
    senders = jnp.array([0, 2, 3, 1, 4, 6, 2])
    receivers = jnp.array([1, 1, 2, 4, 5, 5, 4])

    def test_incoming_hops(self):
        """Incoming traversal follows sender dependencies one frontier at a time."""
        s, r, node_ids, edge_ids = gnn.k_hop_subgraph(
            self.senders, self.receivers, jnp.array([1]), 2, 7
        )

        npt.assert_array_equal(node_ids, jnp.array([1, 0, 2, 3]))
        npt.assert_array_equal(s, jnp.array([1, 2, 3]))
        npt.assert_array_equal(r, jnp.array([0, 0, 2]))
        npt.assert_array_equal(edge_ids, jnp.array([0, 1, 2]))

    def test_outgoing_hops(self):
        """Outgoing traversal follows receivers reached from the frontier."""
        s, r, node_ids, edge_ids = gnn.k_hop_subgraph(
            self.senders, self.receivers, jnp.array([1]), 2, 7, direction="out"
        )

        npt.assert_array_equal(node_ids, jnp.array([1, 4, 5]))
        npt.assert_array_equal(s, jnp.array([0, 1]))
        npt.assert_array_equal(r, jnp.array([1, 2]))
        npt.assert_array_equal(edge_ids, jnp.array([3, 4]))

    def test_both_directions_returns_induced_edges(self):
        """Both-direction traversal includes every edge among discovered nodes."""
        _, _, node_ids, edge_ids = gnn.k_hop_subgraph(
            self.senders, self.receivers, jnp.array([1]), 1, 7, direction="both"
        )

        npt.assert_array_equal(node_ids, jnp.array([1, 0, 2, 4]))
        npt.assert_array_equal(edge_ids, jnp.array([0, 1, 3, 6]))

    def test_starting_nodes_come_first(self):
        """Starting node order is retained before nodes from later hops."""
        _, _, node_ids, _ = gnn.k_hop_subgraph(
            self.senders, self.receivers, jnp.array([5, 1]), 1, 7
        )

        npt.assert_array_equal(node_ids, jnp.array([5, 1, 0, 2, 4, 6]))

    def test_zero_hops(self):
        """Zero hops returns the induced graph over the starting nodes alone."""
        s, r, node_ids, edge_ids = gnn.k_hop_subgraph(
            self.senders, self.receivers, jnp.array([1, 4]), 0, 7
        )

        npt.assert_array_equal(node_ids, jnp.array([1, 4]))
        npt.assert_array_equal(s, jnp.array([0]))
        npt.assert_array_equal(r, jnp.array([1]))
        npt.assert_array_equal(edge_ids, jnp.array([3]))

    def test_empty_nodes(self):
        """An empty starting set produces an empty subgraph."""
        empty = jnp.array([], dtype=jnp.int32)
        s, r, node_ids, edge_ids = gnn.k_hop_subgraph(
            self.senders, self.receivers, empty, 2, 7
        )

        assert s.shape == r.shape == node_ids.shape == edge_ids.shape == (0,)

    def test_invalid_configuration(self):
        """Invalid hop counts, directions, and node sets raise clear errors."""
        with pytest.raises(ValueError, match="non-negative"):
            gnn.k_hop_subgraph(self.senders, self.receivers, jnp.array([1]), -1, 7)
        with pytest.raises(ValueError, match="direction"):
            gnn.k_hop_subgraph(
                self.senders,
                self.receivers,
                jnp.array([1]),
                1,
                7,
                direction="sideways",  # pyright: ignore[reportArgumentType]
            )
        with pytest.raises(ValueError, match="duplicates"):
            gnn.k_hop_subgraph(self.senders, self.receivers, jnp.array([1, 1]), 1, 7)

    def test_numpy_inputs(self):
        """Host arrays follow the same traversal and return JAX arrays."""
        result = gnn.k_hop_subgraph(
            np.asarray(self.senders),
            np.asarray(self.receivers),
            np.array([1], dtype=np.int32),
            2,
            7,
        )

        assert all(isinstance(array, jax.Array) for array in result)
        npt.assert_array_equal(result[2], jnp.array([1, 0, 2, 3]))


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

    def test_edge_capacity_pads_out_of_range(self):
        """Spare slots hold the index num_nodes, one past the last node."""
        adjacency = jnp.array([[0.0, 1.0], [0.0, 0.0]])
        senders, receivers = gnn.from_adjacency(adjacency, 3)
        npt.assert_array_equal(senders, jnp.array([0, 2, 2]))
        npt.assert_array_equal(receivers, jnp.array([1, 2, 2]))

    def test_edge_capacity_truncates(self):
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

    def test_jits_with_edge_capacity(self):
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

