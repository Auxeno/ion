import jax
import jax.numpy as jnp
import numpy.testing as npt

from ion.gnn import add_self_loops, segment_softmax


class TestSegmentSoftmax:
    def test_sums_to_one_per_segment(self):
        """Each segment's weights sum to 1 after normalization."""
        data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        segment_ids = jnp.array([0, 0, 0, 1, 1])
        result = segment_softmax(data, segment_ids, num_segments=2)
        seg_0_sum = result[:3].sum()
        seg_1_sum = result[3:].sum()
        npt.assert_allclose(seg_0_sum, 1.0, atol=1e-5)
        npt.assert_allclose(seg_1_sum, 1.0, atol=1e-5)

    def test_single_segment_matches_softmax(self):
        """With one segment, result matches regular softmax."""
        data = jnp.array([1.0, 2.0, 3.0])
        segment_ids = jnp.array([0, 0, 0])
        result = segment_softmax(data, segment_ids, num_segments=1)
        expected = jax.nn.softmax(data)
        npt.assert_allclose(result, expected, atol=1e-5)

    def test_preserves_relative_order(self):
        """Larger values get larger weights within each segment."""
        data = jnp.array([1.0, 3.0, 2.0])
        segment_ids = jnp.array([0, 0, 0])
        result = segment_softmax(data, segment_ids, num_segments=1)
        assert result[1] > result[2] > result[0]

    def test_large_values_stable(self):
        """Large input values produce finite output (no overflow)."""
        data = jnp.array([1000.0, 1001.0, 999.0])
        segment_ids = jnp.array([0, 0, 0])
        result = segment_softmax(data, segment_ids, num_segments=1)
        assert jnp.all(jnp.isfinite(result))
        npt.assert_allclose(result.sum(), 1.0, atol=1e-5)

    def test_multidimensional_data(self):
        """Works with (e, h) shaped data for multi-head attention."""
        data = jax.random.normal(jax.random.key(0), (6, 4))
        segment_ids = jnp.array([0, 0, 0, 1, 1, 1])
        result = segment_softmax(data, segment_ids, num_segments=2)
        # Each head in each segment sums to 1
        for head in range(4):
            npt.assert_allclose(result[:3, head].sum(), 1.0, atol=1e-5)
            npt.assert_allclose(result[3:, head].sum(), 1.0, atol=1e-5)

    def test_jit_compatible(self):
        """segment_softmax works under jax.jit."""
        data = jnp.array([1.0, 2.0, 3.0])
        segment_ids = jnp.array([0, 0, 0])
        eager = segment_softmax(data, segment_ids, 1)
        jitted = jax.jit(segment_softmax, static_argnums=2)(data, segment_ids, 1)
        npt.assert_allclose(eager, jitted, atol=1e-6)


class TestAddSelfLoops:
    def test_output_length(self):
        """Output has num_nodes extra edges appended."""
        senders = jnp.array([0, 1])
        receivers = jnp.array([1, 0])
        s, r = add_self_loops(senders, receivers, num_nodes=3)
        assert s.shape[0] == 2 + 3
        assert r.shape[0] == 2 + 3

    def test_self_loop_content(self):
        """Appended edges are (0->0), (1->1), ..., (n-1->n-1)."""
        senders = jnp.array([0, 1])
        receivers = jnp.array([1, 0])
        s, r = add_self_loops(senders, receivers, num_nodes=3)
        # Last 3 edges are self-loops
        npt.assert_array_equal(s[2:], jnp.array([0, 1, 2]))
        npt.assert_array_equal(r[2:], jnp.array([0, 1, 2]))

    def test_preserves_original_edges(self):
        """Original edges are unchanged at the start of the array."""
        senders = jnp.array([0, 2, 1])
        receivers = jnp.array([1, 0, 2])
        s, r = add_self_loops(senders, receivers, num_nodes=3)
        npt.assert_array_equal(s[:3], senders)
        npt.assert_array_equal(r[:3], receivers)

    def test_empty_graph(self):
        """Works on a graph with no edges (self-loops only)."""
        senders = jnp.array([], dtype=jnp.int32)
        receivers = jnp.array([], dtype=jnp.int32)
        s, r = add_self_loops(senders, receivers, num_nodes=4)
        assert s.shape[0] == 4
        npt.assert_array_equal(s, jnp.arange(4))
        npt.assert_array_equal(r, jnp.arange(4))
