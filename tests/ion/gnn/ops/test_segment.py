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

