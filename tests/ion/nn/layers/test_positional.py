import jax
import jax.numpy as jnp
import numpy.testing as npt

from ion import nn


class TestLearnedPositionalEmbedding:
    def test_output_shape(self):
        """Output shape matches input shape."""
        pos = nn.LearnedPositionalEmbedding(128, 64, key=jax.random.key(0))
        x = jnp.ones((10, 64))
        assert pos(x).shape == (10, 64)

    def test_output_shape_batched(self):
        """Batched input preserves leading dimensions."""
        pos = nn.LearnedPositionalEmbedding(128, 64, key=jax.random.key(0))
        x = jnp.ones((2, 10, 64))
        assert pos(x).shape == (2, 10, 64)

    def test_adds_to_input(self):
        """Output differs from input (embeddings are added)."""
        pos = nn.LearnedPositionalEmbedding(128, 64, key=jax.random.key(0))
        x = jnp.ones((10, 64))
        assert not jnp.allclose(pos(x), x)

    def test_shorter_sequence(self):
        """Sequences shorter than max_len are handled by slicing."""
        pos = nn.LearnedPositionalEmbedding(128, 64, key=jax.random.key(0))
        x_short = jnp.ones((5, 64))
        x_long = jnp.ones((10, 64))
        # First 5 positions should get the same embeddings
        npt.assert_allclose(pos(x_short), pos(x_long)[:5], rtol=0, atol=0)

    def test_weight_shape(self):
        """Weight matrix has shape (max_len, dim)."""
        pos = nn.LearnedPositionalEmbedding(128, 64, key=jax.random.key(0))
        assert pos.w.shape == (128, 64)

    def test_weight_dtype(self):
        """Weights match the requested dtype."""
        pos = nn.LearnedPositionalEmbedding(128, 64, dtype=jnp.float32, key=jax.random.key(0))
        assert pos.w.dtype == jnp.float32

    def test_sequence_exceeds_max_len_raises(self):
        """Input longer than max_len raises ValueError."""
        import pytest

        pos = nn.LearnedPositionalEmbedding(10, 64, key=jax.random.key(0))
        x = jnp.ones((20, 64))
        with pytest.raises(ValueError, match="exceeds max_len"):
            pos(x)

    def test_different_keys(self):
        """Different PRNG keys produce different weights."""
        p1 = nn.LearnedPositionalEmbedding(128, 64, key=jax.random.key(0))
        p2 = nn.LearnedPositionalEmbedding(128, 64, key=jax.random.key(1))
        assert not jnp.array_equal(p1.w._value, p2.w._value)


class TestSinusoidal:
    def test_odd_dim_raises(self):
        """Odd dim raises ValueError."""
        import pytest

        with pytest.raises(ValueError, match="even"):
            nn.sinusoidal(128, 63)

    def test_output_shape(self):
        """Output shape is (seq_len, dim)."""
        assert nn.sinusoidal(128, 64).shape == (128, 64)

    def test_values_bounded(self):
        """All values are in [-1, 1]."""
        e = nn.sinusoidal(128, 64)
        assert jnp.all(e >= -1.0) and jnp.all(e <= 1.0)

    def test_sin_cos_columns(self):
        """Even columns are sin, odd columns are cos (position 0: sin=0, cos=1)."""
        e = nn.sinusoidal(128, 64)
        npt.assert_allclose(e[0, 0::2], 0.0, atol=1e-6)  # sin(0) = 0
        npt.assert_allclose(e[0, 1::2], 1.0, atol=1e-6)  # cos(0) = 1

    def test_dtype(self):
        """Output respects requested dtype."""
        assert nn.sinusoidal(128, 64, dtype=jnp.float32).dtype == jnp.float32


class TestAlibi:
    def test_output_shape(self):
        """Output shape is (num_heads, seq_len, seq_len)."""
        assert nn.alibi(128, 8).shape == (8, 128, 128)

    def test_diagonal_zero(self):
        """Diagonal entries (self-bias) are zero."""
        b = nn.alibi(16, 4)
        for h in range(4):
            npt.assert_allclose(jnp.diag(b[h]), 0.0, atol=1e-6)

    def test_slopes_decrease(self):
        """Later heads have smaller slopes (check bias at fixed offset)."""
        b = nn.alibi(16, 4)
        # bias[h, 0, 1] = slope[h] * 1, so magnitudes should decrease
        magnitudes = jnp.abs(b[:, 0, 1])
        assert jnp.all(magnitudes[:-1] >= magnitudes[1:])

    def test_non_power_of_2_raises(self):
        """Non-power-of-2 num_heads raises ValueError."""
        import pytest

        with pytest.raises(ValueError):
            nn.alibi(16, 3)


class TestRope:
    def test_output_shapes(self):
        """Returns two arrays of shape (seq_len, head_dim)."""
        cos, sin = nn.rope(128, 64)
        assert cos.shape == (128, 64)
        assert sin.shape == (128, 64)

    def test_values_bounded(self):
        """Cos and sin values are in [-1, 1]."""
        cos, sin = nn.rope(128, 64)
        assert jnp.all(cos >= -1.0) and jnp.all(cos <= 1.0)
        assert jnp.all(sin >= -1.0) and jnp.all(sin <= 1.0)

    def test_position_zero(self):
        """At position 0, cos=1 and sin=0 for all dimensions."""
        cos, sin = nn.rope(128, 64)
        npt.assert_allclose(cos[0], 1.0, atol=1e-6)
        npt.assert_allclose(sin[0], 0.0, atol=1e-6)

    def test_odd_head_dim_raises(self):
        """Odd head_dim raises ValueError."""
        import pytest

        with pytest.raises(ValueError):
            nn.rope(128, 63)


class TestApplyRope:
    def test_output_shape(self):
        """Output shape matches input shape."""
        cos, sin = nn.rope(16, 8)
        x = jnp.ones((16, 8))
        assert nn.apply_rope(x, cos, sin).shape == (16, 8)

    def test_identity_at_position_zero(self):
        """At position 0 (cos=1, sin=0), output equals input."""
        cos, sin = nn.rope(16, 8)
        x = jax.random.normal(jax.random.key(0), (16, 8))
        y = nn.apply_rope(x, cos, sin)
        npt.assert_allclose(y[0], x[0], atol=1e-6)

    def test_preserves_norm(self):
        """RoPE is a rotation, so it preserves vector norms."""
        cos, sin = nn.rope(16, 8)
        x = jax.random.normal(jax.random.key(0), (16, 8))
        y = nn.apply_rope(x, cos, sin)
        npt.assert_allclose(jnp.linalg.norm(y, axis=-1), jnp.linalg.norm(x, axis=-1), atol=1e-5)
