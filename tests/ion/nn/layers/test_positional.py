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

    def test_default_dtype(self):
        """Weights default to float32."""
        pos = nn.LearnedPositionalEmbedding(128, 64, key=jax.random.key(0))
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

    def test_slope_values(self):
        """Slopes follow the paper's geometric sequence 2^(-8i/h)."""
        # bias[h, 1, 0] = slope[h] * (0 - 1) = -slope[h]
        b = nn.alibi(16, 4)
        npt.assert_allclose(b[:, 1, 0], -(0.5 ** (8.0 * jnp.arange(1, 5) / 4)), rtol=1e-6)
        b = nn.alibi(16, 8)
        npt.assert_allclose(b[:, 1, 0], -(0.5 ** jnp.arange(1, 9)), rtol=1e-6)

    def test_non_power_of_2_raises(self):
        """Non-power-of-2 num_heads raises ValueError."""
        import pytest

        with pytest.raises(ValueError):
            nn.alibi(16, 3)


class TestRoPE:
    def test_output_manual(self):
        """Output matches a hand-rolled per-pair 2D rotation."""
        import numpy as np

        rope = nn.RoPE()
        x = jax.random.normal(jax.random.key(0), (4, 6))
        expected = np.zeros((4, 6))
        for m in range(4):
            for i in range(3):
                angle = m / (10_000.0 ** (2 * i / 6))
                x0, x1 = x[m, 2 * i], x[m, 2 * i + 1]
                expected[m, 2 * i] = x0 * np.cos(angle) - x1 * np.sin(angle)
                expected[m, 2 * i + 1] = x0 * np.sin(angle) + x1 * np.cos(angle)
        npt.assert_allclose(rope(x), expected, atol=1e-6)

    def test_output_shape(self):
        """Output shape matches input shape, including leading batch dims."""
        rope = nn.RoPE()
        assert rope(jnp.ones((16, 8))).shape == (16, 8)
        assert rope(jnp.ones((2, 3, 16, 8))).shape == (2, 3, 16, 8)

    def test_identity_at_position_zero(self):
        """At position 0 the rotation angle is zero, so output equals input."""
        rope = nn.RoPE()
        x = jax.random.normal(jax.random.key(0), (16, 8))
        npt.assert_allclose(rope(x)[0], x[0], atol=1e-6)

    def test_preserves_norm(self):
        """RoPE is a rotation, so it preserves vector norms."""
        rope = nn.RoPE()
        x = jax.random.normal(jax.random.key(0), (16, 8))
        npt.assert_allclose(
            jnp.linalg.norm(rope(x), axis=-1), jnp.linalg.norm(x, axis=-1), atol=1e-5
        )

    def test_relative_positions(self):
        """Dot products between rotated Q and K depend only on relative offset."""
        rope = nn.RoPE()
        keys = jax.random.split(jax.random.key(0))
        u = jax.random.normal(keys[0], (8,))
        v = jax.random.normal(keys[1], (8,))
        q = rope(jnp.tile(u, (16, 1)))
        k = rope(jnp.tile(v, (16, 1)))
        # Offset of 3 at two different absolute positions
        npt.assert_allclose(q[2] @ k[5], q[9] @ k[12], atol=1e-5)

    def test_theta(self):
        """Different theta values produce different rotations."""
        x = jax.random.normal(jax.random.key(0), (16, 8))
        assert not jnp.allclose(nn.RoPE()(x), nn.RoPE(theta=500.0)(x))

    def test_no_params(self):
        """RoPE has no trainable parameters."""
        assert nn.RoPE().num_params == 0

    def test_input_dtype(self):
        """Output dtype follows input dtype."""
        rope = nn.RoPE()
        assert rope(jnp.ones((16, 8), dtype=jnp.bfloat16)).dtype == jnp.bfloat16
        assert rope(jnp.ones((16, 8))).dtype == jnp.float32
