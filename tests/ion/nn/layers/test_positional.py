from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

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
        npt.assert_array_equal(pos(x_short), pos(x_long)[:5])

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
        pos = nn.LearnedPositionalEmbedding(10, 64, key=jax.random.key(0))
        x = jnp.ones((20, 64))
        with pytest.raises(ValueError, match="exceeds max_len"):
            pos(x)

    def test_different_keys(self):
        """Different PRNG keys produce different weights."""
        p1 = nn.LearnedPositionalEmbedding(128, 64, key=jax.random.key(0))
        p2 = nn.LearnedPositionalEmbedding(128, 64, key=jax.random.key(1))
        assert not jnp.allclose(p1.w._value, p2.w._value)


class TestSinusoidalPositionalEmbedding:
    def test_odd_dim_raises(self):
        """Odd dim raises ValueError."""
        with pytest.raises(ValueError, match="even"):
            nn.SinusoidalPositionalEmbedding()(jnp.zeros((128, 63)))

    def test_output_shape(self):
        """Output shape matches input shape, including leading batch dims."""
        pos = nn.SinusoidalPositionalEmbedding()
        assert pos(jnp.zeros((128, 64))).shape == (128, 64)
        assert pos(jnp.zeros((2, 128, 64))).shape == (2, 128, 64)

    def test_values_bounded(self):
        """The encoding added to a zero input is in [-1, 1]."""
        e = nn.SinusoidalPositionalEmbedding()(jnp.zeros((128, 64)))
        assert jnp.all(e >= -1.0) and jnp.all(e <= 1.0)

    def test_sin_cos_columns(self):
        """Even columns are sin, odd columns are cos (position 0: sin=0, cos=1)."""
        e = nn.SinusoidalPositionalEmbedding()(jnp.zeros((128, 64)))
        npt.assert_allclose(e[0, 0::2], 0.0, atol=1e-6)  # sin(0) = 0
        npt.assert_allclose(e[0, 1::2], 1.0, atol=1e-6)  # cos(0) = 1

    def test_adds_to_input(self):
        """The encoding is added to the input rather than replacing it."""
        x = jax.random.normal(jax.random.key(0), (16, 8))
        pos = nn.SinusoidalPositionalEmbedding()
        npt.assert_allclose(pos(x) - x, pos(jnp.zeros_like(x)), atol=1e-6)

    def test_shorter_sequence(self):
        """Positions are absolute, so a prefix encodes the same as the full sequence."""
        pos = nn.SinusoidalPositionalEmbedding()
        npt.assert_allclose(pos(jnp.zeros((5, 8))), pos(jnp.zeros((10, 8)))[:5], atol=1e-6)

    def test_theta_changes_frequencies(self):
        """Larger theta lowers frequencies, slowing the encoding across positions."""
        x = jnp.zeros((128, 64))
        base = nn.SinusoidalPositionalEmbedding()(x)
        large = nn.SinusoidalPositionalEmbedding(theta=100_000.0)(x)
        assert not jnp.allclose(base, large)
        npt.assert_allclose(large[0], base[0], atol=1e-6)  # position 0 is theta independent

    def test_input_dtype(self):
        """Output dtype follows input dtype."""
        pos = nn.SinusoidalPositionalEmbedding()
        assert pos(jnp.zeros((128, 64), dtype=jnp.bfloat16)).dtype == jnp.bfloat16
        assert pos(jnp.zeros((128, 64))).dtype == jnp.float32

    def test_no_params(self):
        """Sinusoidal encodings have no trainable parameters."""
        assert nn.SinusoidalPositionalEmbedding().num_params == 0


class TestRoPE:
    def test_odd_head_dim_raises(self):
        """Odd head_dim raises ValueError."""
        with pytest.raises(ValueError, match="divisible"):
            nn.RoPE(axis=-2)(jnp.ones((4, 7)))

    def test_output_manual(self):
        """Output matches a hand-rolled per-pair 2D rotation."""
        rope = nn.RoPE(axis=-2)
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
        assert rope(jnp.ones((16, 3, 8))).shape == (16, 3, 8)
        assert rope(jnp.ones((2, 16, 3, 8))).shape == (2, 16, 3, 8)

    def test_default_axis(self):
        """The default sequence axis matches mapping RoPE over attention heads."""
        x = jax.random.normal(jax.random.key(0), (2, 5, 3, 8))
        expected = jax.vmap(nn.RoPE(axis=-2), in_axes=-2, out_axes=-2)(x)
        npt.assert_allclose(nn.RoPE()(x), expected, atol=1e-6)

    def test_identity_at_position_zero(self):
        """At position 0 the rotation angle is zero, so output equals input."""
        rope = nn.RoPE(axis=-2)
        x = jax.random.normal(jax.random.key(0), (16, 8))
        npt.assert_allclose(rope(x)[0], x[0], atol=1e-6)

    def test_preserves_norm(self):
        """RoPE is a rotation, so it preserves vector norms."""
        rope = nn.RoPE(axis=-2)
        x = jax.random.normal(jax.random.key(0), (16, 8))
        npt.assert_allclose(
            jnp.linalg.norm(rope(x), axis=-1), jnp.linalg.norm(x, axis=-1), atol=1e-5
        )

    def test_relative_positions(self):
        """Dot products between rotated Q and K depend only on relative offset."""
        rope = nn.RoPE(axis=-2)
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
        assert not jnp.allclose(nn.RoPE(axis=-2)(x), nn.RoPE(axis=-2, theta=500.0)(x))

    def test_no_params(self):
        """RoPE has no trainable parameters."""
        assert nn.RoPE().num_params == 0

    def test_input_dtype(self):
        """Output dtype follows input dtype."""
        rope = nn.RoPE(axis=-2)
        assert rope(jnp.ones((16, 8), dtype=jnp.bfloat16)).dtype == jnp.bfloat16
        assert rope(jnp.ones((16, 8))).dtype == jnp.float32


class TestRoPEShape:
    def test_1d_shape_matches_default(self):
        """An explicit 1D shape reproduces the implicit sequence positions exactly."""
        x = jax.random.normal(jax.random.key(0), (16, 8))
        npt.assert_array_equal(nn.RoPE(shape=(16,), axis=-2)(x), nn.RoPE(axis=-2)(x))

    def test_2d_output_shape(self):
        """A 2D lattice leaves the input shape untouched."""
        rope = nn.RoPE(shape=(4, 4))
        assert rope(jnp.ones((16, 3, 8))).shape == (16, 3, 8)
        assert rope(jnp.ones((2, 16, 3, 8))).shape == (2, 16, 3, 8)

    def test_2d_differs_from_1d(self):
        """Laying positions on a grid rotates differently to a flat sequence."""
        x = jax.random.normal(jax.random.key(0), (16, 8))
        assert not jnp.allclose(nn.RoPE(shape=(4, 4), axis=-2)(x), nn.RoPE(axis=-2)(x))

    def test_2d_relative_positions(self):
        """Dot products depend only on the offset between grid coordinates."""
        rope = nn.RoPE(shape=(4, 4), axis=-2)
        keys = jax.random.split(jax.random.key(0))
        u = jax.random.normal(keys[0], (8,))
        v = jax.random.normal(keys[1], (8,))
        q = rope(jnp.tile(u, (16, 1)))
        k = rope(jnp.tile(v, (16, 1)))
        # Offset of (1, 2) taken from two different absolute grid positions
        npt.assert_allclose(q[5] @ k[11], q[0] @ k[6], atol=1e-5)

    def test_2d_preserves_norm(self):
        """A 2D lattice rotation still preserves vector norms."""
        rope = nn.RoPE(shape=(4, 4), axis=-2)
        x = jax.random.normal(jax.random.key(0), (16, 8))
        npt.assert_allclose(
            jnp.linalg.norm(rope(x), axis=-1), jnp.linalg.norm(x, axis=-1), atol=1e-5
        )

    def test_3d_output_shape(self):
        """A 3D lattice works when head_dim divides by twice the axis count."""
        rope = nn.RoPE(shape=(2, 2, 2), axis=-2)
        assert rope(jnp.ones((8, 12))).shape == (8, 12)

    def test_head_dim_not_divisible_by_axes_raises(self):
        """head_dim must divide by 2 * len(shape), not just by 2."""
        with pytest.raises(ValueError, match="divisible"):
            nn.RoPE(shape=(4, 4), axis=-2)(jnp.ones((16, 6)))

    def test_position_count_mismatch_raises(self):
        """The lattice plus its prefix must fill the sequence exactly."""
        with pytest.raises(ValueError, match="does not fill sequence length"):
            nn.RoPE(shape=(4, 4), axis=-2)(jnp.ones((15, 8)))

    def test_empty_shape_raises(self):
        """An empty shape has no axes to split the head dimension across."""
        with pytest.raises(ValueError, match="at least one element"):
            nn.RoPE(shape=())


class TestRoPEPrefixTokens:
    def test_prefix_tokens_unrotated(self):
        """Prefix tokens sit at position 0, so they pass through unchanged."""
        rope = nn.RoPE(shape=(4, 4), num_prefix_tokens=1, axis=-2)
        x = jax.random.normal(jax.random.key(0), (17, 8))
        npt.assert_allclose(rope(x)[0], x[0], atol=1e-6)

    def test_prefix_shifts_grid(self):
        """Tokens after the prefix are rotated as the grid, not as the flat sequence."""
        rope = nn.RoPE(shape=(4, 4), num_prefix_tokens=1, axis=-2)
        x = jax.random.normal(jax.random.key(0), (17, 8))
        npt.assert_allclose(rope(x)[1:], nn.RoPE(shape=(4, 4), axis=-2)(x[1:]), atol=1e-6)

    def test_multiple_prefix_tokens(self):
        """Every prefix token is left unrotated, not just the first."""
        rope = nn.RoPE(shape=(4, 4), num_prefix_tokens=4, axis=-2)
        x = jax.random.normal(jax.random.key(0), (20, 8))
        npt.assert_allclose(rope(x)[:4], x[:4], atol=1e-6)

    def test_prefix_without_shape(self):
        """Without a shape the prefix still offsets the 1D sequence positions."""
        rope = nn.RoPE(num_prefix_tokens=2, axis=-2)
        x = jax.random.normal(jax.random.key(0), (18, 8))
        npt.assert_allclose(rope(x)[:2], x[:2], atol=1e-6)
        npt.assert_allclose(rope(x)[2:], nn.RoPE(axis=-2)(x[2:]), atol=1e-6)

    def test_attention_integration(self):
        """A 2D RoPE composes with MultiHeadAttention through a custom attention_fn."""
        def attention_with_rope(q, k, v, *, rope, **kwargs):
            return jax.nn.dot_product_attention(rope(q), rope(k), v, **kwargs)

        rope = nn.RoPE(shape=(4, 4), num_prefix_tokens=1)
        attn = nn.MultiHeadAttention(
            8,
            num_heads=2,
            attention_fn=partial(attention_with_rope, rope=rope),
            key=jax.random.key(0),
        )
        x = jax.random.normal(jax.random.key(1), (2, 17, 8))
        assert attn(x).shape == (2, 17, 8)
