from functools import partial

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from ion import nn


class TestMultiHeadAttention:
    def test_causal_mask(self):
        """Causal attention: jacobian upper triangle is zero (future masked)."""
        layer = nn.MultiHeadAttention(8, num_heads=1, causal=True, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (1, 4, 8))
        jac = jax.jacobian(layer)(x)  # (1, 4, 8, 1, 4, 8)
        # Sum over output and input feature dims to get (seq, seq)
        jac_seq = jnp.sum(jnp.abs(jac), axis=(0, 2, 3, 5))  # (4, 4)
        # Upper triangle (future positions) should be zero
        upper = jnp.triu(jac_seq, k=1)
        npt.assert_allclose(upper, 0.0, atol=1e-5)

    def test_non_causal_full_jacobian(self):
        """Non-causal attention: all sequence positions influence each other."""
        layer = nn.MultiHeadAttention(8, num_heads=1, causal=False, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (1, 4, 8))
        jac = jax.jacobian(layer)(x)  # (1, 4, 8, 1, 4, 8)
        jac_seq = jnp.sum(jnp.abs(jac), axis=(0, 2, 3, 5))  # (4, 4)
        # All positions should have non-zero influence
        assert jnp.all(jac_seq > 1e-6)

    def test_multi_head(self):
        """Various num_heads values all produce correct output shape."""
        for num_heads in [1, 2, 4]:
            layer = nn.MultiHeadAttention(8, num_heads=num_heads, key=jax.random.key(0))
            x = jnp.ones((1, 4, 8))
            y = layer(x)
            assert y.shape == (1, 4, 8)

    def test_with_bias(self):
        """Output bias is present and output shape is correct when use_bias=True."""
        layer = nn.MultiHeadAttention(8, num_heads=2, use_bias=True, key=jax.random.key(0))
        assert layer.b_out is not None
        x = jnp.ones((1, 4, 8))
        y = layer(x)
        assert y.shape == (1, 4, 8)

    def test_without_bias(self):
        """Output bias is None when use_bias=False."""
        layer = nn.MultiHeadAttention(8, num_heads=2, use_bias=False, key=jax.random.key(0))
        assert layer.b_out is None

    def test_glorot_uniform_init(self):
        """Glorot uniform init gives std close to sqrt(2/(fan_in + fan_out))."""
        layer = nn.MultiHeadAttention(256, num_heads=4, key=jax.random.key(42))
        std = jnp.std(layer.w_q._value)
        expected_std = (2 / (256 + 256)) ** 0.5
        npt.assert_allclose(std, expected_std, rtol=0.1)

    def test_variance_scaling_init_fans(self):
        """Variance-scaling init sees the true (dim, features) fans despite the head split."""
        he = jax.nn.initializers.he_normal()
        layer = nn.MultiHeadAttention(64, num_heads=8, w_init=he, key=jax.random.key(0))
        expected_std = (2 / 64) ** 0.5
        npt.assert_allclose(jnp.std(layer.w_q._value), expected_std, rtol=0.15)
        npt.assert_allclose(jnp.std(layer.w_k._value), expected_std, rtol=0.15)
        npt.assert_allclose(jnp.std(layer.w_v._value), expected_std, rtol=0.15)
        npt.assert_allclose(jnp.std(layer.w_out._value), expected_std, rtol=0.15)

    def test_default_dtype(self):
        """Weights default to float32."""
        layer = nn.MultiHeadAttention(8, num_heads=2, key=jax.random.key(0))
        assert layer.w_q.dtype == jnp.float32
        assert layer.w_k.dtype == jnp.float32
        assert layer.w_v.dtype == jnp.float32
        assert layer.w_out.dtype == jnp.float32

    def test_attention_fn_is_used(self):
        """A custom attention function receives projected QKV and call options."""
        implementation = None

        def attention(query, key, value, *, backend, **kwargs):
            nonlocal implementation
            implementation = backend
            return query

        layer = nn.MultiHeadAttention(
            8,
            num_heads=2,
            attention_fn=partial(attention, backend="cudnn"),
            key=jax.random.key(0),
        )
        layer(jnp.ones((1, 4, 8)))

        assert implementation == "cudnn"

    def test_dot_product_attention_with_rope(self):
        """The RoPE helper rotates query and key along their sequence axis."""
        keys = jax.random.split(jax.random.key(0), 3)
        query = jax.random.normal(keys[0], (2, 5, 2, 4))
        key = jax.random.normal(keys[1], (2, 5, 2, 4))
        value = jax.random.normal(keys[2], (2, 5, 2, 4))
        apply_rope = jax.vmap(nn.RoPE(axis=-2), in_axes=-2, out_axes=-2)

        expected = jax.nn.dot_product_attention(apply_rope(query), apply_rope(key), value)
        actual = nn.dot_product_attention_with_rope(query, key, value, rope=nn.RoPE())

        npt.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def test_mask_blocks_positions(self):
        """Masked positions have zero gradient (no information flow)."""
        layer = nn.MultiHeadAttention(8, num_heads=1, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (1, 4, 8))
        # Block position 2 from attending to position 0
        mask = jnp.ones((1, 4, 4), dtype=bool).at[0, 2, 0].set(False)
        jac = jax.jacobian(lambda x: layer(x, mask=mask))(x)
        # Gradient from position 0 input to position 2 output should be zero
        npt.assert_allclose(jac[0, 2, :, 0, 0, :], 0.0, atol=1e-5)

    def test_mask_with_causal(self):
        """Mask combines with causal: both causal future and masked positions are blocked."""
        layer = nn.MultiHeadAttention(8, num_heads=1, causal=True, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (1, 4, 8))
        # Additionally block position 1 from attending to position 0
        mask = jnp.ones((1, 4, 4), dtype=bool).at[0, 1, 0].set(False)
        jac = jax.jacobian(lambda x: layer(x, mask=mask))(x)
        jac_seq = jnp.sum(jnp.abs(jac), axis=(0, 2, 3, 5))
        # Causal upper triangle still zero
        upper = jnp.triu(jac_seq, k=1)
        npt.assert_allclose(upper, 0.0, atol=1e-5)
        # Additionally, position 1 should not depend on position 0
        npt.assert_allclose(jac_seq[1, 0], 0.0, atol=1e-5)

    def test_mask_per_head(self):
        """Per-head mask (... 1 s s) broadcasts across heads."""
        layer = nn.MultiHeadAttention(8, num_heads=2, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (1, 4, 8))
        mask = jnp.ones((1, 1, 4, 4), dtype=bool).at[0, 0, 0, 1].set(False)
        y = layer(x, mask=mask)
        assert y.shape == (1, 4, 8)

    def test_no_mask_unchanged(self):
        """mask=None produces identical output to omitting it."""
        layer = nn.MultiHeadAttention(8, num_heads=2, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (1, 4, 8))
        y1 = layer(x)
        y2 = layer(x, mask=None)
        npt.assert_array_equal(y1, y2)

    def test_mask_ss(self):
        """(s, s) mask broadcasts across batch and heads."""
        layer = nn.MultiHeadAttention(8, num_heads=2, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (2, 4, 8))
        mask = jnp.ones((4, 4), dtype=bool).at[0, 1].set(False)
        y = layer(x, mask=mask)
        assert y.shape == (2, 4, 8)

    def test_mask_bss(self):
        """(b, s, s) mask applies per batch, shared across heads."""
        layer = nn.MultiHeadAttention(8, num_heads=2, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (2, 4, 8))
        # b == num_heads: regression for rank-3 masks broadcasting onto the head axis
        mask = jnp.ones((2, 4, 4), dtype=bool).at[0, 0, 1].set(False)
        y = layer(x, mask=mask)
        npt.assert_array_equal(y, layer(x, mask=mask[:, None]))

    def test_mask_bhss(self):
        """(b, h, s, s) per-head mask works with batched input."""
        layer = nn.MultiHeadAttention(8, num_heads=2, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (2, 4, 8))
        mask = jnp.ones((2, 2, 4, 4), dtype=bool).at[0, 0, 0, 1].set(False)
        y = layer(x, mask=mask)
        assert y.shape == (2, 4, 8)

    def test_fully_masked_row_zero(self):
        """A query row with no attendable positions contributes zero."""
        layer = nn.MultiHeadAttention(8, num_heads=2, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (2, 4, 8))
        mask = jnp.ones((2, 4, 4), dtype=bool).at[0, 2].set(False)
        y = layer(x, mask=mask)
        npt.assert_array_equal(y[0, 2], 0.0)

    def test_fully_masked_row_other_rows_unchanged(self):
        """Rows with attendable positions are unaffected by a fully masked row."""
        layer = nn.MultiHeadAttention(8, num_heads=2, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (2, 4, 8))
        mask = jnp.ones((2, 4, 4), dtype=bool).at[0, 2].set(False)
        y = layer(x, mask=mask)
        y_full = layer(x)
        npt.assert_allclose(y[0, :2], y_full[0, :2], atol=1e-6)
        npt.assert_allclose(y[0, 3], y_full[0, 3], atol=1e-6)
        npt.assert_allclose(y[1], y_full[1], atol=1e-6)

    def test_fully_masked_row_with_causal(self):
        """A fully masked query row contributes zero with causal masking."""
        layer = nn.MultiHeadAttention(8, num_heads=2, causal=True, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (2, 4, 8))
        mask = jnp.ones((2, 4, 4), dtype=bool).at[0, 2].set(False)
        y = layer(x, mask=mask)
        npt.assert_array_equal(y[0, 2], 0.0)

    def test_fully_masked_row_gradients_finite(self):
        """A fully masked query row has zero input gradient."""
        layer = nn.MultiHeadAttention(8, num_heads=2, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (2, 4, 8))
        mask = jnp.ones((2, 4, 4), dtype=bool).at[0, 2].set(False)
        grads = jax.grad(lambda x: layer(x, mask=mask)[0, 2].sum())(x)
        npt.assert_array_equal(grads, 0.0)

    def test_grouped_query_attention(self):
        """num_kv_heads below num_heads (GQA) gives fewer kv heads and correct output shape."""
        layer = nn.MultiHeadAttention(8, num_heads=4, num_kv_heads=2, key=jax.random.key(0))
        assert layer.w_q.shape == (8, 8)  # (dim, num_heads * head_dim)
        assert layer.w_k.shape == (8, 4)  # (dim, num_kv_heads * head_dim)
        assert layer.w_v.shape == (8, 4)
        x = jax.random.normal(jax.random.key(1), (2, 5, 8))
        assert layer(x).shape == (2, 5, 8)

    def test_multi_query_attention(self):
        """num_kv_heads=1 (MQA) shares a single kv head across all query heads."""
        layer = nn.MultiHeadAttention(8, num_heads=4, num_kv_heads=1, key=jax.random.key(0))
        assert layer.w_k.shape == (8, 2)
        assert layer.w_v.shape == (8, 2)
        x = jnp.ones((2, 5, 8))
        assert layer(x).shape == (2, 5, 8)

    def test_sliding_window(self):
        """Sliding window blocks attention between positions farther apart than the window."""
        layer = nn.MultiHeadAttention(8, num_heads=1, window=1, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (1, 5, 8))
        jac = jax.jacobian(layer)(x)  # (1, 5, 8, 1, 5, 8)
        jac_seq = jnp.sum(jnp.abs(jac), axis=(0, 2, 3, 5))  # (5, 5)
        # Positions more than 1 apart cannot influence each other
        ids = jnp.arange(5)
        far = jnp.abs(ids[:, None] - ids[None, :]) > 1
        npt.assert_allclose(jac_seq * far, 0.0, atol=1e-5)


class TestMultiHeadAttentionValidation:
    def test_dim_not_divisible_by_num_heads_raises(self):
        """dim must be divisible by num_heads."""
        with pytest.raises(ValueError, match="divisible"):
            nn.MultiHeadAttention(dim=7, num_heads=3, key=jax.random.key(0))

    def test_num_heads_not_divisible_by_num_kv_heads_raises(self):
        """num_heads must be divisible by num_kv_heads."""
        with pytest.raises(ValueError, match="divisible"):
            nn.MultiHeadAttention(dim=8, num_heads=4, num_kv_heads=3, key=jax.random.key(0))


class TestMultiHeadAttentionCross:
    def test_output_shape(self):
        """Output shape matches query sequence shape."""
        layer = nn.MultiHeadAttention(8, num_heads=2, key=jax.random.key(0))
        x = jnp.ones((1, 4, 8))
        x_kv = jnp.ones((1, 6, 8))
        y = layer(x, x_kv)
        assert y.shape == (1, 4, 8)

    def test_different_sequence_lengths(self):
        """Query and context can have different sequence lengths."""
        layer = nn.MultiHeadAttention(8, num_heads=2, key=jax.random.key(0))
        for s, t in [(3, 7), (1, 10), (5, 5)]:
            x = jnp.ones((1, s, 8))
            x_kv = jnp.ones((1, t, 8))
            y = layer(x, x_kv)
            assert y.shape == (1, s, 8)

    def test_multi_head(self):
        """Various num_heads values all produce correct output shape."""
        for num_heads in [1, 2, 4]:
            layer = nn.MultiHeadAttention(8, num_heads=num_heads, key=jax.random.key(0))
            x = jnp.ones((1, 4, 8))
            x_kv = jnp.ones((1, 6, 8))
            y = layer(x, x_kv)
            assert y.shape == (1, 4, 8)

    def test_with_bias(self):
        """Output bias is present and output shape is correct when use_bias=True."""
        layer = nn.MultiHeadAttention(8, num_heads=2, use_bias=True, key=jax.random.key(0))
        assert layer.b_out is not None
        x = jnp.ones((1, 4, 8))
        x_kv = jnp.ones((1, 6, 8))
        y = layer(x, x_kv)
        assert y.shape == (1, 4, 8)

    def test_without_bias(self):
        """Output bias is None when use_bias=False."""
        layer = nn.MultiHeadAttention(8, num_heads=2, use_bias=False, key=jax.random.key(0))
        assert layer.b_out is None

    def test_kv_dim(self):
        """kv_dim lets the context have a different feature dim than the query input."""
        layer = nn.MultiHeadAttention(8, num_heads=2, kv_dim=12, key=jax.random.key(0))
        assert layer.w_k.shape == (12, 8)  # (kv_dim, num_kv_heads * head_dim)
        assert layer.w_v.shape == (12, 8)
        x = jnp.ones((1, 4, 8))
        x_kv = jnp.ones((1, 6, 12))
        y = layer(x, x_kv)
        assert y.shape == (1, 4, 8)

    def test_kv_dim_defaults_to_dim(self):
        """kv_dim=None gives the same kv shapes as kv_dim=dim."""
        layer = nn.MultiHeadAttention(8, num_heads=2, key=jax.random.key(0))
        assert layer.w_k.shape == (8, 8)
        assert layer.w_v.shape == (8, 8)

    def test_x_kv_influences_output(self):
        """Changing the context changes the output (verifies cross-attention wiring)."""
        layer = nn.MultiHeadAttention(8, num_heads=2, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (1, 4, 8))
        x_kv1 = jax.random.normal(jax.random.key(2), (1, 6, 8))
        x_kv2 = jax.random.normal(jax.random.key(3), (1, 6, 8))
        y1 = layer(x, x_kv1)
        y2 = layer(x, x_kv2)
        assert not jnp.allclose(y1, y2)

    def test_default_dtype(self):
        """Weights default to float32."""
        layer = nn.MultiHeadAttention(8, num_heads=2, key=jax.random.key(0))
        assert layer.w_q.dtype == jnp.float32
        assert layer.w_k.dtype == jnp.float32
        assert layer.w_v.dtype == jnp.float32
        assert layer.w_out.dtype == jnp.float32

    def test_mask_blocks_context_positions(self):
        """Masked context positions have zero gradient (no information flow)."""
        layer = nn.MultiHeadAttention(8, num_heads=1, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (1, 3, 8))
        x_kv = jax.random.normal(jax.random.key(2), (1, 5, 8))
        # Block query position 0 from attending to context position 2
        mask = jnp.ones((1, 3, 5), dtype=bool).at[0, 0, 2].set(False)
        jac = jax.jacobian(lambda c: layer(x, c, mask=mask))(x_kv)
        # Gradient from context position 2 to query position 0 output should be zero
        npt.assert_allclose(jac[0, 0, :, 0, 2, :], 0.0, atol=1e-5)

    def test_mask_per_head(self):
        """Per-head mask (... 1 s t) broadcasts across heads."""
        layer = nn.MultiHeadAttention(8, num_heads=2, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (1, 3, 8))
        x_kv = jax.random.normal(jax.random.key(2), (1, 5, 8))
        mask = jnp.ones((1, 1, 3, 5), dtype=bool).at[0, 0, 0, 0].set(False)
        y = layer(x, x_kv, mask=mask)
        assert y.shape == (1, 3, 8)

    def test_no_mask_unchanged(self):
        """mask=None produces identical output to omitting it."""
        layer = nn.MultiHeadAttention(8, num_heads=2, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (1, 3, 8))
        x_kv = jax.random.normal(jax.random.key(2), (1, 5, 8))
        y1 = layer(x, x_kv)
        y2 = layer(x, x_kv, mask=None)
        npt.assert_array_equal(y1, y2)

    def test_mask_st(self):
        """(s, t) mask broadcasts across batch and heads."""
        layer = nn.MultiHeadAttention(8, num_heads=2, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (2, 3, 8))
        x_kv = jax.random.normal(jax.random.key(2), (2, 5, 8))
        mask = jnp.ones((3, 5), dtype=bool).at[0, 0].set(False)
        y = layer(x, x_kv, mask=mask)
        assert y.shape == (2, 3, 8)

    def test_mask_bst(self):
        """(b, s, t) mask applies per batch, shared across heads."""
        layer = nn.MultiHeadAttention(8, num_heads=2, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (2, 3, 8))
        x_kv = jax.random.normal(jax.random.key(2), (2, 5, 8))
        # b == num_heads: regression for rank-3 masks broadcasting onto the head axis
        mask = jnp.ones((2, 3, 5), dtype=bool).at[0, 0, 0].set(False)
        y = layer(x, x_kv, mask=mask)
        npt.assert_array_equal(y, layer(x, x_kv, mask=mask[:, None]))

    def test_mask_bhst(self):
        """(b, h, s, t) per-head mask works with batched input."""
        layer = nn.MultiHeadAttention(8, num_heads=2, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (2, 3, 8))
        x_kv = jax.random.normal(jax.random.key(2), (2, 5, 8))
        mask = jnp.ones((2, 2, 3, 5), dtype=bool).at[0, 0, 0, 0].set(False)
        y = layer(x, x_kv, mask=mask)
        assert y.shape == (2, 3, 8)

    def test_fully_masked_row_zero(self):
        """A query row with no attendable context positions contributes zero."""
        layer = nn.MultiHeadAttention(8, num_heads=2, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (2, 3, 8))
        x_kv = jax.random.normal(jax.random.key(2), (2, 5, 8))
        mask = jnp.ones((2, 3, 5), dtype=bool).at[0, 1].set(False)
        y = layer(x, x_kv, mask=mask)
        npt.assert_array_equal(y[0, 1], 0.0)

    def test_fully_masked_row_other_rows_unchanged(self):
        """Rows with attendable positions are unaffected by a fully masked row."""
        layer = nn.MultiHeadAttention(8, num_heads=2, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (2, 3, 8))
        x_kv = jax.random.normal(jax.random.key(2), (2, 5, 8))
        mask = jnp.ones((2, 3, 5), dtype=bool).at[0, 1].set(False)
        y = layer(x, x_kv, mask=mask)
        y_full = layer(x, x_kv)
        npt.assert_allclose(y[0, 0], y_full[0, 0], atol=1e-6)
        npt.assert_allclose(y[0, 2], y_full[0, 2], atol=1e-6)
        npt.assert_allclose(y[1], y_full[1], atol=1e-6)

    def test_fully_masked_row_gradients_finite(self):
        """A fully masked query row has zero context gradient."""
        layer = nn.MultiHeadAttention(8, num_heads=2, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (2, 3, 8))
        x_kv = jax.random.normal(jax.random.key(2), (2, 5, 8))
        mask = jnp.ones((2, 3, 5), dtype=bool).at[0, 1].set(False)
        grads = jax.grad(lambda c: layer(x, c, mask=mask)[0, 1].sum())(x_kv)
        npt.assert_array_equal(grads, 0.0)

    def test_x_kv_none_is_self_attention(self):
        """Omitting x_kv is identical to passing the input as its own context."""
        layer = nn.MultiHeadAttention(8, num_heads=2, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (2, 4, 8))
        npt.assert_array_equal(layer(x), layer(x, x))

    def test_grouped_query_attention(self):
        """num_kv_heads below num_heads applies to the context projections too."""
        layer = nn.MultiHeadAttention(
            8, num_heads=4, num_kv_heads=2, kv_dim=12, key=jax.random.key(0)
        )
        assert layer.w_k.shape == (12, 4)  # (kv_dim, num_kv_heads * head_dim)
        assert layer.w_v.shape == (12, 4)
        x = jax.random.normal(jax.random.key(1), (2, 3, 8))
        x_kv = jax.random.normal(jax.random.key(2), (2, 5, 12))
        assert layer(x, x_kv).shape == (2, 3, 8)

    def test_attention_fn_is_used(self):
        """A custom attention function receives the context-projected key and value."""
        shapes = None

        def attention(query, key, value, **kwargs):
            nonlocal shapes
            shapes = (query.shape, key.shape, value.shape)
            return query

        layer = nn.MultiHeadAttention(8, num_heads=2, attention_fn=attention, key=jax.random.key(0))
        layer(jnp.ones((1, 3, 8)), jnp.ones((1, 5, 8)))

        assert shapes == ((1, 3, 2, 4), (1, 5, 2, 4), (1, 5, 2, 4))
