import jax
import jax.numpy as jnp
import numpy.testing as npt

from ion import nn


class TestTransformerBlock:
    def test_output_shape(self):
        """Output shape matches input shape."""
        enc = nn.TransformerBlock(32, num_heads=4, key=jax.random.key(0))
        x = jnp.ones((5, 32))
        y = enc(x)
        assert y.shape == (5, 32)

    def test_output_shape_batched(self):
        """Batch dimensions are preserved."""
        enc = nn.TransformerBlock(32, num_heads=4, key=jax.random.key(0))
        x = jnp.ones((2, 5, 32))
        y = enc(x)
        assert y.shape == (2, 5, 32)

    def test_output_shape_multi_batch(self):
        """Multiple batch dimensions are preserved."""
        enc = nn.TransformerBlock(32, num_heads=4, key=jax.random.key(0))
        x = jnp.ones((3, 2, 5, 32))
        y = enc(x)
        assert y.shape == (3, 2, 5, 32)

    def test_custom_ff_dim(self):
        """Custom feed-forward dimension is respected."""
        enc = nn.TransformerBlock(32, num_heads=4, ff_dim=64, key=jax.random.key(0))
        assert enc.ff_1.w.shape == (32, 64)
        assert enc.ff_2.w.shape == (64, 32)
        x = jnp.ones((5, 32))
        y = enc(x)
        assert y.shape == (5, 32)

    def test_default_ff_dim(self):
        """Default feed-forward dimension is 4 * dim."""
        enc = nn.TransformerBlock(32, num_heads=4, key=jax.random.key(0))
        assert enc.ff_1.w.shape == (32, 128)
        assert enc.ff_2.w.shape == (128, 32)

    def test_residual_connection(self):
        """Output differs from input (non-trivial transformation)."""
        enc = nn.TransformerBlock(32, num_heads=4, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (5, 32))
        y = enc(x)
        assert not jnp.allclose(x, y)

    def test_deterministic(self):
        """Same input produces same output."""
        enc = nn.TransformerBlock(32, num_heads=4, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (5, 32))
        y1 = enc(x)
        y2 = enc(x)
        npt.assert_array_equal(y1, y2)

    def test_bias(self):
        """Bias mode creates attention and linear biases."""
        enc = nn.TransformerBlock(32, num_heads=4, bias=True, key=jax.random.key(0))
        assert enc.ff_1.b is not None
        assert enc.ff_2.b is not None
        assert enc.att.b_out is not None

    def test_no_bias(self):
        """No-bias mode creates no biases."""
        enc = nn.TransformerBlock(32, num_heads=4, bias=False, key=jax.random.key(0))
        assert enc.ff_1.b is None
        assert enc.ff_2.b is None
        assert enc.att.b_out is None

    def test_causal_output_shape(self):
        """Causal mode preserves output shape."""
        enc = nn.TransformerBlock(32, num_heads=4, causal=True, key=jax.random.key(0))
        x = jnp.ones((5, 32))
        y = enc(x)
        assert y.shape == (5, 32)

    def test_causal_differs_from_non_causal(self):
        """Causal and non-causal produce different outputs."""
        enc = nn.TransformerBlock(32, num_heads=4, causal=False, key=jax.random.key(0))
        enc_causal = nn.TransformerBlock(32, num_heads=4, causal=True, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (5, 32))
        y = enc(x)
        y_causal = enc_causal(x)
        assert not jnp.allclose(y, y_causal)

    def test_custom_w_init(self):
        """Custom w_init is threaded to attention and ff layers."""
        default = nn.TransformerBlock(32, num_heads=4, key=jax.random.key(0))
        custom = nn.TransformerBlock(
            32, num_heads=4, w_init=jax.nn.initializers.ones, key=jax.random.key(0)
        )
        assert not jnp.allclose(default.att.w_qkv.value, custom.att.w_qkv.value)
        assert not jnp.allclose(default.ff_1.w.value, custom.ff_1.w.value)
        assert not jnp.allclose(default.ff_2.w.value, custom.ff_2.w.value)

    def test_mask_output_shape(self):
        """Mask is threaded through; output shape is preserved."""
        block = nn.TransformerBlock(32, num_heads=4, key=jax.random.key(0))
        x = jnp.ones((5, 32))
        mask = jnp.ones((5, 5), dtype=bool)
        y = block(x, mask=mask)
        assert y.shape == (5, 32)

    def test_mask_affects_output(self):
        """Different masks produce different outputs."""
        block = nn.TransformerBlock(32, num_heads=4, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (5, 32))
        mask_full = jnp.ones((5, 5), dtype=bool)
        mask_partial = jnp.ones((5, 5), dtype=bool).at[0, 1].set(False)
        y1 = block(x, mask=mask_full)
        y2 = block(x, mask=mask_partial)
        assert not jnp.allclose(y1, y2)


class TestCrossTransformerBlock:
    def test_output_shape(self):
        """Output shape matches query input shape."""
        dec = nn.CrossTransformerBlock(32, num_heads=4, key=jax.random.key(0))
        x = jnp.ones((5, 32))
        ctx = jnp.ones((8, 32))
        y = dec(x, ctx)
        assert y.shape == (5, 32)

    def test_output_shape_batched(self):
        """Batch dimensions are preserved."""
        dec = nn.CrossTransformerBlock(32, num_heads=4, key=jax.random.key(0))
        x = jnp.ones((2, 5, 32))
        ctx = jnp.ones((2, 8, 32))
        y = dec(x, ctx)
        assert y.shape == (2, 5, 32)

    def test_output_shape_multi_batch(self):
        """Multiple batch dimensions are preserved."""
        dec = nn.CrossTransformerBlock(32, num_heads=4, key=jax.random.key(0))
        x = jnp.ones((3, 2, 5, 32))
        ctx = jnp.ones((3, 2, 8, 32))
        y = dec(x, ctx)
        assert y.shape == (3, 2, 5, 32)

    def test_custom_ff_dim(self):
        """Custom feed-forward dimension is respected."""
        dec = nn.CrossTransformerBlock(32, num_heads=4, ff_dim=64, key=jax.random.key(0))
        assert dec.ff_1.w.shape == (32, 64)
        assert dec.ff_2.w.shape == (64, 32)
        x = jnp.ones((5, 32))
        ctx = jnp.ones((8, 32))
        y = dec(x, ctx)
        assert y.shape == (5, 32)

    def test_default_ff_dim(self):
        """Default feed-forward dimension is 4 * dim."""
        dec = nn.CrossTransformerBlock(32, num_heads=4, key=jax.random.key(0))
        assert dec.ff_1.w.shape == (32, 128)
        assert dec.ff_2.w.shape == (128, 32)

    def test_different_context_length(self):
        """Context sequence length can differ from query length."""
        dec = nn.CrossTransformerBlock(32, num_heads=4, key=jax.random.key(0))
        x = jnp.ones((5, 32))
        ctx = jnp.ones((20, 32))
        y = dec(x, ctx)
        assert y.shape == (5, 32)

    def test_context_affects_output(self):
        """Different contexts produce different outputs."""
        dec = nn.CrossTransformerBlock(32, num_heads=4, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (5, 32))
        ctx1 = jax.random.normal(jax.random.key(2), (8, 32))
        ctx2 = jax.random.normal(jax.random.key(3), (8, 32))
        y1 = dec(x, ctx1)
        y2 = dec(x, ctx2)
        assert not jnp.allclose(y1, y2)

    def test_deterministic(self):
        """Same inputs produce same output."""
        dec = nn.CrossTransformerBlock(32, num_heads=4, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (5, 32))
        ctx = jax.random.normal(jax.random.key(2), (8, 32))
        y1 = dec(x, ctx)
        y2 = dec(x, ctx)
        npt.assert_array_equal(y1, y2)

    def test_bias(self):
        """Bias mode creates attention and linear biases."""
        dec = nn.CrossTransformerBlock(32, num_heads=4, bias=True, key=jax.random.key(0))
        assert dec.ff_1.b is not None
        assert dec.ff_2.b is not None
        assert dec.att.b_out is not None

    def test_no_bias(self):
        """No-bias mode creates no biases."""
        dec = nn.CrossTransformerBlock(32, num_heads=4, bias=False, key=jax.random.key(0))
        assert dec.ff_1.b is None
        assert dec.ff_2.b is None
        assert dec.att.b_out is None

    def test_custom_w_init(self):
        """Custom w_init is threaded to attention and ff layers."""
        default = nn.CrossTransformerBlock(32, num_heads=4, key=jax.random.key(0))
        custom = nn.CrossTransformerBlock(
            32, num_heads=4, w_init=jax.nn.initializers.ones, key=jax.random.key(0)
        )
        assert not jnp.allclose(default.att.w_q.value, custom.att.w_q.value)
        assert not jnp.allclose(default.ff_1.w.value, custom.ff_1.w.value)
        assert not jnp.allclose(default.ff_2.w.value, custom.ff_2.w.value)

    def test_mask_output_shape(self):
        """Mask is threaded through; output shape is preserved."""
        block = nn.CrossTransformerBlock(32, num_heads=4, key=jax.random.key(0))
        x = jnp.ones((5, 32))
        ctx = jnp.ones((8, 32))
        mask = jnp.ones((5, 8), dtype=bool)
        y = block(x, ctx, mask=mask)
        assert y.shape == (5, 32)

    def test_mask_affects_output(self):
        """Different masks produce different outputs."""
        block = nn.CrossTransformerBlock(32, num_heads=4, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (5, 32))
        ctx = jax.random.normal(jax.random.key(2), (8, 32))
        mask_full = jnp.ones((5, 8), dtype=bool)
        mask_partial = jnp.ones((5, 8), dtype=bool).at[0, 0].set(False)
        y1 = block(x, ctx, mask=mask_full)
        y2 = block(x, ctx, mask=mask_partial)
        assert not jnp.allclose(y1, y2)
