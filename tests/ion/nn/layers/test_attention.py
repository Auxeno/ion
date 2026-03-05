import jax
import jax.numpy as jnp
import numpy.testing as npt

from ion import nn


class TestSelfAttention:
    def test_causal_mask(self):
        """Causal attention: jacobian upper triangle is zero (future masked)."""
        layer = nn.SelfAttention(8, num_heads=1, causal=True, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (4, 8))
        jac = jax.jacobian(layer)(x)  # (4, 8, 4, 8)
        # Sum over output and input feature dims to get (seq, seq)
        jac_seq = jnp.sum(jnp.abs(jac), axis=(1, 3))  # (4, 4)
        # Upper triangle (future positions) should be zero
        upper = jnp.triu(jac_seq, k=1)
        npt.assert_allclose(upper, 0.0, atol=1e-5)

    def test_non_causal_full_jacobian(self):
        """Non-causal attention: all sequence positions influence each other."""
        layer = nn.SelfAttention(8, num_heads=1, causal=False, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (4, 8))
        jac = jax.jacobian(layer)(x)  # (4, 8, 4, 8)
        jac_seq = jnp.sum(jnp.abs(jac), axis=(1, 3))  # (4, 4)
        # All positions should have non-zero influence
        assert jnp.all(jac_seq > 1e-6)

    def test_multi_head(self):
        """Various num_heads values all produce correct output shape."""
        for num_heads in [1, 2, 4]:
            layer = nn.SelfAttention(8, num_heads=num_heads, key=jax.random.key(0))
            x = jnp.ones((4, 8))
            y = layer(x)
            assert y.shape == (4, 8)

    def test_with_bias(self):
        """Output bias is present and output shape is correct when bias=True."""
        layer = nn.SelfAttention(8, num_heads=2, bias=True, key=jax.random.key(0))
        assert layer.b_out is not None
        x = jnp.ones((4, 8))
        y = layer(x)
        assert y.shape == (4, 8)

    def test_without_bias(self):
        """Output bias is None when bias=False."""
        layer = nn.SelfAttention(8, num_heads=2, bias=False, key=jax.random.key(0))
        assert layer.b_out is None

    def test_truncated_normal_init(self):
        """Truncated normal init gives std close to 0.02 with no values beyond 2 sigma."""
        layer = nn.SelfAttention(256, num_heads=4, key=jax.random.key(42))
        std = jnp.std(layer.w_qkv.value)
        npt.assert_allclose(std, 0.02, atol=0.005)
        # Truncated normal: no values beyond 2 sigma
        assert jnp.all(jnp.abs(layer.w_qkv.value) <= 0.04 + 1e-6)

    def test_weight_dtype(self):
        """Weights match the requested dtype."""
        layer = nn.SelfAttention(8, num_heads=2, dtype=jnp.float32, key=jax.random.key(0))
        assert layer.w_qkv.dtype == jnp.float32
        assert layer.w_out.dtype == jnp.float32


class TestCrossAttention:
    def test_output_shape(self):
        """Output shape matches query sequence shape."""
        layer = nn.CrossAttention(8, num_heads=2, key=jax.random.key(0))
        x = jnp.ones((4, 8))
        context = jnp.ones((6, 8))
        y = layer(x, context)
        assert y.shape == (4, 8)

    def test_different_sequence_lengths(self):
        """Query and context can have different sequence lengths."""
        layer = nn.CrossAttention(8, num_heads=2, key=jax.random.key(0))
        for s, t in [(3, 7), (1, 10), (5, 5)]:
            x = jnp.ones((s, 8))
            context = jnp.ones((t, 8))
            y = layer(x, context)
            assert y.shape == (s, 8)

    def test_multi_head(self):
        """Various num_heads values all produce correct output shape."""
        for num_heads in [1, 2, 4]:
            layer = nn.CrossAttention(8, num_heads=num_heads, key=jax.random.key(0))
            x = jnp.ones((4, 8))
            context = jnp.ones((6, 8))
            y = layer(x, context)
            assert y.shape == (4, 8)

    def test_with_bias(self):
        """Output bias is present and output shape is correct when bias=True."""
        layer = nn.CrossAttention(8, num_heads=2, bias=True, key=jax.random.key(0))
        assert layer.b_out is not None
        x = jnp.ones((4, 8))
        context = jnp.ones((6, 8))
        y = layer(x, context)
        assert y.shape == (4, 8)

    def test_without_bias(self):
        """Output bias is None when bias=False."""
        layer = nn.CrossAttention(8, num_heads=2, bias=False, key=jax.random.key(0))
        assert layer.b_out is None

    def test_context_influences_output(self):
        """Changing context changes the output (verifies cross-attention wiring)."""
        layer = nn.CrossAttention(8, num_heads=2, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (4, 8))
        ctx1 = jax.random.normal(jax.random.key(2), (6, 8))
        ctx2 = jax.random.normal(jax.random.key(3), (6, 8))
        y1 = layer(x, ctx1)
        y2 = layer(x, ctx2)
        assert not jnp.allclose(y1, y2)

    def test_weight_dtype(self):
        """Weights match the requested dtype."""
        layer = nn.CrossAttention(8, num_heads=2, dtype=jnp.float32, key=jax.random.key(0))
        assert layer.w_q.dtype == jnp.float32
        assert layer.w_kv.dtype == jnp.float32
        assert layer.w_out.dtype == jnp.float32
