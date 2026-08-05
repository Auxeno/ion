"""Flax NNX GPT benchmark."""

from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float, Int

from ...configs import ModelConfig


def sinusoidal(length: int, dim: int) -> Array:
    position = jnp.arange(length)[:, None]
    frequency = jnp.exp(jnp.arange(0, dim, 2) * (-jnp.log(10_000.0) / dim))
    encoding = jnp.zeros((length, dim))
    encoding = encoding.at[:, 0::2].set(jnp.sin(position * frequency))
    return encoding.at[:, 1::2].set(jnp.cos(position * frequency[: dim // 2]))


def attention_fn(query, key, value, mask=None, *, implementation, **kwargs):
    """Flax attention_fn selecting a JAX dot-product attention backend."""
    return jax.nn.dot_product_attention(query, key, value, mask=mask, implementation=implementation)


class TransformerBlock(nnx.Module):
    """Pre-norm causal transformer block."""

    def __init__(self, dim: int, num_heads: int, implementation: str, *, rngs: nnx.Rngs) -> None:
        self.attention = nnx.MultiHeadAttention(
            num_heads,
            dim,
            use_bias=False,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
            decode=False,
            attention_fn=partial(attention_fn, implementation=implementation),
            rngs=rngs,
        )
        self.attention_norm = nnx.LayerNorm(
            dim, dtype=jnp.bfloat16, param_dtype=jnp.float32, rngs=rngs
        )
        self.mlp_norm = nnx.LayerNorm(dim, dtype=jnp.bfloat16, param_dtype=jnp.float32, rngs=rngs)
        self.mlp_in = nnx.Linear(
            dim,
            4 * dim,
            use_bias=False,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
            rngs=rngs,
        )
        self.mlp_out = nnx.Linear(
            4 * dim,
            dim,
            use_bias=False,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
            rngs=rngs,
        )

    def __call__(self, x: Float[Array, "b s d"]) -> Float[Array, "b s d"]:
        normalized = self.attention_norm(x)
        mask = nnx.make_causal_mask(jnp.ones(x.shape[:-1], dtype=bool), dtype=jnp.bool_)
        x = x + self.attention(normalized, mask=mask)
        return x + self.mlp_out(jax.nn.gelu(self.mlp_in(self.mlp_norm(x))))


class GPT(nnx.Module):
    """Decoder-only transformer with tied token embeddings."""

    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs) -> None:
        self.embedding = nnx.Embed(
            config.vocab_size,
            config.width,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
            rngs=rngs,
        )
        self.position = nnx.data(sinusoidal(config.seq_len, config.width))

        # cuDNN flash attention is unavailable off GPU, so fall back to the XLA backend
        implementation = "cudnn" if config.use_flash and jax.default_backend() == "gpu" else "xla"
        self.blocks = nnx.List(
            TransformerBlock(config.width, config.num_heads, implementation, rngs=rngs)
            for _ in range(config.depth)
        )
        self.norm = nnx.LayerNorm(
            config.width,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
            rngs=rngs,
        )

    def __call__(self, tokens: Int[Array, "b s"]) -> Float[Array, "b s vocab"]:
        x = self.embedding(tokens) + self.position[: tokens.shape[1]]
        for block in self.blocks:
            x = block(x)
        return self.norm(x) @ self.embedding.embedding.value.T


def create_model(config: ModelConfig, *, seed: int) -> GPT:
    """Create the benchmark model."""
    return GPT(config, rngs=nnx.Rngs(seed))


def forward(model: GPT, inputs: Array) -> Array:
    """Apply the model to a batch."""
    return model(inputs)
