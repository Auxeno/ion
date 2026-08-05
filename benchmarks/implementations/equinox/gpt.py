"""Equinox GPT benchmark."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray

from ...configs import ModelConfig


def linear(layer: eqx.nn.Linear, x: Array) -> Array:
    shape = x.shape
    x = jax.vmap(layer)(x.reshape(-1, shape[-1]))
    return x.reshape(*shape[:-1], -1)


def sinusoidal(length: int, dim: int) -> Array:
    position = jnp.arange(length)[:, None]
    frequency = jnp.exp(jnp.arange(0, dim, 2) * (-jnp.log(10_000.0) / dim))
    encoding = jnp.zeros((length, dim))
    encoding = encoding.at[:, 0::2].set(jnp.sin(position * frequency))
    return encoding.at[:, 1::2].set(jnp.cos(position * frequency[: dim // 2]))


class TransformerBlock(eqx.Module):
    """Pre-norm causal transformer block."""

    attention: eqx.nn.MultiheadAttention
    attention_norm: eqx.nn.LayerNorm
    mlp_norm: eqx.nn.LayerNorm
    mlp_in: eqx.nn.Linear
    mlp_out: eqx.nn.Linear

    def __init__(self, dim: int, num_heads: int, *, key: PRNGKeyArray) -> None:
        key_attention, key_in, key_out = jax.random.split(key, 3)
        self.attention = eqx.nn.MultiheadAttention(num_heads, dim, key=key_attention)
        self.attention_norm = eqx.nn.LayerNorm(dim)
        self.mlp_norm = eqx.nn.LayerNorm(dim)
        self.mlp_in = eqx.nn.Linear(dim, 4 * dim, use_bias=False, key=key_in)
        self.mlp_out = eqx.nn.Linear(4 * dim, dim, use_bias=False, key=key_out)

    def __call__(self, x: Float[Array, "s d"]) -> Float[Array, "s d"]:
        normalized = jax.vmap(self.attention_norm)(x)
        mask = jnp.tril(jnp.ones((x.shape[0], x.shape[0]), dtype=bool))
        x = x + self.attention(normalized, normalized, normalized, mask=mask)
        normalized = jax.vmap(self.mlp_norm)(x)
        hidden = jax.nn.gelu(linear(self.mlp_in, normalized))
        return x + linear(self.mlp_out, hidden)


class GPT(eqx.Module):
    """Decoder-only transformer with tied token embeddings."""

    embedding: eqx.nn.Embedding
    blocks: tuple[TransformerBlock, ...]
    norm: eqx.nn.LayerNorm
    seq_len: int
    dim: int

    def __init__(self, config: ModelConfig, *, key: PRNGKeyArray) -> None:
        keys = jax.random.split(key, config.depth + 1)
        self.embedding = eqx.nn.Embedding(config.vocab_size, config.width, key=keys[0])
        self.blocks = tuple(
            TransformerBlock(config.width, config.num_heads, key=keys[index + 1])
            for index in range(config.depth)
        )
        self.norm = eqx.nn.LayerNorm(config.width)
        self.seq_len = config.seq_len
        self.dim = config.width

    def __call__(self, tokens: Int[Array, " s"]) -> Float[Array, "s vocab"]:
        position = sinusoidal(self.seq_len, self.dim)
        x = jax.vmap(self.embedding)(tokens) + position[: tokens.shape[0]]
        for block in self.blocks:
            x = block(x)
        x = jax.vmap(self.norm)(x)
        return x @ self.embedding.weight.T


def create_model(config: ModelConfig, *, key: PRNGKeyArray) -> tuple[GPT, None]:
    """Create the benchmark model."""
    return GPT(config, key=key), None


def forward(model: GPT, state: None, inputs: Array) -> tuple[Array, None]:
    """Apply the model to a batch."""
    return jax.vmap(model)(inputs), state
