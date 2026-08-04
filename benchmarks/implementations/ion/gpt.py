"""Ion GPT benchmark."""

import jax
from jaxtyping import Array, Float, Int, PRNGKeyArray

import ion.nn as nn

from ...configs import ModelConfig


class TransformerBlock(nn.Module):
    """Pre-norm causal transformer block."""

    attention: nn.Attention
    attention_norm: nn.LayerNorm
    mlp_norm: nn.LayerNorm
    mlp_in: nn.Linear
    mlp_out: nn.Linear

    def __init__(self, dim: int, num_heads: int, *, key: PRNGKeyArray) -> None:
        key_attention, key_in, key_out = jax.random.split(key, 3)
        self.attention = nn.Attention(
            dim,
            num_heads,
            causal=True,
            key=key_attention,
        )
        self.attention_norm = nn.LayerNorm(dim)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp_in = nn.Linear(dim, 4 * dim, bias=False, key=key_in)
        self.mlp_out = nn.Linear(4 * dim, dim, bias=False, key=key_out)

    def __call__(self, x: Float[Array, "b s d"]) -> Float[Array, "b s d"]:
        x = x + self.attention(self.attention_norm(x))
        return x + self.mlp_out(jax.nn.gelu(self.mlp_in(self.mlp_norm(x))))


class GPT(nn.Module):
    """Decoder-only transformer with tied token embeddings."""

    embedding: nn.Embedding
    position: Array
    blocks: tuple[TransformerBlock, ...]
    norm: nn.LayerNorm

    def __init__(self, config: ModelConfig, *, key: PRNGKeyArray) -> None:
        keys = jax.random.split(key, config.depth + 1)
        self.embedding = nn.Embedding(config.vocab_size, config.width, key=keys[0])
        self.position = nn.sinusoidal(config.seq_len, config.width)
        self.blocks = tuple(
            TransformerBlock(config.width, config.num_heads, key=keys[index + 1])
            for index in range(config.depth)
        )
        self.norm = nn.LayerNorm(config.width)

    def __call__(self, tokens: Int[Array, "b s"]) -> Float[Array, "b s vocab"]:
        x = self.embedding(tokens) + self.position[: tokens.shape[1]]
        for block in self.blocks:
            x = block(x)
        return self.norm(x) @ self.embedding.w.T


def create_model(config: ModelConfig, *, key: PRNGKeyArray) -> GPT:
    """Create the benchmark model."""
    return GPT(config, key=key)
