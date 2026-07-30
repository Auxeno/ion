"""PyTorch GPT benchmark."""

import math

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from ...configs import ModelConfig


def _sinusoidal(length: int, dim: int) -> Tensor:
    position = torch.arange(length)[:, None]
    frequency = torch.exp(torch.arange(0, dim, 2) * (-math.log(10_000.0) / dim))
    encoding = torch.zeros(length, dim)
    encoding[:, 0::2] = torch.sin(position * frequency)
    encoding[:, 1::2] = torch.cos(position * frequency[: dim // 2])
    return encoding


class TransformerBlock(nn.Module):
    """Pre-norm causal transformer block."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, bias=False, batch_first=True)
        self.attention_norm = nn.LayerNorm(dim)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp_in = nn.Linear(dim, 4 * dim, bias=False)
        self.mlp_out = nn.Linear(4 * dim, dim, bias=False)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        normalized = self.attention_norm(x)
        attention, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=mask,
            need_weights=False,
            is_causal=True,
        )
        x = x + attention
        return x + self.mlp_out(functional.gelu(self.mlp_in(self.mlp_norm(x))))


class GPT(nn.Module):
    """Decoder-only transformer with tied token embeddings."""

    mask: Tensor
    position: Tensor

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.width)
        self.register_buffer(
            "position", _sinusoidal(config.seq_len, config.width), persistent=False
        )
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(config.seq_len, config.seq_len, dtype=torch.bool), diagonal=1),
            persistent=False,
        )
        self.blocks = nn.ModuleList(
            TransformerBlock(config.width, config.num_heads) for _ in range(config.depth)
        )
        self.norm = nn.LayerNorm(config.width)

    def forward(self, tokens: Tensor) -> Tensor:
        x = self.embedding(tokens) + self.position[: tokens.shape[1]]
        mask = self.mask[: tokens.shape[1], : tokens.shape[1]]
        for block in self.blocks:
            x = block(x, mask)
        return self.norm(x) @ self.embedding.weight.T


def create_model(config: ModelConfig) -> GPT:
    """Create the benchmark model."""
    return GPT(config)
