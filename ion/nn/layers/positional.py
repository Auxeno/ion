"""Positional encoding layers and functions.

Modules:
    LearnedPositionalEmbedding  Trainable lookup table added to input.
    RoPE                        Rotary position embedding for Q or K.  (Su et al., 2021)

Functions:
    sinusoidal  Fixed sin/cos encodings.  (Vaswani et al., 2017)
    alibi       Linear attention bias.    (Press et al., 2022)

Truncated normal weight init (std=0.02) for the learned embedding table.
"""

import jax
import jax.numpy as jnp
from jax.nn.initializers import Initializer
from jaxtyping import Array, Float, PRNGKeyArray

from ..module import Module
from ..param import Param


class LearnedPositionalEmbedding(Module):
    """Learnable positional embeddings added to input features.

    >>> pos = LearnedPositionalEmbedding(128, 64, key=key)
    >>> pos(x)  # (*, s, 64) -> (*, s, 64)
    """

    w: Param[Float[Array, "s d"]]

    def __init__(
        self,
        max_len: int,
        dim: int,
        w_init: Initializer = jax.nn.initializers.truncated_normal(0.02),
        *,
        key: PRNGKeyArray,
    ) -> None:

        self.w = Param(w_init(shape=(max_len, dim), key=key))

    def __call__(self, x: Float[Array, "... s d"]) -> Float[Array, "... s d"]:

        seq_len = x.shape[-2]
        max_len = self.w.shape[0]
        if seq_len > max_len:
            raise ValueError(f"input seq_len ({seq_len}) exceeds max_len ({max_len})")

        x = x + self.w[:seq_len]

        return x


class RoPE(Module):
    """Rotary positional embeddings applied to query or key vectors.

    >>> rope = RoPE()
    >>> rope(q)  # (*, s, d) -> (*, s, d)
    >>> rope(k)  # (*, s, d) -> (*, s, d)
    """

    theta: float

    def __init__(self, theta: float = 10_000.0) -> None:

        self.theta = theta

    def __call__(self, x: Float[Array, "... s d"]) -> Float[Array, "... s d"]:

        seq_len, head_dim = x.shape[-2], x.shape[-1]
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim ({head_dim}) must be even")

        # Inverse frequencies for feature pairs (d / 2,)
        freq_indices = jnp.arange(0, head_dim, 2, dtype=jnp.float32)
        inv_freqs = 1.0 / (self.theta ** (freq_indices / head_dim))

        # Phase angles from positions and frequencies, duplicated per feature pair (s, d)
        positions = jnp.arange(seq_len, dtype=jnp.float32)
        freqs = jnp.repeat(jnp.outer(positions, inv_freqs), 2, axis=-1)
        cos = jnp.cos(freqs).astype(x.dtype)
        sin = jnp.sin(freqs).astype(x.dtype)

        # Swap and negate adjacent pairs: [x0, x1, x2, x3] -> [-x1, x0, -x3, x2]
        x_pairs = x.reshape(x.shape[:-1] + (-1, 2))
        x_rotated = jnp.stack((-x_pairs[..., 1], x_pairs[..., 0]), axis=-1).reshape(x.shape)

        return (x * cos) + (x_rotated * sin)


def sinusoidal(
    seq_len: int,
    dim: int,
    theta: float = 10_000.0,
    dtype: jnp.dtype = jnp.float32,
) -> Float[Array, "s d"]:
    """Sinusoidal positional encodings.

    >>> sinusoidal(128, 64)  # (128, 64)
    """

    if dim % 2 != 0:
        raise ValueError(f"dim ({dim}) must be even")

    # Relative positions (s, 1) and frequency scales (d / 2,)
    positions = jnp.arange(seq_len, dtype=jnp.float32)[:, None]
    divisor = jnp.exp(jnp.arange(0, dim, 2, dtype=jnp.float32) * (-jnp.log(theta) / dim))

    # Phase angles (s, d / 2)
    angles = positions * divisor

    # Interleave sin and cos into alternating columns (s, d)
    return (
        jnp.stack([jnp.sin(angles), jnp.cos(angles)], axis=-1).reshape(seq_len, dim).astype(dtype)
    )


def alibi(
    seq_len: int,
    num_heads: int,
    dtype: jnp.dtype = jnp.float32,
) -> Float[Array, "h s s"]:
    """ALiBi linear position bias for attention logits.

    >>> alibi(128, 8)  # (8, 128, 128)
    """

    if num_heads & (num_heads - 1) != 0:
        raise ValueError(f"num_heads ({num_heads}) must be a power of 2")

    # Geometric per-head slopes from the paper: 2^(-8/h), 2^(-16/h), ..., 2^-8 (h,)
    slopes = 0.5 ** (8.0 * jnp.arange(1, num_heads + 1) / num_heads)

    # Relative distances between positions (s, s)
    positions = jnp.arange(seq_len, dtype=jnp.float32)
    rel_pos = positions[None, :] - positions[:, None]

    # Broadcast slopes over sequence dims (h, s, s)
    bias = slopes[:, None, None] * rel_pos[None, :, :]

    return bias.astype(dtype)
