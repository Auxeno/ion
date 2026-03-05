"""Positional encoding layers and functions.

Modules:
    LearnedPositionalEmbedding  Trainable lookup table added to input.
    sinusoidal                  Fixed sin/cos encodings.               (Vaswani et al., 2017)
    rope                        Rotary position embedding frequencies. (Su et al., 2021)
    apply_rope                  Apply rotary embeddings to Q or K.
    alibi                       Linear attention bias.                 (Press et al., 2022)
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
        dtype: jnp.dtype = jnp.float32,
        w_init: Initializer = jax.nn.initializers.truncated_normal(0.02),
        *,
        key: PRNGKeyArray,
    ) -> None:

        self.w = Param(w_init(shape=(max_len, dim), dtype=dtype, key=key))

    def __call__(self, x: Float[Array, "... s d"]) -> Float[Array, "... s d"]:

        # Slice to input sequence length and broadcast-add
        x = x + self.w[: x.shape[-2]]

        return x


def sinusoidal(
    seq_len: int,
    dim: int,
    dtype: jnp.dtype = jnp.float32,
) -> Float[Array, "s d"]:
    """Sinusoidal positional encodings.

    >>> sinusoidal(128, 64)  # (128, 64)
    """

    # Relative positions (s, 1) and frequency scales (d / 2,)
    positions = jnp.arange(seq_len, dtype=jnp.float32)[:, None]
    divisor = jnp.exp(jnp.arange(0, dim, 2, dtype=jnp.float32) * (-jnp.log(10_000.0) / dim))

    # Phase angles (s, d / 2)
    angles = positions * divisor

    # Interleave sin and cos into alternating columns (s, d)
    return jnp.stack([jnp.sin(angles), jnp.cos(angles)], axis=-1).reshape(seq_len, dim).astype(dtype)


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

    # Geometric per-head slopes: 0.5, 0.25, 0.125, ... (h,)
    slopes = 0.5 ** jnp.arange(1, num_heads + 1)

    # Relative distances between positions (s, s)
    positions = jnp.arange(seq_len, dtype=jnp.float32)
    rel_pos = positions[None, :] - positions[:, None]

    # Broadcast slopes over sequence dims (h, s, s)
    bias = slopes[:, None, None] * rel_pos[None, :, :]

    return bias.astype(dtype)


def rope(
    seq_len: int,
    head_dim: int,
    theta: float = 10_000.0,
    dtype: jnp.dtype = jnp.float32,
) -> tuple[Float[Array, "s d"], Float[Array, "s d"]]:
    """Cosine and sine frequency tables for Rotary Positional Embeddings.

    >>> cos, sin = rope(128, 64)  # (128, 64), (128, 64)
    """

    if head_dim % 2 != 0:
        raise ValueError(f"head_dim ({head_dim}) must be even")

    # Inverse frequencies for feature pairs (d / 2,)
    freq_indices = jnp.arange(0, head_dim, 2, dtype=jnp.float32)
    inv_freqs = 1.0 / (theta ** (freq_indices / head_dim))

    # Outer product of positions and frequencies (s, d / 2)
    t = jnp.arange(seq_len, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freqs)

    # Duplicate for each feature pair (s, d)
    freqs = jnp.repeat(freqs, 2, axis=-1)

    return jnp.cos(freqs).astype(dtype), jnp.sin(freqs).astype(dtype)


def apply_rope(
    x: Float[Array, "... s d"],
    cos: Float[Array, "s d"],
    sin: Float[Array, "s d"],
) -> Float[Array, "... s d"]:
    """Apply rotary positional embeddings to query or key vectors.

    >>> cos, sin = rope(128, 64)
    >>> apply_rope(q, cos, sin)  # (*, 128, 64) -> (*, 128, 64)
    """

    # Swap and negate adjacent pairs: [x0, x1, x2, x3] -> [-x1, x0, -x3, x2]
    x_pairs = x.reshape(x.shape[:-1] + (-1, 2))
    x_rotated = jnp.stack((-x_pairs[..., 1], x_pairs[..., 0]), axis=-1).reshape(x.shape)

    return (x * cos) + (x_rotated * sin)
