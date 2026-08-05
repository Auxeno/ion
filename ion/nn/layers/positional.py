"""Positional encoding layers and functions.

Modules:
    LearnedPositionalEmbedding  Trainable lookup table added to input.
    RoPE                        Rotary position embedding for Q or K.  (Su et al., 2021)

Functions:
    sinusoidal  Fixed sin/cos encodings.  (Vaswani et al., 2017)
    alibi       Linear attention bias.    (Press et al., 2022)

Fan-in variance scaling weight init (std 1/sqrt(dim)) for the learned embedding table.
RoPE positions are a 1D sequence unless given a shape, which lays them out on an N-dimensional
lattice and splits the head dimension evenly across its axes.
"""

import math

import jax.numpy as jnp
from jax.nn.initializers import Initializer, variance_scaling
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
        *,
        w_init: Initializer = variance_scaling(1.0, "fan_in", "uniform", out_axis=0),
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
    >>> rope = RoPE(shape=(14, 14), num_prefix_tokens=1)
    >>> rope(q)  # 14x14 patch grid behind a CLS token: (*, 197, d) -> (*, 197, d)
    """

    shape: tuple[int, ...] | None
    num_prefix_tokens: int
    theta: float

    def __init__(
        self,
        *,
        shape: tuple[int, ...] | None = None,
        num_prefix_tokens: int = 0,
        theta: float = 10_000.0,
    ) -> None:

        if shape is not None and len(shape) < 1:
            raise ValueError("shape must have at least one element")

        self.shape = shape
        self.num_prefix_tokens = num_prefix_tokens
        self.theta = theta

    def __call__(self, x: Float[Array, "... d"], axis: int = -2) -> Float[Array, "... d"]:

        x = jnp.moveaxis(x, axis, -2)
        seq_len, head_dim = x.shape[-2], x.shape[-1]
        shape = (seq_len - self.num_prefix_tokens,) if self.shape is None else self.shape
        num_axes = len(shape)
        if head_dim % (2 * num_axes) != 0:
            raise ValueError(f"head_dim ({head_dim}) must be divisible by {2 * num_axes}")
        if math.prod(shape) + self.num_prefix_tokens != seq_len:
            raise ValueError(f"lattice {shape} does not fill sequence length ({seq_len})")

        # Lattice coordinates per axis, behind prefix tokens pinned to position 0 (n, s)
        positions = jnp.indices(shape, dtype=jnp.float32).reshape(num_axes, -1)
        positions = jnp.pad(positions, ((0, 0), (self.num_prefix_tokens, 0)))

        # Inverse frequencies for the feature pairs each axis owns (d / 2n,)
        axis_dim = head_dim // num_axes
        inv_freqs = 1.0 / (self.theta ** (jnp.arange(0, axis_dim, 2, jnp.float32) / axis_dim))

        # Phase angles per axis, contiguous sections of the head duplicated per pair (s, d)
        freqs = (positions[..., None] * inv_freqs).swapaxes(0, 1).reshape(seq_len, -1)
        freqs = jnp.repeat(freqs, 2, axis=-1)
        cos = jnp.cos(freqs).astype(x.dtype)
        sin = jnp.sin(freqs).astype(x.dtype)

        # Swap and negate adjacent pairs: [x0, x1, x2, x3] -> [-x1, x0, -x3, x2]
        x_rotated = jnp.stack((-x[..., 1::2], x[..., ::2]), axis=-1).reshape(x.shape)

        return jnp.moveaxis((x * cos) + (x_rotated * sin), -2, axis)


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
