"""Positional encoding layers.

Modules:
    LearnedPositionalEmbedding     Trainable lookup table added to input.
    RoPE                           Rotary position embedding for Q or K.  (Su et al., 2021)
    SinusoidalPositionalEmbedding  Fixed sin/cos encodings added to input.  (Vaswani et al., 2017)

Fan-in variance scaling weight init (std 1/sqrt(dim)) for the learned embedding table.
RoPE positions are a 1D sequence unless given a shape, which lays them out on an N-dimensional
lattice and splits the head dimension evenly across its axes.
"""

import math

import jax.numpy as jnp
from jax.nn.initializers import Initializer, variance_scaling

from ...typing import Array, Float, PRNGKey
from ..module import Module
from ..param import Param


class LearnedPositionalEmbedding(Module):
    """Learnable positional embeddings added to input features.

    >>> pos = LearnedPositionalEmbedding(128, 64, key=key)
    >>> pos(x)  # (*, s, 64) -> (*, s, 64)
    """

    w: Param[Float[Array, "m d"]]

    def __init__(
        self,
        max_len: int,
        dim: int,
        *,
        w_init: Initializer = variance_scaling(1.0, "fan_in", "uniform", out_axis=0),
        key: PRNGKey,
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
    """Rotary positional embeddings applied to query and key vectors.

    >>> rope = RoPE()
    >>> rope(q)  # (*, s, h, d) -> (*, s, h, d)
    >>> rope = RoPE(shape=(14, 14), num_prefix_tokens=1)
    >>> rope(q)  # 14x14 patch grid behind a CLS token: (*, 197, h, d) -> (*, 197, h, d)
    """

    shape: tuple[int, ...] | None
    num_prefix_tokens: int
    axis: int
    theta: float

    def __init__(
        self,
        *,
        shape: tuple[int, ...] | None = None,
        num_prefix_tokens: int = 0,
        axis: int = -3,
        theta: float = 10_000.0,
    ) -> None:

        if shape is not None and len(shape) < 1:
            raise ValueError("shape must have at least one element")

        self.shape = shape
        self.num_prefix_tokens = num_prefix_tokens
        self.axis = axis
        self.theta = theta

    def __call__(self, x: Float[Array, "... d"]) -> Float[Array, "... d"]:

        # Rotate over the sequence axis, brought to -2 and restored on return
        x = jnp.moveaxis(x, self.axis, -2)
        seq_len, head_dim = x.shape[-2], x.shape[-1]

        # Without a lattice the positions are the sequence itself
        shape = (seq_len - self.num_prefix_tokens,) if self.shape is None else self.shape
        num_axes = len(shape)

        if head_dim % (2 * num_axes) != 0:
            raise ValueError(f"head_dim ({head_dim}) must be divisible by {2 * num_axes}")
        if math.prod(shape) + self.num_prefix_tokens != seq_len:
            raise ValueError(f"lattice {shape} does not fill sequence length ({seq_len})")

        # Lattice coordinates per axis, prefix tokens padded in at position 0 (n, s)
        positions = jnp.indices(shape, dtype=jnp.float32).reshape(num_axes, -1)
        positions = jnp.pad(positions, ((0, 0), (self.num_prefix_tokens, 0)))

        # Each axis spans the full theta spectrum within the head section it owns (d / 2n,)
        axis_dim = head_dim // num_axes
        inv_freqs = 1.0 / (self.theta ** (jnp.arange(0, axis_dim, 2, jnp.float32) / axis_dim))

        # Phase angles, axis-major so each axis owns one contiguous section (s, d / 2)
        freqs = (positions[..., None] * inv_freqs).swapaxes(0, 1).reshape(seq_len, -1)

        # Both features of a pair turn by the same angle (s, d)
        freqs = jnp.repeat(freqs, 2, axis=-1)
        cos = jnp.cos(freqs).astype(x.dtype)
        sin = jnp.sin(freqs).astype(x.dtype)

        # Swap and negate adjacent pairs: [x0, x1, x2, x3] -> [-x1, x0, -x3, x2]
        x_rotated = jnp.stack((-x[..., 1::2], x[..., ::2]), axis=-1).reshape(x.shape)

        x = (x * cos) + (x_rotated * sin)

        return jnp.moveaxis(x, -2, self.axis)


class SinusoidalPositionalEmbedding(Module):
    """Fixed sinusoidal positional encodings added to input features.

    >>> pos = SinusoidalPositionalEmbedding()
    >>> pos(x)  # (*, s, d) -> (*, s, d)
    """

    theta: float

    def __init__(self, *, theta: float = 10_000.0) -> None:

        self.theta = theta

    def __call__(self, x: Float[Array, "... s d"]) -> Float[Array, "... s d"]:

        seq_len, dim = x.shape[-2], x.shape[-1]
        if dim % 2 != 0:
            raise ValueError(f"dim ({dim}) must be even")

        # Relative positions (s, 1) and frequency scales (d / 2,)
        positions = jnp.arange(seq_len, dtype=jnp.float32)[:, None]
        divisor = jnp.exp(jnp.arange(0, dim, 2, jnp.float32) * (-jnp.log(self.theta) / dim))

        # Phase angles (s, d / 2)
        angles = positions * divisor

        # Interleave sin and cos into alternating columns (s, d)
        encoding = jnp.stack([jnp.sin(angles), jnp.cos(angles)], axis=-1).reshape(seq_len, dim)

        x = x + encoding.astype(x.dtype)

        return x
