"""Normalization layers.

Modules:
    LayerNorm     Layer normalization.                    (Ba et al., 2016)
    RMSNorm       RMS normalization, no mean centering.   (Zhang & Sennrich, 2019)
    GroupNorm     Normalization over channel groups.      (Wu & He, 2018)

Scales initialized to ones, bias to zeros. Normalizes over the last dimension by default.
BatchNorm is intentionally omitted. See docs/layers.md for rationale.
"""

import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float

from ..module import Module
from ..param import Param


class LayerNorm(Module):
    """Layer normalization over the last dimension.

    >>> norm = LayerNorm(64)
    >>> norm(x)  # (*, 64) -> (*, 64)
    """

    scale: Param[Float[Array, " d"]]
    b: Param[Float[Array, " d"]]
    eps: float

    def __init__(self, dim: int, eps: float = 1e-5, dtype: jnp.dtype = jnp.float32) -> None:

        self.scale = Param(jnp.ones(dim, dtype=dtype))
        self.b = Param(jnp.zeros(dim, dtype=dtype))

        self.eps = eps

    def __call__(self, x: Float[Array, "... d"]) -> Float[Array, "... d"]:

        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)

        x = (x - mean) * lax.rsqrt(var + self.eps)

        return x * self.scale + self.b


class GroupNorm(Module):
    """Group normalization, splitting channels into groups.

    >>> norm = GroupNorm(64, num_groups=8)
    >>> norm(x)  # (*, 64) -> (*, 64)

    >>> norm = GroupNorm(64, num_groups=8, num_spatial_dims=2)
    >>> norm(x)  # (*, h, w, 64) -> (*, h, w, 64)
    """

    scale: Param[Float[Array, " d"]]
    b: Param[Float[Array, " d"]]
    num_groups: int
    num_spatial_dims: int
    eps: float

    def __init__(
        self,
        dim: int,
        num_groups: int,
        num_spatial_dims: int = 0,
        eps: float = 1e-5,
        dtype: jnp.dtype = jnp.float32,
    ) -> None:

        if dim % num_groups != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_groups ({num_groups})")

        self.scale = Param(jnp.ones(dim, dtype=dtype))
        self.b = Param(jnp.zeros(dim, dtype=dtype))

        self.num_groups = num_groups
        self.num_spatial_dims = num_spatial_dims
        self.eps = eps

    def __call__(self, x: Float[Array, "b ... d"]) -> Float[Array, "b ... d"]:

        num_spatial = self.num_spatial_dims

        # Split channels into groups
        group_shape = (*x.shape[:-1], self.num_groups, x.shape[-1] // self.num_groups)
        x = x.reshape(group_shape)

        reduce_axes = tuple(range(1, num_spatial + 1)) + (-1,)

        mean = jnp.mean(x, axis=reduce_axes, keepdims=True)
        var = jnp.mean(jnp.square(x - mean), axis=reduce_axes, keepdims=True)

        x = (x - mean) * lax.rsqrt(var + self.eps)

        # Merge groups back
        x = x.reshape(*x.shape[: num_spatial + 1], -1)

        return x * self.scale + self.b


class RMSNorm(Module):
    """Root mean square normalization (no mean centering).

    >>> norm = RMSNorm(64)
    >>> norm(x)  # (*, 64) -> (*, 64)
    """

    scale: Param[Float[Array, " d"]]
    eps: float

    def __init__(self, dim: int, eps: float = 1e-5, dtype: jnp.dtype = jnp.float32) -> None:

        self.scale = Param(jnp.ones(dim, dtype=dtype))
        self.eps = eps

    def __call__(self, x: Float[Array, "... d"]) -> Float[Array, "... d"]:

        rms = jnp.mean(jnp.square(x), axis=-1, keepdims=True)

        x = x * lax.rsqrt(rms + self.eps)

        return x * self.scale
