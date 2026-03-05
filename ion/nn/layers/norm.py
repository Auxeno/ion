"""Normalization layers.

Modules:
    LayerNorm     Layer normalization.                    (Ba et al., 2016)
    RMSNorm       RMS normalization, no mean centering.   (Zhang & Sennrich, 2019)
    GroupNorm     Normalization over channel groups.      (Wu & He, 2018)
    BatchNorm     Normalization with running statistics.  (Ioffe & Szegedy, 2015)
    InstanceNorm  Per-instance spatial normalization.     (Ulyanov et al., 2016)

Scale initialized to 1, bias to 0. Normalizes over the last dimension by default.
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

        x = x * self.scale + self.b

        return x


class GroupNorm(Module):
    """Group normalization, splitting channels into groups.

    >>> norm = GroupNorm(num_groups=8, dim=64)
    >>> norm(x)  # (*, 64) -> (*, 64)
    """

    scale: Param[Float[Array, " d"]]
    b: Param[Float[Array, " d"]]
    num_groups: int
    eps: float

    def __init__(
        self,
        num_groups: int,
        dim: int,
        eps: float = 1e-5,
        dtype: jnp.dtype = jnp.float32,
    ) -> None:

        if dim % num_groups != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_groups ({num_groups})")

        self.scale = Param(jnp.ones(dim, dtype=dtype))
        self.b = Param(jnp.zeros(dim, dtype=dtype))

        self.num_groups = num_groups
        self.eps = eps

    def __call__(self, x: Float[Array, "... d"]) -> Float[Array, "... d"]:

        group_shape = (*x.shape[:-1], self.num_groups, x.shape[-1] // self.num_groups)
        x = x.reshape(group_shape)

        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)

        x = (x - mean) * lax.rsqrt(var + self.eps)

        x = x.reshape(*group_shape[:-2], -1)

        x = x * self.scale + self.b

        return x


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

        x = x * self.scale

        return x


class BatchNorm(Module):
    """Batch normalization with running statistics.

    >>> bn = BatchNorm(64)
    >>> y, state = bn(x, state=bn.initial_state, training=True)  # (*, 64) -> (*, 64)
    """

    scale: Param[Float[Array, " d"]]
    b: Param[Float[Array, " d"]] | None
    state: tuple[Float[Array, " d"], Float[Array, " d"]] | None
    momentum: float
    eps: float

    def __init__(
        self,
        dim: int,
        momentum: float = 0.1,
        eps: float = 1e-5,
        bias: bool = True,
        dtype: jnp.dtype = jnp.float32,
    ) -> None:

        self.scale = Param(jnp.ones(dim, dtype=dtype))
        self.b = Param(jnp.zeros(dim, dtype=dtype)) if bias else None

        self.state = None
        self.momentum = momentum
        self.eps = eps

    def __call__(
        self,
        x: Float[Array, "... d"],
        state: tuple[Float[Array, " d"], Float[Array, " d"]] | None = None,
        training: bool = False,
    ) -> tuple[Float[Array, "... d"], tuple[Float[Array, " d"], Float[Array, " d"]]]:

        if state is None:
            state = self.state
        if state is None:
            raise ValueError(
                "No state provided and self.state is None. "
                "Pass state explicitly or call replace(state=initial_state) first."
            )

        running_mean, running_var = state
        reduce_axes = tuple(range(x.ndim - 1))

        if training:
            mean = jnp.mean(x, axis=reduce_axes)
            var = jnp.mean(jnp.square(x - mean), axis=reduce_axes)

            new_running_mean = lax.stop_gradient(
                (1 - self.momentum) * running_mean + self.momentum * mean
            )
            new_running_var = lax.stop_gradient(
                (1 - self.momentum) * running_var + self.momentum * var
            )

            new_state = (new_running_mean, new_running_var)
        else:
            mean = running_mean
            var = running_var
            new_state = state

        y = (x - mean) * lax.rsqrt(var + self.eps)
        y = y * self.scale

        if self.b is not None:
            y = y + self.b

        return y, new_state

    @property
    def initial_state(self) -> tuple[Float[Array, " d"], Float[Array, " d"]]:
        d = self.scale.shape[0]
        return (jnp.zeros(d, dtype=self.scale.dtype), jnp.ones(d, dtype=self.scale.dtype))


class InstanceNorm(Module):
    """Instance normalization over spatial dimensions.

    >>> norm = InstanceNorm(16, num_spatial_dims=2)
    >>> norm(x)  # (*, h, w, 16) -> (*, h, w, 16)
    """

    scale: Param[Float[Array, " d"]]
    b: Param[Float[Array, " d"]]
    num_spatial_dims: int
    eps: float

    def __init__(
        self,
        dim: int,
        num_spatial_dims: int = 1,
        eps: float = 1e-5,
        dtype: jnp.dtype = jnp.float32,
    ) -> None:

        self.scale = Param(jnp.ones(dim, dtype=dtype))
        self.b = Param(jnp.zeros(dim, dtype=dtype))

        self.num_spatial_dims = num_spatial_dims
        self.eps = eps

    def __call__(self, x: Float[Array, "... d"]) -> Float[Array, "... d"]:

        reduce_axes = tuple(range(-self.num_spatial_dims - 1, -1))

        mean = jnp.mean(x, axis=reduce_axes, keepdims=True)
        var = jnp.mean(jnp.square(x - mean), axis=reduce_axes, keepdims=True)

        x = (x - mean) * lax.rsqrt(var + self.eps)

        x = x * self.scale + self.b

        return x
