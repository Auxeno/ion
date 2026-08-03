"""Normalization layers.

Modules:
    BatchNorm     Batch normalization.                   (Ioffe & Szegedy, 2015)
    LayerNorm     Layer normalization.                   (Ba et al., 2016)
    RMSNorm       RMS normalization, no mean centering.  (Zhang & Sennrich, 2019)
    GroupNorm     Normalization over channel groups.     (Wu & He, 2018)
    SpectralNorm  Spectral normalization wrapper.        (Miyato et al., 2018)

Affine scales initialize to ones and biases to zeros. Feature dimensions are always final.
"""

from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float, PRNGKeyArray

from ..buffer import Buffer
from ..module import Module
from ..param import Param


class BatchNorm(Module):
    """Batch normalization with running mean and variance.

    >>> norm = BatchNorm(64)
    >>> norm(x, training=True)  # (*, 64) -> (*, 64)
    """

    scale: Param[Float[Array, " d"]]
    b: Param[Float[Array, " d"]] | None
    running_mean: Buffer
    running_var: Buffer
    momentum: float
    eps: float

    def __init__(
        self,
        dim: int,
        momentum: float = 0.1,
        eps: float = 1e-5,
        bias: bool = True,
    ) -> None:

        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f"momentum ({momentum}) must be in [0, 1]")
        if eps <= 0.0:
            raise ValueError(f"eps ({eps}) must be greater than 0")

        self.scale = Param(jnp.ones(dim))
        self.b = Param(jnp.zeros(dim)) if bias else None

        self.running_mean = Buffer(jnp.zeros(dim))
        self.running_var = Buffer(jnp.ones(dim))

        self.momentum = momentum
        self.eps = eps

    def __call__(
        self,
        x: Float[Array, "... d"],
        *,
        training: bool,
    ) -> Float[Array, "... d"]:

        if x.ndim < 2:
            raise ValueError("BatchNorm input must have at least one reduction dimension")

        # Use batch statistics and update the running values during training
        if training:
            reduce_axes = tuple(range(x.ndim - 1))
            mean = jnp.mean(x, axis=reduce_axes)
            var = jnp.mean(jnp.square(x - mean), axis=reduce_axes)

            self.running_mean.set(
                (1.0 - self.momentum) * self.running_mean.value + self.momentum * mean
            )
            self.running_var.set(
                (1.0 - self.momentum) * self.running_var.value + self.momentum * var
            )
        else:
            mean = self.running_mean.value
            var = self.running_var.value

        # Normalize, then apply the learned affine transform
        x = ((x - mean) * lax.rsqrt(var + self.eps)).astype(x.dtype)

        x = x * self.scale

        if self.b is not None:
            x = x + self.b

        return x


class LayerNorm(Module):
    """Layer normalization over the last dimension.

    >>> norm = LayerNorm(64)
    >>> norm(x)  # (*, 64) -> (*, 64)
    """

    scale: Param[Float[Array, " d"]]
    b: Param[Float[Array, " d"]] | None
    eps: float

    def __init__(self, dim: int, eps: float = 1e-5, bias: bool = True) -> None:

        self.scale = Param(jnp.ones(dim))
        self.b = Param(jnp.zeros(dim)) if bias else None

        self.eps = eps

    def __call__(self, x: Float[Array, "... d"]) -> Float[Array, "... d"]:

        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)

        x = (x - mean) * lax.rsqrt(var + self.eps)

        x = x * self.scale

        if self.b is not None:
            x = x + self.b

        return x


class GroupNorm(Module):
    """Group normalization, splitting channels into groups.

    >>> norm = GroupNorm(64, num_groups=8, num_spatial_dims=2)
    >>> norm(x)  # (*, h, w, 64) -> (*, h, w, 64)

    >>> norm = GroupNorm(64, num_groups=8, num_spatial_dims=0)
    >>> norm(x)  # (*, 64) -> (*, 64), per-position normalization
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
        num_spatial_dims: int,
        eps: float = 1e-5,
    ) -> None:

        if dim % num_groups != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_groups ({num_groups})")

        self.scale = Param(jnp.ones(dim))
        self.b = Param(jnp.zeros(dim))

        self.num_groups = num_groups
        self.num_spatial_dims = num_spatial_dims
        self.eps = eps

    def __call__(self, x: Float[Array, "... d"]) -> Float[Array, "... d"]:

        num_spatial = self.num_spatial_dims

        # Split channels into groups
        group_shape = (*x.shape[:-1], self.num_groups, x.shape[-1] // self.num_groups)
        x = x.reshape(group_shape)

        # Reduce over spatial dims and within-group channels, indexed from the back
        reduce_axes = tuple(range(-num_spatial - 2, -2)) + (-1,)

        mean = jnp.mean(x, axis=reduce_axes, keepdims=True)
        var = jnp.mean(jnp.square(x - mean), axis=reduce_axes, keepdims=True)

        x = (x - mean) * lax.rsqrt(var + self.eps)

        # Merge groups back
        x = x.reshape(*x.shape[:-2], -1)

        return x * self.scale + self.b


class RMSNorm(Module):
    """Root mean square normalization (no mean centering).

    >>> norm = RMSNorm(64)
    >>> norm(x)  # (*, 64) -> (*, 64)
    """

    scale: Param[Float[Array, " d"]]
    eps: float

    def __init__(self, dim: int, eps: float = 1e-5) -> None:

        self.scale = Param(jnp.ones(dim))
        self.eps = eps

    def __call__(self, x: Float[Array, "... d"]) -> Float[Array, "... d"]:

        rms = jnp.mean(jnp.square(x), axis=-1, keepdims=True)

        x = x * lax.rsqrt(rms + self.eps)

        return x * self.scale


class SpectralNorm(Module):
    """Normalize a module parameter by its spectral norm.

    >>> layer = SpectralNorm(Linear(64, 64, key=linear_key), key=key)
    >>> layer(x, training=True)  # (*, 64) -> (*, 64)
    """

    module: Module
    parameter: str
    u: Buffer
    v: Buffer
    power_iterations: int
    eps: float

    def __init__(
        self,
        module: Module,
        parameter: str = "w",
        power_iterations: int = 1,
        eps: float = 1e-12,
        *,
        key: PRNGKeyArray,
    ) -> None:

        param = getattr(module, parameter)
        if not isinstance(param, Param):
            raise TypeError(f"{type(module).__name__}.{parameter} must be a Param")
        if param.ndim < 2:
            raise ValueError(f"{type(module).__name__}.{parameter} must be at least 2D")
        if power_iterations < 1:
            raise ValueError(f"power_iterations ({power_iterations}) must be at least 1")
        if eps <= 0.0:
            raise ValueError(f"eps ({eps}) must be greater than 0")

        weight = jnp.asarray(param)
        matrix = weight.reshape(-1, weight.shape[-1]).T

        u = jax.random.normal(key, (matrix.shape[0],), dtype=matrix.dtype)
        u, v = self._power_iteration(matrix, u, power_iterations, eps)

        self.u = Buffer(u)
        self.v = Buffer(v)

        self.module = module
        self.parameter = parameter
        self.power_iterations = power_iterations
        self.eps = eps

    @staticmethod
    def _power_iteration(
        matrix: Array,
        u: Array,
        power_iterations: int,
        eps: float,
    ) -> tuple[Array, Array]:

        for _ in range(power_iterations):
            v = matrix.T @ u
            v = v / jnp.maximum(jnp.linalg.norm(v), eps)
            u = matrix @ v
            u = u / jnp.maximum(jnp.linalg.norm(u), eps)

        return lax.stop_gradient((u, v))

    def __call__(
        self,
        x: Any,
        *,
        training: bool,
    ) -> Any:

        weight = jnp.asarray(getattr(self.module, self.parameter))
        matrix = weight.reshape(-1, weight.shape[-1]).T
        u, v = self.u.value, self.v.value

        # Refine the singular vectors only during training
        if training:
            u, v = self._power_iteration(matrix, u, self.power_iterations, self.eps)
            self.u.set(u)
            self.v.set(v)

        # Normalize the parameter without modifying the wrapped module
        sigma = u @ matrix @ v
        sigma = jnp.maximum(sigma, jnp.asarray(self.eps, dtype=sigma.dtype))
        weight = (weight / sigma).astype(weight.dtype)

        normalized_module = getattr(self.module.at, self.parameter).set(weight)

        return normalized_module(x)
