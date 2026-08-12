"""Stochastic regularization with element-wise and broadcast dropout.

Modules:
    Dropout  Stochastic dropout with inverse scaling and shared masks.

Uses inverse dropout: outputs are scaled by 1/(1-p) during training.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from ..module import Module


class Dropout(Module):
    """Stochastic dropout layer.

    >>> drop = Dropout(0.5)
    >>> drop(x, training=True, key=key)  # (*, d) -> (*, d)
    >>> drop(x, training=False)  # evaluation identity
    """

    p: float
    broadcast_dims: tuple[int, ...]

    def __init__(self, p: float, *, broadcast_dims: tuple[int, ...] = ()) -> None:

        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p ({p}) must be in [0, 1]")
        if len(set(broadcast_dims)) != len(broadcast_dims):
            raise ValueError(f"broadcast_dims {tuple(broadcast_dims)} must not contain duplicates")

        self.p = p
        self.broadcast_dims = tuple(broadcast_dims)

    def __call__(
        self,
        x: Float[Array, "..."],
        *,
        training: bool,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "..."]:

        if not training or self.p == 0.0:
            return x

        if key is None:
            raise ValueError("key is required when training=True")

        if self.p == 1.0:
            return jnp.zeros_like(x)

        if any(dim < -x.ndim or dim >= x.ndim for dim in self.broadcast_dims):
            raise ValueError(f"broadcast_dims invalid for input rank ({x.ndim})")

        broadcast_dims = [dim % x.ndim for dim in self.broadcast_dims]

        keep_prob = 1.0 - self.p
        mask_shape = tuple(1 if dim in broadcast_dims else size for dim, size in enumerate(x.shape))
        mask = jax.random.bernoulli(key, p=keep_prob, shape=mask_shape)

        x = (x * mask.astype(x.dtype)) / keep_prob

        return x
