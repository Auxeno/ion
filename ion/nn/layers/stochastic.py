"""Stochastic regularization with element-wise dropout and stochastic depth.

Modules:
    Dropout   Stochastic dropout with inverse scaling.
    DropPath  Stochastic depth, dropping a whole residual branch per sample.

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

    def __init__(self, p: float) -> None:

        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p ({p}) must be in [0, 1]")

        self.p = p

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

        keep_prob = 1.0 - self.p
        mask = jax.random.bernoulli(key, p=keep_prob, shape=x.shape)

        x = (x * mask.astype(x.dtype)) / keep_prob

        return x


class DropPath(Module):
    """Stochastic depth layer.

    >>> drop_path = DropPath(0.1)
    >>> x + drop_path(branch, training=True, key=key)  # (b, *) -> (b, *)
    >>> drop_path(branch, training=False)  # evaluation identity
    """

    p: float

    def __init__(self, p: float) -> None:

        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p ({p}) must be in [0, 1]")

        self.p = p

    def __call__(
        self,
        x: Float[Array, "b ..."],
        *,
        training: bool,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "b ..."]:

        if not training or self.p == 0.0:
            return x

        if key is None:
            raise ValueError("key is required when training=True")

        if self.p == 1.0:
            return jnp.zeros_like(x)

        keep_prob = 1.0 - self.p
        mask_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = jax.random.bernoulli(key, p=keep_prob, shape=mask_shape)

        x = (x * mask.astype(x.dtype)) / keep_prob

        return x
