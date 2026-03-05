"""Dropout regularization.

Modules:
    Dropout  Stochastic dropout with inverse scaling.

Uses inverse dropout: outputs are scaled by 1/(1-p) during training.
"""

import jax
from jaxtyping import Array, Float, PRNGKeyArray

from ..module import Module


class Dropout(Module):
    """Stochastic dropout.

    >>> drop = Dropout(0.5)
    >>> drop(x, key=key)  # (*, d) -> (*, d)
    """

    p: float
    deterministic: bool

    def __init__(self, p: float, deterministic: bool = False) -> None:
        self.p = p
        self.deterministic = deterministic

    def __call__(
        self,
        x: Float[Array, "..."],
        deterministic: bool | None = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "..."]:

        is_deterministic = self.deterministic if deterministic is None else deterministic

        if is_deterministic or self.p == 0.0:
            return x

        if key is None:
            raise ValueError("key is required when not in deterministic mode")

        keep_prob = 1.0 - self.p
        mask = jax.random.bernoulli(key, p=keep_prob, shape=x.shape)

        x = (x * mask.astype(x.dtype)) / keep_prob

        return x
