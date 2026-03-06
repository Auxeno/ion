"""Embedding layers.

Modules:
    Embedding  Token embedding lookup table.

Truncated normal weight init (std=0.02).
"""

import jax
import jax.numpy as jnp
from jax.nn.initializers import Initializer
from jaxtyping import Array, Float, Int, PRNGKeyArray

from ..module import Module
from ..param import Param


class Embedding(Module):
    """Token embedding lookup table.

    >>> embed = Embedding(1000, 64, key=key)
    >>> embed(ids)  # (*,) -> (*, 64)
    """

    w: Param[Float[Array, "v d"]]

    def __init__(
        self,
        num_embeddings: int,
        dim: int,
        dtype: jnp.dtype = jnp.float32,
        w_init: Initializer = jax.nn.initializers.truncated_normal(0.02),
        *,
        key: PRNGKeyArray,
    ) -> None:

        self.w = Param(w_init(shape=(num_embeddings, dim), dtype=dtype, key=key))

    def __call__(self, x: Int[Array, "..."]) -> Float[Array, "... d"]:

        x = self.w[x]

        return x
