"""Embedding layers.

Modules:
    Embedding  Token embedding lookup table.

Fan-in variance scaling weight init (std 1/sqrt(dim)), independent of vocab size.
"""

from jax.nn.initializers import Initializer, variance_scaling
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
        *,
        w_init: Initializer = variance_scaling(1.0, "fan_in", "uniform", out_axis=0),
        key: PRNGKeyArray,
    ) -> None:

        self.w = Param(w_init(shape=(num_embeddings, dim), key=key))

    def __call__(self, x: Int[Array, "..."]) -> Float[Array, "... d"]:

        x = self.w[x]

        return x
