"""Low-rank adaptation layers.

Modules:
    LoRALinear  Low-rank wrapper around a frozen Linear.  (Hu et al., 2021)

The wrapped Linear is frozen; only A and B are trainable.
Output scaled by alpha/rank (default alpha=rank for neutral scaling).
"""

import jax
import jax.numpy as jnp
from jax.nn.initializers import Initializer
from jaxtyping import Array, Float, PRNGKeyArray

from ...tree import freeze
from ..module import Module
from ..param import Param
from .linear import Linear


class LoRALinear(Module):
    """Low-rank adaptation wrapper around a frozen Linear layer.

    >>> lora = LoRALinear(Linear(64, 128, key=key_1), rank=8, key=key_2)
    >>> lora(x)  # (*, 64) -> (*, 128)
    """

    linear: Linear
    a: Param[Float[Array, "id r"]]
    b: Param[Float[Array, "r od"]]
    alpha: float
    rank: int

    def __init__(
        self,
        linear: Linear,
        rank: int = 8,
        alpha: float | None = None,
        dtype: jnp.dtype = jnp.float32,
        a_init: Initializer = jax.nn.initializers.he_normal(),
        b_init: Initializer = jax.nn.initializers.zeros,
        *,
        key: PRNGKeyArray,
    ) -> None:

        self.linear = freeze(linear)

        key_a, key_b = jax.random.split(key)
        in_dim, out_dim = linear.w.shape
        self.a = Param(a_init(shape=(in_dim, rank), dtype=dtype, key=key_a))
        self.b = Param(b_init(shape=(rank, out_dim), dtype=dtype, key=key_b))

        self.alpha = float(rank) if alpha is None else float(alpha)
        self.rank = rank

    def __call__(self, x: Float[Array, "... id"]) -> Float[Array, "... od"]:

        x = self.linear(x) + (x @ self.a @ self.b) * (self.alpha / self.rank)

        return x
