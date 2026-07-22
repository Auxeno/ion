"""Low-rank adaptation layers.

Modules:
    LoRALinear  Low-rank wrapper around a frozen Linear.  (Hu et al., 2021)

The wrapped Linear is frozen; only A and B are trainable.
Output scaled by alpha/rank (default alpha=rank for neutral scaling).
"""

import jax
from jax.nn.initializers import Initializer, he_normal, zeros
from jaxtyping import Array, Float, PRNGKeyArray

from ...tree import freeze
from ..module import Module
from ..param import Param
from .linear import Linear


class LoRALinear(Module):
    """Low-rank adaptation wrapper around a frozen Linear layer.

    >>> lora = LoRALinear(Linear(64, 128, key=keys[0]), rank=8, key=keys[1])
    >>> lora(x)  # (*, 64) -> (*, 128)
    """

    linear: Linear
    a: Param[Float[Array, "i r"]]
    b: Param[Float[Array, "r o"]]
    alpha: float
    rank: int

    def __init__(
        self,
        linear: Linear,
        rank: int = 8,
        alpha: float | None = None,
        a_init: Initializer = he_normal(),
        b_init: Initializer = zeros,
        *,
        key: PRNGKeyArray,
    ) -> None:

        if rank < 1:
            raise ValueError(f"rank ({rank}) must be >= 1")

        self.linear = freeze(linear)

        key_a, key_b = jax.random.split(key)
        in_dim, out_dim = linear.w.shape
        self.a = Param(a_init(shape=(in_dim, rank), key=key_a))
        self.b = Param(b_init(shape=(rank, out_dim), key=key_b))

        self.alpha = float(rank) if alpha is None else float(alpha)
        self.rank = rank

    def __call__(self, x: Float[Array, "... i"]) -> Float[Array, "... o"]:

        x = self.linear(x) + (x @ self.a @ self.b) * (self.alpha / self.rank)

        return x
