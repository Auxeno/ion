"""Linear layers.

Modules:
    Linear    Fully connected layer.
    Identity  Pass-through, returns input unchanged.

He normal weight init for ReLU activation, zeros for bias.
"""

from typing import Any

import jax
import jax.numpy as jnp
from jax.nn.initializers import Initializer
from jaxtyping import Array, Float, PRNGKeyArray

from ..module import Module
from ..param import Param


class Linear(Module):
    """Fully connected linear layer.

    >>> linear = Linear(3, 16, key=key)
    >>> linear(x)  # (*, 3) -> (*, 16)
    """

    w: Param[Float[Array, "id od"]]
    b: Param[Float[Array, " od"]] | None

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        dtype: jnp.dtype = jnp.float32,
        w_init: Initializer = jax.nn.initializers.he_normal(),
        b_init: Initializer = jax.nn.initializers.zeros,
        *,
        key: PRNGKeyArray,
    ) -> None:

        key_w, key_b = jax.random.split(key)
        self.w = Param(w_init(shape=(in_dim, out_dim), dtype=dtype, key=key_w))
        self.b = Param(b_init(shape=(out_dim,), dtype=dtype, key=key_b)) if bias else None

    def __call__(self, x: Float[Array, "... id"]) -> Float[Array, "... od"]:

        x = x @ self.w

        if self.b is not None:
            x = x + self.b

        return x


class Identity(Module):
    """Pass-through layer, ignores all arguments."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __call__(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        return x
