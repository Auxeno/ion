"""Transformer blocks.

Modules:
    TransformerBlock       Pre-norm self-attention + FFN block.
    CrossTransformerBlock  Pre-norm cross-attention + FFN block.

Pre-norm architecture: LayerNorm before each sublayer, residual add after.
FFN uses 4x hidden dimension with GELU activation by default.
Truncated normal weight init (std=0.02), zeros for bias.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.nn.initializers import Initializer
from jaxtyping import Array, Float, PRNGKeyArray

from ..layers.attention import CrossAttention, SelfAttention
from ..layers.linear import Linear
from ..layers.norm import LayerNorm
from ..module import Module


class TransformerBlock(Module):
    """Pre-norm transformer encoder block (self-attention + FFN).

    >>> block = TransformerBlock(64, num_heads=8, key=key)
    >>> block(x)  # (*, seq, 64) -> (*, seq, 64)
    """

    att: SelfAttention
    norm_att: LayerNorm
    norm_ff: LayerNorm
    ff_1: Linear
    ff_2: Linear
    activation: Callable[[Array], Array]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        ff_dim: int | None = None,
        bias: bool = False,
        causal: bool = False,
        activation: Callable[[Array], Array] = jax.nn.gelu,
        dtype: jnp.dtype = jnp.float32,
        w_init: Initializer = jax.nn.initializers.truncated_normal(0.02),
        b_init: Initializer = jax.nn.initializers.zeros,
        *,
        key: PRNGKeyArray,
    ) -> None:

        if ff_dim is None:
            ff_dim = 4 * dim

        key_att, key_ff_1, key_ff_2 = jax.random.split(key, 3)
        self.att = SelfAttention(dim, num_heads, bias, causal, dtype, w_init, b_init, key=key_att)
        self.norm_att = LayerNorm(dim, dtype=dtype)
        self.norm_ff = LayerNorm(dim, dtype=dtype)
        self.ff_1 = Linear(dim, ff_dim, bias, dtype, w_init, b_init, key=key_ff_1)
        self.ff_2 = Linear(ff_dim, dim, bias, dtype, w_init, b_init, key=key_ff_2)

        self.activation = activation

    def __call__(self, x: Float[Array, "... s d"]) -> Float[Array, "... s d"]:

        residual = x
        x = self.norm_att(x)
        x = self.att(x)
        x = x + residual

        residual = x
        x = self.norm_ff(x)
        x = self.ff_1(x)
        x = self.activation(x)
        x = self.ff_2(x)
        x = x + residual

        return x


class CrossTransformerBlock(Module):
    """Pre-norm transformer block with cross-attention.

    >>> block = CrossTransformerBlock(64, num_heads=8, key=key)
    >>> block(x, context)  # (*, s, 64), (*, t, 64) -> (*, s, 64)
    """

    att: CrossAttention
    norm_att: LayerNorm
    norm_ff: LayerNorm
    ff_1: Linear
    ff_2: Linear
    activation: Callable[[Array], Array]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        ff_dim: int | None = None,
        bias: bool = False,
        activation: Callable[[Array], Array] = jax.nn.gelu,
        dtype: jnp.dtype = jnp.float32,
        w_init: Initializer = jax.nn.initializers.truncated_normal(0.02),
        b_init: Initializer = jax.nn.initializers.zeros,
        *,
        key: PRNGKeyArray,
    ) -> None:

        if ff_dim is None:
            ff_dim = 4 * dim

        key_att, key_ff_1, key_ff_2 = jax.random.split(key, 3)
        self.att = CrossAttention(dim, num_heads, bias, dtype, w_init, b_init, key=key_att)
        self.norm_att = LayerNorm(dim, dtype=dtype)
        self.norm_ff = LayerNorm(dim, dtype=dtype)
        self.ff_1 = Linear(dim, ff_dim, bias, dtype, w_init, b_init, key=key_ff_1)
        self.ff_2 = Linear(ff_dim, dim, bias, dtype, w_init, b_init, key=key_ff_2)

        self.activation = activation

    def __call__(
        self,
        x: Float[Array, "... s d"],
        context: Float[Array, "... t d"],
    ) -> Float[Array, "... s d"]:

        residual = x
        x = self.norm_att(x)
        x = self.att(x, context)
        x = x + residual

        residual = x
        x = self.norm_ff(x)
        x = self.ff_1(x)
        x = self.activation(x)
        x = self.ff_2(x)
        x = x + residual

        return x
