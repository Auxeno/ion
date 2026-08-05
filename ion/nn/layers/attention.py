"""Multi-head attention layers from Vaswani et al., 2017.

Modules:
    MultiHeadAttention  Multi-head self and cross-attention.

Functions:
    dot_product_attention_with_rope  Apply RoPE before dot-product attention.

Glorot uniform weight init, zeros for bias.
Keys and values come from the input unless a separate context is passed.
Grouped-query and multi-query attention use fewer key/value heads than query heads.
Optional boolean mask: True = attend, False = ignore.
Masks may be (s, t) shared, (b, s, t) per batch, or (b, h, s, t) per head.
"""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax.nn.initializers import Initializer, glorot_uniform, zeros
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from ..module import Module
from ..param import Param
from .positional import RoPE


class MultiHeadAttention(Module):
    """Multi-head attention.

    >>> attn = MultiHeadAttention(64, num_heads=8, key=key)
    >>> attn(x)  # self-attention: (b, s, 64) -> (b, s, 64)
    >>> attn(x, x_kv)  # cross-attention: (b, s, 64), (b, t, 64) -> (b, s, 64)
    >>> attn(x, mask=mask)  # mask: bool (s, t), (b, s, t) or (b, h, s, t)
    """

    w_q: Param[Float[Array, "d hk"]]
    w_k: Param[Float[Array, "c jk"]]
    w_v: Param[Float[Array, "c jk"]]
    w_out: Param[Float[Array, "hk d"]]
    b_out: Param[Float[Array, " d"]] | None
    num_heads: int
    num_kv_heads: int
    causal: bool
    window: int | tuple[int, int] | None
    attention_fn: Callable[..., Array]

    def __init__(
        self,
        dim: int,
        *,
        num_heads: int = 1,
        num_kv_heads: int | None = None,
        kv_dim: int | None = None,
        use_bias: bool = False,
        causal: bool = False,
        window: int | tuple[int, int] | None = None,
        w_init: Initializer = glorot_uniform(),
        b_init: Initializer = zeros,
        attention_fn: Callable[..., Array] = jax.nn.dot_product_attention,
        key: PRNGKeyArray,
    ) -> None:

        num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        kv_dim = dim if kv_dim is None else kv_dim
        head_dim = dim // num_heads

        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
            )

        key_q, key_k, key_v, key_out, key_b = jax.random.split(key, 5)
        self.w_q = Param(w_init(shape=(dim, num_heads * head_dim), key=key_q))
        self.w_k = Param(w_init(shape=(kv_dim, num_kv_heads * head_dim), key=key_k))
        self.w_v = Param(w_init(shape=(kv_dim, num_kv_heads * head_dim), key=key_v))
        self.w_out = Param(w_init(shape=(num_heads * head_dim, dim), key=key_out))
        self.b_out = Param(b_init(shape=(dim,), key=key_b)) if use_bias else None

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.causal = causal
        self.window = window
        self.attention_fn = attention_fn

    def __call__(
        self,
        x: Float[Array, "b s d"],
        x_kv: Float[Array, "b t c"] | None = None,
        mask: Bool[Array, "s t"] | Bool[Array, "b s t"] | Bool[Array, "b h s t"] | None = None,
    ) -> Float[Array, "b s d"]:

        x_kv = x if x_kv is None else x_kv
        b, s, d = x.shape
        t = x_kv.shape[1]

        q = (x @ self.w_q).reshape(b, s, self.num_heads, -1)
        k = (x_kv @ self.w_k).reshape(b, t, self.num_kv_heads, -1)
        v = (x_kv @ self.w_v).reshape(b, t, self.num_kv_heads, -1)

        if mask is not None and mask.ndim < 4:
            mask = mask.reshape(-1, 1, *mask.shape[-2:])

        x = self.attention_fn(
            q, k, v, mask=mask, is_causal=self.causal, local_window_size=self.window
        )

        # Zero out fully masked query rows
        if mask is not None:
            valid = mask.any(axis=-1).swapaxes(-1, -2)[..., None]
            x = jnp.where(valid, x, 0.0)

        x = x.reshape(b, s, -1) @ self.w_out

        if self.b_out is not None:
            x = x + self.b_out

        return x


def dot_product_attention_with_rope(
    query: Float[Array, "b s h k"],
    key: Float[Array, "b t j k"],
    value: Float[Array, "b t j k"],
    *,
    rope: RoPE,
    **kwargs: Any,
) -> Float[Array, "b s h k"]:
    """Apply rotary embeddings to query and key before dot-product attention."""
    return jax.nn.dot_product_attention(rope(query), rope(key), value, **kwargs)
