"""Multi-head attention layers from Vaswani et al., 2017.

Modules:
    SelfAttention   Multi-head self-attention.
    CrossAttention  Multi-head cross-attention.

Truncated normal weight init (std=0.02), zeros for bias.
Grouped-query and multi-query attention use fewer key/value heads than query heads.
Optional boolean mask: True = attend, False = ignore.
Masks may be (s, t) shared, (b, s, t) per batch, or (b, h, s, t) per head.
"""

import jax
import jax.numpy as jnp
from jax.nn.initializers import Initializer
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from ..module import Module
from ..param import Param


class SelfAttention(Module):
    """Multi-head self-attention.

    >>> attn = SelfAttention(64, num_heads=8, key=key)
    >>> attn(x)  # (b, s, 64) -> (b, s, 64)
    >>> attn(x, mask=mask)  # mask: bool (s, s), (b, s, s) or (b, h, s, s)
    """

    w_q: Param[Float[Array, "d h k"]]
    w_k: Param[Float[Array, "d j k"]]
    w_v: Param[Float[Array, "d j k"]]
    w_out: Param[Float[Array, "h k d"]]
    b_out: Param[Float[Array, " d"]] | None
    causal: bool
    window: int | tuple[int, int] | None

    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        num_kv_heads: int | None = None,
        bias: bool = False,
        causal: bool = False,
        window: int | tuple[int, int] | None = None,
        w_init: Initializer = jax.nn.initializers.truncated_normal(0.02),
        b_init: Initializer = jax.nn.initializers.zeros,
        *,
        key: PRNGKeyArray,
    ) -> None:

        if num_kv_heads is None:
            num_kv_heads = num_heads
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
            )

        key_q, key_k, key_v, key_out, key_b = jax.random.split(key, 5)
        head_dim = dim // num_heads
        w_q = w_init(shape=(dim, num_heads * head_dim), key=key_q)
        w_k = w_init(shape=(dim, num_kv_heads * head_dim), key=key_k)
        w_v = w_init(shape=(dim, num_kv_heads * head_dim), key=key_v)
        w_out = w_init(shape=(num_heads * head_dim, dim), key=key_out)

        self.w_q = Param(w_q.reshape(dim, num_heads, head_dim))
        self.w_k = Param(w_k.reshape(dim, num_kv_heads, head_dim))
        self.w_v = Param(w_v.reshape(dim, num_kv_heads, head_dim))
        self.w_out = Param(w_out.reshape(num_heads, head_dim, dim))
        self.b_out = Param(b_init(shape=(dim,), key=key_b)) if bias else None

        self.causal = causal
        self.window = window

    def __call__(
        self,
        x: Float[Array, "b s d"],
        mask: Bool[Array, "s s"] | Bool[Array, "b s s"] | Bool[Array, "b h s s"] | None = None,
    ) -> Float[Array, "b s d"]:

        q = jnp.einsum("bsd, dhk -> bshk", x, self.w_q)
        k = jnp.einsum("bsd, djk -> bsjk", x, self.w_k)
        v = jnp.einsum("bsd, djk -> bsjk", x, self.w_v)

        if mask is not None and mask.ndim == 3:
            mask = mask[:, None]

        x = jax.nn.dot_product_attention(
            q, k, v, mask=mask, is_causal=self.causal, local_window_size=self.window
        )

        x = jnp.einsum("bshk, hkd -> bsd", x, self.w_out)

        if self.b_out is not None:
            x = x + self.b_out

        return x


class CrossAttention(Module):
    """Multi-head cross-attention.

    >>> attn = CrossAttention(64, num_heads=8, key=key)
    >>> attn(x, context)  # (b, s, 64), (b, t, 64) -> (b, s, 64)
    >>> attn(x, context, mask=mask)  # mask: bool (s, t), (b, s, t) or (b, h, s, t)
    """

    w_q: Param[Float[Array, "d h k"]]
    w_k: Param[Float[Array, "c h k"]]
    w_v: Param[Float[Array, "c h k"]]
    w_out: Param[Float[Array, "h k d"]]
    b_out: Param[Float[Array, " d"]] | None

    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        context_dim: int | None = None,
        bias: bool = False,
        w_init: Initializer = jax.nn.initializers.truncated_normal(0.02),
        b_init: Initializer = jax.nn.initializers.zeros,
        *,
        key: PRNGKeyArray,
    ) -> None:

        if context_dim is None:
            context_dim = dim
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        key_q, key_k, key_v, key_out, key_b = jax.random.split(key, 5)
        head_dim = dim // num_heads
        w_q = w_init(shape=(dim, num_heads * head_dim), key=key_q)
        w_k = w_init(shape=(context_dim, num_heads * head_dim), key=key_k)
        w_v = w_init(shape=(context_dim, num_heads * head_dim), key=key_v)
        w_out = w_init(shape=(num_heads * head_dim, dim), key=key_out)

        self.w_q = Param(w_q.reshape(dim, num_heads, head_dim))
        self.w_k = Param(w_k.reshape(context_dim, num_heads, head_dim))
        self.w_v = Param(w_v.reshape(context_dim, num_heads, head_dim))
        self.w_out = Param(w_out.reshape(num_heads, head_dim, dim))
        self.b_out = Param(b_init(shape=(dim,), key=key_b)) if bias else None

    def __call__(
        self,
        x: Float[Array, "b s d"],
        context: Float[Array, "b t c"],
        mask: Bool[Array, "s t"] | Bool[Array, "b s t"] | Bool[Array, "b h s t"] | None = None,
    ) -> Float[Array, "b s d"]:

        q = jnp.einsum("bsd, dhk -> bshk", x, self.w_q)
        k = jnp.einsum("btc, chk -> bthk", context, self.w_k)
        v = jnp.einsum("btc, chk -> bthk", context, self.w_v)

        if mask is not None and mask.ndim == 3:
            mask = mask[:, None]

        x = jax.nn.dot_product_attention(q, k, v, mask=mask)

        x = jnp.einsum("bshk, hkd -> bsd", x, self.w_out)

        if self.b_out is not None:
            x = x + self.b_out

        return x
