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
    w_kv: Param[Float[Array, "2 d j k"]]
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

        key_q, key_kv, key_out, key_b = jax.random.split(key, 4)
        head_dim = dim // num_heads
        self.w_q = Param(w_init(shape=(dim, num_heads, head_dim), key=key_q))
        self.w_kv = Param(w_init(shape=(2, dim, num_kv_heads, head_dim), key=key_kv))
        self.w_out = Param(w_init(shape=(num_heads, head_dim, dim), key=key_out))
        self.b_out = Param(b_init(shape=(dim,), key=key_b)) if bias else None

        self.causal = causal
        self.window = window

    def __call__(
        self,
        x: Float[Array, "b s d"],
        mask: Bool[Array, "s s"] | Bool[Array, "b s s"] | Bool[Array, "b h s s"] | None = None,
    ) -> Float[Array, "b s d"]:

        q = jnp.einsum("bsd, dhk -> bshk", x, self.w_q)
        k, v = jnp.einsum("bsd, idjk -> ibsjk", x, self.w_kv)

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
    w_kv: Param[Float[Array, "2 d h k"]]
    w_out: Param[Float[Array, "h k d"]]
    b_out: Param[Float[Array, " d"]] | None

    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        bias: bool = False,
        w_init: Initializer = jax.nn.initializers.truncated_normal(0.02),
        b_init: Initializer = jax.nn.initializers.zeros,
        *,
        key: PRNGKeyArray,
    ) -> None:

        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        key_q, key_kv, key_out, key_b = jax.random.split(key, 4)
        head_dim = dim // num_heads
        self.w_q = Param(w_init(shape=(dim, num_heads, head_dim), key=key_q))
        self.w_kv = Param(w_init(shape=(2, dim, num_heads, head_dim), key=key_kv))
        self.w_out = Param(w_init(shape=(num_heads, head_dim, dim), key=key_out))
        self.b_out = Param(b_init(shape=(dim,), key=key_b)) if bias else None

    def __call__(
        self,
        x: Float[Array, "b s d"],
        context: Float[Array, "b t d"],
        mask: Bool[Array, "s t"] | Bool[Array, "b s t"] | Bool[Array, "b h s t"] | None = None,
    ) -> Float[Array, "b s d"]:

        q = jnp.einsum("bsd, dhk -> bshk", x, self.w_q)
        k, v = jnp.einsum("btd, idhk -> ibthk", context, self.w_kv)

        if mask is not None and mask.ndim == 3:
            mask = mask[:, None]

        x = jax.nn.dot_product_attention(q, k, v, mask=mask)

        x = jnp.einsum("bshk, hkd -> bsd", x, self.w_out)

        if self.b_out is not None:
            x = x + self.b_out

        return x
