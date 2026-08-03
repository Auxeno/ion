"""Multi-head attention layers from Vaswani et al., 2017.

Modules:
    SelfAttention   Multi-head self-attention.
    CrossAttention  Multi-head cross-attention.

Glorot uniform weight init, zeros for bias.
Grouped-query and multi-query attention use fewer key/value heads than query heads.
Optional boolean mask: True = attend, False = ignore.
Masks may be (s, t) shared, (b, s, t) per batch, or (b, h, s, t) per head.
"""

import jax
from jax.nn.initializers import Initializer, glorot_uniform, zeros
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from ..module import Module
from ..param import Param


class SelfAttention(Module):
    """Multi-head self-attention.

    >>> attn = SelfAttention(64, num_heads=8, key=key)
    >>> attn(x)  # (b, s, 64) -> (b, s, 64)
    >>> attn(x, mask=mask)  # mask: bool (s, s), (b, s, s) or (b, h, s, s)
    """

    w_q: Param[Float[Array, "d hk"]]
    w_k: Param[Float[Array, "d jk"]]
    w_v: Param[Float[Array, "d jk"]]
    w_out: Param[Float[Array, "hk d"]]
    b_out: Param[Float[Array, " d"]] | None
    num_heads: int
    num_kv_heads: int
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
        w_init: Initializer = glorot_uniform(),
        b_init: Initializer = zeros,
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

        head_dim = dim // num_heads

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.causal = causal
        self.window = window

        key_q, key_k, key_v, key_out, key_b = jax.random.split(key, 5)
        self.w_q = Param(w_init(shape=(dim, num_heads * head_dim), key=key_q))
        self.w_k = Param(w_init(shape=(dim, num_kv_heads * head_dim), key=key_k))
        self.w_v = Param(w_init(shape=(dim, num_kv_heads * head_dim), key=key_v))
        self.w_out = Param(w_init(shape=(num_heads * head_dim, dim), key=key_out))
        self.b_out = Param(b_init(shape=(dim,), key=key_b)) if bias else None

    def __call__(
        self,
        x: Float[Array, "b s d"],
        mask: Bool[Array, "s s"] | Bool[Array, "b s s"] | Bool[Array, "b h s s"] | None = None,
    ) -> Float[Array, "b s d"]:

        b, s, d = x.shape

        q = (x @ self.w_q).reshape(b, s, self.num_heads, -1)
        k = (x @ self.w_k).reshape(b, s, self.num_kv_heads, -1)
        v = (x @ self.w_v).reshape(b, s, self.num_kv_heads, -1)

        if mask is not None and mask.ndim == 3:
            mask = mask[:, None]

        x = jax.nn.dot_product_attention(
            q, k, v, mask=mask, is_causal=self.causal, local_window_size=self.window
        )

        x = x.reshape(b, s, -1) @ self.w_out

        if self.b_out is not None:
            x = x + self.b_out

        return x


class CrossAttention(Module):
    """Multi-head cross-attention.

    >>> attn = CrossAttention(64, num_heads=8, key=key)
    >>> attn(x, context)  # (b, s, 64), (b, t, 64) -> (b, s, 64)
    >>> attn(x, context, mask=mask)  # mask: bool (s, t), (b, s, t) or (b, h, s, t)
    """

    w_q: Param[Float[Array, "d hk"]]
    w_k: Param[Float[Array, "c hk"]]
    w_v: Param[Float[Array, "c hk"]]
    w_out: Param[Float[Array, "hk d"]]
    b_out: Param[Float[Array, " d"]] | None
    num_heads: int

    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        context_dim: int | None = None,
        bias: bool = False,
        w_init: Initializer = glorot_uniform(),
        b_init: Initializer = zeros,
        *,
        key: PRNGKeyArray,
    ) -> None:

        if context_dim is None:
            context_dim = dim
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        head_dim = dim // num_heads

        self.num_heads = num_heads

        key_q, key_k, key_v, key_out, key_b = jax.random.split(key, 5)
        self.w_q = Param(w_init(shape=(dim, num_heads * head_dim), key=key_q))
        self.w_k = Param(w_init(shape=(context_dim, num_heads * head_dim), key=key_k))
        self.w_v = Param(w_init(shape=(context_dim, num_heads * head_dim), key=key_v))
        self.w_out = Param(w_init(shape=(num_heads * head_dim, dim), key=key_out))
        self.b_out = Param(b_init(shape=(dim,), key=key_b)) if bias else None

    def __call__(
        self,
        x: Float[Array, "b s d"],
        context: Float[Array, "b t c"],
        mask: Bool[Array, "s t"] | Bool[Array, "b s t"] | Bool[Array, "b h s t"] | None = None,
    ) -> Float[Array, "b s d"]:

        b, s, d = x.shape
        t = context.shape[1]

        q = (x @ self.w_q).reshape(b, s, self.num_heads, -1)
        k = (context @ self.w_k).reshape(b, t, self.num_heads, -1)
        v = (context @ self.w_v).reshape(b, t, self.num_heads, -1)

        if mask is not None and mask.ndim == 3:
            mask = mask[:, None]

        x = jax.nn.dot_product_attention(q, k, v, mask=mask)

        x = x.reshape(b, s, -1) @ self.w_out

        if self.b_out is not None:
            x = x + self.b_out

        return x
