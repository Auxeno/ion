"""Multi-head attention layers from Vaswani et al., 2017.

Modules:
    SelfAttention   Fused QKV projection for self-attention.
    CrossAttention  Separate Q and KV projections for cross-attention.

Truncated normal weight init (std=0.02), zeros for bias.
QKV fused into a single weight matrix for self-attention.
Optional boolean mask: True = attend, False = ignore.
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
    >>> attn(x)  # (*, seq, 64) -> (*, seq, 64)
    >>> attn(x, mask=mask)  # mask: bool (*, s, s) or (*, 1, s, s)
    """

    w_qkv: Param[Float[Array, "d 3 n h"]]
    w_out: Param[Float[Array, "n h d"]]
    b_out: Param[Float[Array, " d"]] | None
    causal: bool

    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        bias: bool = False,
        causal: bool = False,
        dtype: jnp.dtype = jnp.float32,
        w_init: Initializer = jax.nn.initializers.truncated_normal(0.02),
        b_init: Initializer = jax.nn.initializers.zeros,
        *,
        key: PRNGKeyArray,
    ) -> None:

        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        key_w_qkv, key_w_out, key_b_out = jax.random.split(key, 3)
        head_dim = dim // num_heads
        self.w_qkv = Param(w_init(shape=(dim, 3, num_heads, head_dim), dtype=dtype, key=key_w_qkv))
        self.w_out = Param(w_init(shape=(num_heads, head_dim, dim), dtype=dtype, key=key_w_out))
        self.b_out = Param(b_init(shape=(dim,), dtype=dtype, key=key_b_out)) if bias else None

        self.causal = causal

    def __call__(
        self,
        x: Float[Array, "... s d"],
        mask: Bool[Array, "... s s"] | Bool[Array, "... 1 s s"] | None = None,
    ) -> Float[Array, "... s d"]:

        qkv = jnp.einsum("...d, dinh -> ...inh", x, self.w_qkv)
        q, k, v = jnp.moveaxis(qkv, -3, 0)

        logits = jnp.einsum("...snh, ...tnh -> ...nst", q, k) / jnp.sqrt(self.w_qkv.shape[-1])

        if self.causal:
            causal_mask = jnp.tril(jnp.ones(logits.shape[-2:], dtype=bool))
            logits = jnp.where(causal_mask, logits, -jnp.inf)

        if mask is not None:
            logits = jnp.where(mask, logits, -jnp.inf)

        attention = jax.nn.softmax(logits, axis=-1)
        x = jnp.einsum("...nst, ...tnh -> ...snh", attention, v)

        x = jnp.einsum("...nh, nhd -> ...d", x, self.w_out)

        if self.b_out is not None:
            x = x + self.b_out

        return x


class CrossAttention(Module):
    """Multi-head cross-attention.

    >>> attn = CrossAttention(64, num_heads=8, key=key)
    >>> attn(x, context)  # (*, s, 64), (*, t, 64) -> (*, s, 64)
    >>> attn(x, context, mask=mask)  # mask: bool (*, s, t) or (*, 1, s, t)
    """

    w_q: Param[Float[Array, "d n h"]]
    w_kv: Param[Float[Array, "d 2 n h"]]
    w_out: Param[Float[Array, "n h d"]]
    b_out: Param[Float[Array, " d"]] | None

    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        bias: bool = False,
        dtype: jnp.dtype = jnp.float32,
        w_init: Initializer = jax.nn.initializers.truncated_normal(0.02),
        b_init: Initializer = jax.nn.initializers.zeros,
        *,
        key: PRNGKeyArray,
    ) -> None:

        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        key_w_q, key_w_kv, key_w_out, key_b_out = jax.random.split(key, 4)
        head_dim = dim // num_heads
        self.w_q = Param(w_init(shape=(dim, num_heads, head_dim), dtype=dtype, key=key_w_q))
        self.w_kv = Param(w_init(shape=(dim, 2, num_heads, head_dim), dtype=dtype, key=key_w_kv))
        self.w_out = Param(w_init(shape=(num_heads, head_dim, dim), dtype=dtype, key=key_w_out))
        self.b_out = Param(b_init(shape=(dim,), dtype=dtype, key=key_b_out)) if bias else None

    def __call__(
        self,
        x: Float[Array, "... s d"],
        context: Float[Array, "... t d"],
        mask: Bool[Array, "... s t"] | Bool[Array, "... 1 s t"] | None = None,
    ) -> Float[Array, "... s d"]:

        q = jnp.einsum("...d, dnh -> ...nh", x, self.w_q)
        kv = jnp.einsum("...d, dinh -> ...inh", context, self.w_kv)
        k, v = jnp.moveaxis(kv, -3, 0)

        logits = jnp.einsum("...snh, ...tnh -> ...nst", q, k) / jnp.sqrt(self.w_q.shape[-1])

        if mask is not None:
            logits = jnp.where(mask, logits, -jnp.inf)

        attention = jax.nn.softmax(logits, axis=-1)
        x = jnp.einsum("...nst, ...tnh -> ...snh", attention, v)

        x = jnp.einsum("...nh, nhd -> ...d", x, self.w_out)

        if self.b_out is not None:
            x = x + self.b_out

        return x
