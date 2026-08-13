"""Graph Transformer layer from Shi et al., 2020.

Modules:
    TransformerConv  Multi-head scaled dot-product graph attention.

Glorot uniform weight init, zeros for bias.
Self-loops are not needed by default because a learned root projection includes
each receiving node's own features.
Optional edge features require `edge_dim` at init and `x_edge` at call.
Optional boolean edge mask: True = keep edge, False = ignore.
"""

import math

import jax
import jax.numpy as jnp
from jax.nn.initializers import Initializer, glorot_uniform, zeros
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from ...nn.module import Module
from ...nn.param import Param
from ..ops import segment_softmax, segment_sum


class TransformerConv(Module):
    """Multi-head scaled dot-product graph attention layer.

    >>> conv = TransformerConv(16, 32, num_heads=4, key=key)
    >>> conv(x, senders, receivers)  # (n, 16) -> (n, 32)
    >>> conv(x, senders, receivers, x_edge=x_edge, edge_mask=mask)
    """

    w_q: Param[Float[Array, "j o"]]
    w_k: Param[Float[Array, "i o"]]
    w_v: Param[Float[Array, "i o"]]
    w_root: Param[Float[Array, "j o"]] | None
    w_edge: Param[Float[Array, "f o"]] | None
    w_beta: Param[Float[Array, "p 1"]] | None
    b_out: Param[Float[Array, " o"]] | None
    num_heads: int
    edge_dim: int | None
    use_beta: bool

    def __init__(
        self,
        in_dim: int | tuple[int, int],
        out_dim: int,
        *,
        num_heads: int = 1,
        edge_dim: int | None = None,
        use_root_weight: bool = True,
        use_beta: bool = False,
        use_bias: bool = True,
        w_init: Initializer = glorot_uniform(),
        b_init: Initializer = zeros,
        key: PRNGKeyArray,
    ) -> None:

        if out_dim % num_heads != 0:
            raise ValueError(f"out_dim ({out_dim}) must be divisible by num_heads ({num_heads})")
        if use_beta and not use_root_weight:
            raise ValueError("use_beta=True requires use_root_weight=True")

        in_src, in_dst = in_dim if isinstance(in_dim, tuple) else (in_dim, in_dim)

        key_q, key_k, key_v, key_root, key_edge, key_beta, key_b = jax.random.split(key, 7)
        self.w_q = Param(w_init(shape=(in_dst, out_dim), key=key_q))
        self.w_k = Param(w_init(shape=(in_src, out_dim), key=key_k))
        self.w_v = Param(w_init(shape=(in_src, out_dim), key=key_v))
        self.w_root = (
            Param(w_init(shape=(in_dst, out_dim), key=key_root)) if use_root_weight else None
        )
        self.w_edge = (
            Param(w_init(shape=(edge_dim, out_dim), key=key_edge)) if edge_dim is not None else None
        )
        self.w_beta = Param(w_init(shape=(3 * out_dim, 1), key=key_beta)) if use_beta else None
        self.b_out = Param(b_init(shape=(out_dim,), key=key_b)) if use_bias else None

        self.num_heads = num_heads
        self.edge_dim = edge_dim
        self.use_beta = use_beta

    def __call__(
        self,
        x: Float[Array, "n i"] | tuple[Float[Array, "s i"], Float[Array, "t j"]],
        senders: Int[Array, " e"],
        receivers: Int[Array, " e"],
        *,
        x_edge: Float[Array, "e f"] | None = None,
        edge_mask: Bool[Array, " e"] | None = None,
    ) -> Float[Array, "t o"]:

        if x_edge is None and self.edge_dim is not None:
            raise ValueError(f"edge_dim={self.edge_dim} set at init but no x_edge passed at call")
        if x_edge is not None and self.edge_dim is None:
            raise ValueError("x_edge passed at call but edge_dim not set at init")

        x_src, x_dst = x if isinstance(x, tuple) else (x, x)
        n_src, n_dst = x_src.shape[0], x_dst.shape[0]

        # Project nodes
        q = (x_dst @ self.w_q).reshape(n_dst, self.num_heads, -1)
        k = (x_src @ self.w_k).reshape(n_src, self.num_heads, -1)
        v = (x_src @ self.w_v).reshape(n_src, self.num_heads, -1)
        head_dim = q.shape[-1]

        # Gather sender keys and values, adding the edge embedding to both
        edge_k = k[senders]
        edge_v = v[senders]
        if x_edge is not None:
            assert self.w_edge is not None
            if edge_mask is not None:
                x_edge = jnp.where(edge_mask[:, None], x_edge, 0.0)
            e = (x_edge @ self.w_edge).reshape(-1, self.num_heads, head_dim)
            edge_k = edge_k + e
            edge_v = edge_v + e

        # Scaled dot product between each receiver's query and its senders' keys
        logits = (q[receivers] * edge_k).sum(axis=-1) / math.sqrt(head_dim)

        if edge_mask is not None:
            logits = jnp.where(edge_mask[:, None], logits, -jnp.inf)

        # Softmax over each receiver's incoming edges
        attention = segment_softmax(logits, receivers, n_dst)

        # Aggregate sender values weighted by attention
        out = segment_sum(edge_v * attention[..., None], receivers, n_dst).reshape(n_dst, -1)

        # Include the receiving node through its learned root projection
        if self.w_root is not None:
            root = x_dst @ self.w_root
            if self.w_beta is not None:
                gate_in = jnp.concatenate([root, out, root - out], axis=-1)
                gate = jax.nn.sigmoid(gate_in @ self.w_beta)
                out = gate * root + (1 - gate) * out
            else:
                out = out + root

        if self.b_out is not None:
            out = out + self.b_out

        return out
