"""Graph Transformer layer from Shi et al., 2020.

Modules:
    TransformerConv  Multi-head scaled dot-product graph attention.

Glorot uniform weight init, zeros for bias.
Self-loops are not needed by default because a learned root projection includes
each receiving node's own features.
Optional edge features require `edge_dim` at init and `x_edge` at call.
Optional boolean edge mask: True = keep edge, False = ignore.
"""

import jax
import jax.numpy as jnp
from jax.nn.initializers import Initializer, glorot_uniform, zeros
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from ..nn.module import Module
from ..nn.param import Param
from .ops import segment_softmax, segment_sum


class TransformerConv(Module):
    """Multi-head scaled dot-product graph attention layer.

    >>> conv = TransformerConv(16, 32, num_heads=4, key=key)
    >>> conv(x, senders, receivers)  # (n, 16) -> (n, 32)
    >>> conv(x, senders, receivers, x_edge=x_edge, edge_mask=mask)
    """

    w_q: Param[Float[Array, "i o"]]
    w_k: Param[Float[Array, "i o"]]
    w_v: Param[Float[Array, "i o"]]
    w_root: Param[Float[Array, "i o"]] | None
    w_edge: Param[Float[Array, "f o"]] | None
    w_beta: Param[Float[Array, "p 1"]] | None
    b_out: Param[Float[Array, " o"]] | None
    num_heads: int
    edge_dim: int | None
    beta: bool

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 1,
        edge_dim: int | None = None,
        root_weight: bool = True,
        beta: bool = False,
        bias: bool = True,
        w_init: Initializer = glorot_uniform(),
        b_init: Initializer = zeros,
        *,
        key: PRNGKeyArray,
    ) -> None:

        if out_dim % num_heads != 0:
            raise ValueError(f"out_dim ({out_dim}) must be divisible by num_heads ({num_heads})")
        if beta and not root_weight:
            raise ValueError("beta=True requires root_weight=True")

        key_q, key_k, key_v, key_root, key_edge, key_beta, key_b = jax.random.split(key, 7)

        # Keep projection parameters flat; head axes exist only in activations
        self.w_q = Param(w_init(shape=(in_dim, out_dim), key=key_q))
        self.w_k = Param(w_init(shape=(in_dim, out_dim), key=key_k))
        self.w_v = Param(w_init(shape=(in_dim, out_dim), key=key_v))

        # Create the optional root, edge, and gated-residual projections
        self.w_root = Param(w_init(shape=(in_dim, out_dim), key=key_root)) if root_weight else None
        self.w_edge = (
            Param(w_init(shape=(edge_dim, out_dim), key=key_edge)) if edge_dim is not None else None
        )
        self.w_beta = Param(w_init(shape=(3 * out_dim, 1), key=key_beta)) if beta else None
        self.b_out = Param(b_init(shape=(out_dim,), key=key_b)) if bias else None

        self.num_heads = num_heads
        self.edge_dim = edge_dim
        self.beta = beta

    def __call__(
        self,
        x: Float[Array, "n i"],
        senders: Int[Array, " e"],
        receivers: Int[Array, " e"],
        *,
        x_edge: Float[Array, "e f"] | None = None,
        edge_mask: Bool[Array, " e"] | None = None,
    ) -> Float[Array, "n o"]:

        # Require edge features exactly when an edge projection was configured
        if x_edge is None and self.edge_dim is not None:
            raise ValueError(f"edge_dim={self.edge_dim} set at init but no x_edge passed at call")
        if x_edge is not None and self.edge_dim is None:
            raise ValueError("x_edge passed at call but edge_dim not set at init")

        n, i = x.shape
        head_dim = self.w_q.shape[-1] // self.num_heads

        # Project nodes, then split the output dimension into attention heads
        q = (x @ self.w_q).reshape(n, self.num_heads, head_dim)
        k = (x @ self.w_k).reshape(n, self.num_heads, head_dim)
        v = (x @ self.w_v).reshape(n, self.num_heads, head_dim)

        # Inject each edge embedding into its sender key and value
        edge_k = k[senders]
        messages = v[senders]
        if x_edge is not None:
            assert self.w_edge is not None
            e = (x_edge @ self.w_edge).reshape(-1, self.num_heads, head_dim)
            edge_k = edge_k + e
            messages = messages + e

        # Score edges and normalize over each receiver's incoming neighbourhood
        logits = (q[receivers] * edge_k).sum(axis=-1)
        logits = logits * jnp.asarray(head_dim, dtype=logits.dtype) ** -0.5
        if edge_mask is not None:
            logits = jnp.where(edge_mask[:, None], logits, -jnp.inf)
        attn = segment_softmax(logits, receivers, n)

        # Weight sender values and sum them into receiver nodes
        out = segment_sum(messages * attn[..., None], receivers, n).reshape(n, -1)

        # Include the receiving node through its learned root projection
        if self.w_root is not None:
            root = x @ self.w_root
            if self.w_beta is not None:
                gate_in = jnp.concatenate([root, out, root - out], axis=-1)
                gate = jax.nn.sigmoid(gate_in @ self.w_beta)
                out = gate * root + (1 - gate) * out
            else:
                out = out + root

        if self.b_out is not None:
            out = out + self.b_out

        return out
