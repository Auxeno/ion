"""Graph Attention Network layers.

Modules:
    GATConv    Multi-head graph attention.          (Velickovic et al., 2018)
    GATv2Conv  Multi-head dynamic graph attention.  (Brody et al., 2022)

Glorot uniform weight init to match original papers, zeros for bias.
Self-loops are the caller's responsibility, see `gnn.add_self_loops`.
Optional edge features require `edge_dim` at init and `x_edge` at call.
Optional boolean edge mask: True = keep edge, False = ignore.
"""

import jax
import jax.numpy as jnp
from jax.nn.initializers import Initializer, glorot_uniform, zeros
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from ...nn.module import Module
from ...nn.param import Param
from ..ops import segment_softmax, segment_sum


class GATConv(Module):
    """Multi-head graph attention layer.

    >>> gat = GATConv(16, 32, num_heads=4, key=key)
    >>> gat(x, senders, receivers)  # (n, 16) -> (n, 32)
    >>> gat(x, senders, receivers, edge_mask=mask)  # mask: bool (e,)
    """

    w: Param[Float[Array, "i o"]]
    w_receiver: Param[Float[Array, "j o"]] | None
    att_sender: Param[Float[Array, "h k"]]
    att_receiver: Param[Float[Array, "h k"]]
    w_edge: Param[Float[Array, "f o"]] | None
    att_edge: Param[Float[Array, "h k"]] | None
    b_out: Param[Float[Array, " o"]] | None
    num_heads: int
    negative_slope: float
    edge_dim: int | None

    def __init__(
        self,
        in_dim: int | tuple[int, int],
        out_dim: int,
        *,
        num_heads: int = 1,
        edge_dim: int | None = None,
        negative_slope: float = 0.2,
        use_bias: bool = True,
        w_init: Initializer = glorot_uniform(),
        att_init: Initializer = glorot_uniform(),
        b_init: Initializer = zeros,
        key: PRNGKeyArray,
    ) -> None:

        head_dim = out_dim // num_heads

        if out_dim % num_heads != 0:
            raise ValueError(f"out_dim ({out_dim}) must be divisible by num_heads ({num_heads})")

        keys = jax.random.split(key, 6)
        key_w, key_att_sender, key_att_receiver, key_w_edge, key_att_edge, key_b = keys
        if isinstance(in_dim, tuple):
            in_src, in_dst = in_dim
            key_w_sender, key_w_receiver = jax.random.split(key_w)
            self.w = Param(w_init(shape=(in_src, out_dim), key=key_w_sender))
            self.w_receiver = Param(w_init(shape=(in_dst, out_dim), key=key_w_receiver))
        else:
            self.w = Param(w_init(shape=(in_dim, out_dim), key=key_w))
            self.w_receiver = None
        self.att_sender = Param(att_init(shape=(num_heads, head_dim), key=key_att_sender))
        self.att_receiver = Param(att_init(shape=(num_heads, head_dim), key=key_att_receiver))

        if edge_dim is not None:
            self.w_edge = Param(w_init(shape=(edge_dim, out_dim), key=key_w_edge))
            self.att_edge = Param(att_init(shape=(num_heads, head_dim), key=key_att_edge))
        else:
            self.w_edge = None
            self.att_edge = None

        self.b_out = Param(b_init(shape=(out_dim,), key=key_b)) if use_bias else None

        self.num_heads = num_heads
        self.negative_slope = negative_slope
        self.edge_dim = edge_dim

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
        x_src = (x_src @ self.w).reshape(n_src, self.num_heads, -1)
        receiver_weight = self.w if self.w_receiver is None else self.w_receiver
        x_dst = (x_dst @ receiver_weight).reshape(n_dst, self.num_heads, -1)

        # Compute attention scores at node level, then combine at edges
        logits_sender = (x_src * self.att_sender).sum(axis=-1)
        logits_receiver = (x_dst * self.att_receiver).sum(axis=-1)
        logits = logits_sender[senders] + logits_receiver[receivers]

        # Add edge feature contribution
        if x_edge is not None:
            assert self.w_edge is not None
            assert self.att_edge is not None
            if edge_mask is not None:
                x_edge = x_edge * edge_mask[:, None]
            edge = (x_edge @ self.w_edge).reshape(-1, self.num_heads, x_src.shape[-1])
            logits = logits + (edge * self.att_edge).sum(axis=-1)

        logits = jax.nn.leaky_relu(logits, self.negative_slope)

        if edge_mask is not None:
            logits = jnp.where(edge_mask[:, None], logits, -jnp.inf)

        # Softmax over each receiver's incoming edges
        attention = segment_softmax(logits, receivers, n_dst)

        # Aggregate sender features weighted by attention
        messages = x_src[senders] * attention[..., None]
        x_out = segment_sum(messages, receivers, n_dst)

        x_out = x_out.reshape(n_dst, -1)

        if self.b_out is not None:
            x_out = x_out + self.b_out

        return x_out


class GATv2Conv(Module):
    """Multi-head dynamic graph attention layer.

    >>> gat = GATv2Conv(16, 32, num_heads=4, key=key)
    >>> gat(x, senders, receivers)  # (n, 16) -> (n, 32)
    >>> gat(x, senders, receivers, edge_mask=mask)  # mask: bool (e,)
    """

    w_sender: Param[Float[Array, "i o"]]
    w_receiver: Param[Float[Array, "j o"]]
    att: Param[Float[Array, "h k"]]
    w_edge: Param[Float[Array, "f o"]] | None
    b_out: Param[Float[Array, " o"]] | None
    num_heads: int
    negative_slope: float
    edge_dim: int | None

    def __init__(
        self,
        in_dim: int | tuple[int, int],
        out_dim: int,
        *,
        num_heads: int = 1,
        edge_dim: int | None = None,
        negative_slope: float = 0.2,
        use_bias: bool = True,
        w_init: Initializer = glorot_uniform(),
        att_init: Initializer = glorot_uniform(),
        b_init: Initializer = zeros,
        key: PRNGKeyArray,
    ) -> None:

        head_dim = out_dim // num_heads

        if out_dim % num_heads != 0:
            raise ValueError(f"out_dim ({out_dim}) must be divisible by num_heads ({num_heads})")

        in_src, in_dst = in_dim if isinstance(in_dim, tuple) else (in_dim, in_dim)

        key_w_sender, key_w_receiver, key_att, key_w_edge, key_b = jax.random.split(key, 5)
        self.w_sender = Param(w_init(shape=(in_src, out_dim), key=key_w_sender))
        self.w_receiver = Param(w_init(shape=(in_dst, out_dim), key=key_w_receiver))
        self.att = Param(att_init(shape=(num_heads, head_dim), key=key_att))

        if edge_dim is not None:
            self.w_edge = Param(w_init(shape=(edge_dim, out_dim), key=key_w_edge))
        else:
            self.w_edge = None

        self.b_out = Param(b_init(shape=(out_dim,), key=key_b)) if use_bias else None

        self.num_heads = num_heads
        self.negative_slope = negative_slope
        self.edge_dim = edge_dim

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
        x_s = (x_src @ self.w_sender).reshape(n_src, self.num_heads, -1)
        x_r = (x_dst @ self.w_receiver).reshape(n_dst, self.num_heads, -1)

        # Sum of the two projections is the paper's concatenation, factored
        edge_h = x_s[senders] + x_r[receivers]

        # Add edge feature contribution before the nonlinearity
        if x_edge is not None:
            assert self.w_edge is not None
            if edge_mask is not None:
                x_edge = x_edge * edge_mask[:, None]
            edge = (x_edge @ self.w_edge).reshape(-1, self.num_heads, x_s.shape[-1])
            edge_h = edge_h + edge

        # Apply nonlinearity then dot with attention vector (GATv2 difference)
        edge_h = jax.nn.leaky_relu(edge_h, self.negative_slope)
        logits = (edge_h * self.att).sum(axis=-1)

        if edge_mask is not None:
            logits = jnp.where(edge_mask[:, None], logits, -jnp.inf)

        # Softmax over each receiver's incoming edges
        attention = segment_softmax(logits, receivers, n_dst)

        # Aggregate sender features weighted by attention
        messages = x_s[senders] * attention[..., None]
        x_out = segment_sum(messages, receivers, n_dst)

        x_out = x_out.reshape(n_dst, -1)

        if self.b_out is not None:
            x_out = x_out + self.b_out

        return x_out
