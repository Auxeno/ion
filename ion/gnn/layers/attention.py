"""Graph attention layers.

Modules:
    GATConv          Multi-head graph attention.               (Velickovic et al., 2018)
    GATv2Conv        Multi-head dynamic graph attention.       (Brody et al., 2022)
    TransformerConv  Multi-head scaled dot-product attention.  (Shi et al., 2020)

Glorot uniform weight init to match original papers, zeros for bias.
GATConv and GATv2Conv self-loops are the caller's responsibility, see `gnn.add_self_loops`.
TransformerConv does not need them because a learned root projection includes each
receiving node's own features.
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


class GATConv(Module):
    """Multi-head graph attention layer.

    >>> gat = GATConv(16, 32, num_heads=4, key=key)
    >>> gat(x, senders, receivers)  # (n, 16) -> (n, 32)
    >>> gat(x, senders, receivers, edge_mask=mask)  # mask: bool (e,)
    """

    w_sender: Param[Float[Array, "i o"]]
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

        in_src, in_dst = in_dim if isinstance(in_dim, tuple) else (in_dim, in_dim)
        head_dim = out_dim // num_heads

        if out_dim % num_heads != 0:
            raise ValueError(f"out_dim ({out_dim}) must be divisible by num_heads ({num_heads})")

        keys = jax.random.split(key, 6)
        key_w, key_att_sender, key_att_receiver, key_w_edge, key_att_edge, key_b = keys
        if isinstance(in_dim, tuple):
            key_w_sender, key_w_receiver = jax.random.split(key_w)
            self.w_sender = Param(w_init(shape=(in_src, out_dim), key=key_w_sender))
            self.w_receiver = Param(w_init(shape=(in_dst, out_dim), key=key_w_receiver))
        else:
            self.w_sender = Param(w_init(shape=(in_src, out_dim), key=key_w))
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
        x_src = (x_src @ self.w_sender).reshape(n_src, self.num_heads, -1)
        w_receiver = self.w_sender if self.w_receiver is None else self.w_receiver
        x_dst = (x_dst @ w_receiver).reshape(n_dst, self.num_heads, -1)

        # Compute attention scores at node level, then combine at edges
        logits_sender = (x_src * self.att_sender).sum(axis=-1)
        logits_receiver = (x_dst * self.att_receiver).sum(axis=-1)
        logits = logits_sender[senders] + logits_receiver[receivers]

        # Add edge feature contribution
        if x_edge is not None:
            assert self.w_edge is not None
            assert self.att_edge is not None
            if edge_mask is not None:
                x_edge = jnp.where(edge_mask[:, None], x_edge, 0.0)
            x_edge = (x_edge @ self.w_edge).reshape(-1, self.num_heads, x_src.shape[-1])
            logits = logits + (x_edge * self.att_edge).sum(axis=-1)

        logits = jax.nn.leaky_relu(logits, self.negative_slope)

        if edge_mask is not None:
            logits = jnp.where(edge_mask[:, None], logits, -jnp.inf)

        # Softmax over each receiver's incoming edges
        attention = segment_softmax(logits, receivers, n_dst)

        # Aggregate sender features weighted by attention
        messages = x_src[senders] * attention[..., None]
        aggregated = segment_sum(messages, receivers, n_dst)
        x_out = aggregated.reshape(n_dst, -1)

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

        in_src, in_dst = in_dim if isinstance(in_dim, tuple) else (in_dim, in_dim)
        head_dim = out_dim // num_heads

        if out_dim % num_heads != 0:
            raise ValueError(f"out_dim ({out_dim}) must be divisible by num_heads ({num_heads})")

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
        x_src = (x_src @ self.w_sender).reshape(n_src, self.num_heads, -1)
        x_dst = (x_dst @ self.w_receiver).reshape(n_dst, self.num_heads, -1)

        # Sum of the two projections is the paper's concatenation, factored
        edge_h = x_src[senders] + x_dst[receivers]

        # Add edge feature contribution before the nonlinearity
        if x_edge is not None:
            assert self.w_edge is not None
            if edge_mask is not None:
                x_edge = jnp.where(edge_mask[:, None], x_edge, 0.0)
            x_edge = (x_edge @ self.w_edge).reshape(-1, self.num_heads, x_src.shape[-1])
            edge_h = edge_h + x_edge

        # Apply nonlinearity then dot with attention vector (GATv2 difference)
        edge_h = jax.nn.leaky_relu(edge_h, self.negative_slope)
        logits = (edge_h * self.att).sum(axis=-1)

        if edge_mask is not None:
            logits = jnp.where(edge_mask[:, None], logits, -jnp.inf)

        # Softmax over each receiver's incoming edges
        attention = segment_softmax(logits, receivers, n_dst)

        # Aggregate sender features weighted by attention
        messages = x_src[senders] * attention[..., None]
        aggregated = segment_sum(messages, receivers, n_dst)
        x_out = aggregated.reshape(n_dst, -1)

        if self.b_out is not None:
            x_out = x_out + self.b_out

        return x_out


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

        in_src, in_dst = in_dim if isinstance(in_dim, tuple) else (in_dim, in_dim)
        root_shape, edge_shape = (in_dst, out_dim), (edge_dim or 0, out_dim)
        use_edge_features = edge_dim is not None

        if out_dim % num_heads != 0:
            raise ValueError(f"out_dim ({out_dim}) must be divisible by num_heads ({num_heads})")
        if use_beta and not use_root_weight:
            raise ValueError("use_beta=True requires use_root_weight=True")

        key_q, key_k, key_v, key_root, key_edge, key_beta, key_b = jax.random.split(key, 7)
        self.w_q = Param(w_init(shape=(in_dst, out_dim), key=key_q))
        self.w_k = Param(w_init(shape=(in_src, out_dim), key=key_k))
        self.w_v = Param(w_init(shape=(in_src, out_dim), key=key_v))
        self.w_root = Param(w_init(shape=root_shape, key=key_root)) if use_root_weight else None
        self.w_edge = Param(w_init(shape=edge_shape, key=key_edge)) if use_edge_features else None
        self.w_beta = Param(w_init(shape=(3 * out_dim, 1), key=key_beta)) if use_beta else None
        self.b_out = Param(b_init(shape=(out_dim,), key=key_b)) if use_bias else None

        self.num_heads = num_heads
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
            x_edge = (x_edge @ self.w_edge).reshape(-1, self.num_heads, head_dim)
            edge_k = edge_k + x_edge
            edge_v = edge_v + x_edge

        # Scaled dot product between each receiver's query and its senders' keys
        logits = (q[receivers] * edge_k).sum(axis=-1) / math.sqrt(head_dim)

        if edge_mask is not None:
            logits = jnp.where(edge_mask[:, None], logits, -jnp.inf)

        # Softmax over each receiver's incoming edges
        attention = segment_softmax(logits, receivers, n_dst)

        # Aggregate sender values weighted by attention
        messages = edge_v * attention[..., None]
        aggregated = segment_sum(messages, receivers, n_dst).reshape(n_dst, -1)
        x_out = aggregated

        # Include the receiving node through its learned root projection
        if self.w_root is not None:
            root = x_dst @ self.w_root
            if self.w_beta is not None:
                gate_in = jnp.concatenate([root, x_out, root - x_out], axis=-1)
                gate = jax.nn.sigmoid(gate_in @ self.w_beta)
                x_out = gate * root + (1 - gate) * x_out
            else:
                x_out = x_out + root

        if self.b_out is not None:
            x_out = x_out + self.b_out

        return x_out
