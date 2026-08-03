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

from ..nn.module import Module
from ..nn.param import Param
from .ops import segment_softmax, segment_sum


class GATConv(Module):
    """Multi-head graph attention layer.

    >>> gat = GATConv(16, 32, num_heads=4, key=key)
    >>> gat(x, senders, receivers)  # (n, 16) -> (n, 32)
    >>> gat(x, senders, receivers, edge_mask=mask)  # mask: bool (e,)
    """

    w: Param[Float[Array, "i o"]]
    att_sender: Param[Float[Array, "h k"]]
    att_receiver: Param[Float[Array, "h k"]]
    b: Param[Float[Array, " o"]] | None
    w_edge: Param[Float[Array, "f o"]] | None
    att_edge: Param[Float[Array, "h k"]] | None
    num_heads: int
    negative_slope: float
    edge_dim: int | None

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 1,
        edge_dim: int | None = None,
        negative_slope: float = 0.2,
        bias: bool = True,
        w_init: Initializer = glorot_uniform(),
        att_init: Initializer = glorot_uniform(),
        b_init: Initializer = zeros,
        *,
        key: PRNGKeyArray,
    ) -> None:

        if out_dim % num_heads != 0:
            raise ValueError(f"out_dim ({out_dim}) must be divisible by num_heads ({num_heads})")

        head_dim = out_dim // num_heads

        self.num_heads = num_heads
        self.negative_slope = negative_slope
        self.edge_dim = edge_dim

        key_w, key_att_s, key_att_r, key_b, key_w_e, key_att_e = jax.random.split(key, 6)
        self.w = Param(w_init(shape=(in_dim, out_dim), key=key_w))
        self.att_sender = Param(att_init(shape=(num_heads, head_dim), key=key_att_s))
        self.att_receiver = Param(att_init(shape=(num_heads, head_dim), key=key_att_r))
        self.b = Param(b_init(shape=(out_dim,), key=key_b)) if bias else None

        if edge_dim is not None:
            self.w_edge = Param(w_init(shape=(edge_dim, out_dim), key=key_w_e))
            self.att_edge = Param(att_init(shape=(num_heads, head_dim), key=key_att_e))
        else:
            self.w_edge = None
            self.att_edge = None

    def __call__(
        self,
        x: Float[Array, "n i"],
        senders: Int[Array, " e"],
        receivers: Int[Array, " e"],
        *,
        x_edge: Float[Array, "e f"] | None = None,
        edge_mask: Bool[Array, " e"] | None = None,
    ) -> Float[Array, "n o"]:

        # Guard silent edge-feature mismatches between construction and call
        if x_edge is None and self.edge_dim is not None:
            raise ValueError(f"edge_dim={self.edge_dim} set at init but no x_edge passed at call")
        if x_edge is not None and self.edge_dim is None:
            raise ValueError("x_edge passed at call but edge_dim not set at init")

        n = x.shape[0]

        # Project nodes, then split the output dimension into attention heads
        x = (x @ self.w).reshape(n, self.num_heads, -1)

        # Compute attention scores at node level, then combine at edges
        logits_sender = (x * self.att_sender).sum(axis=-1)
        logits_receiver = (x * self.att_receiver).sum(axis=-1)
        logits = logits_sender[senders] + logits_receiver[receivers]

        # Add edge feature contribution to attention logits
        if x_edge is not None:
            assert self.w_edge is not None
            assert self.att_edge is not None
            if edge_mask is not None:
                x_edge = x_edge * edge_mask[:, None]
            edge = (x_edge @ self.w_edge).reshape(-1, self.num_heads, x.shape[-1])
            logits = logits + (edge * self.att_edge).sum(axis=-1)

        logits = jax.nn.leaky_relu(logits, self.negative_slope)

        # Mask out edges so they receive zero attention weight
        if edge_mask is not None:
            logits = jnp.where(edge_mask[:, None], logits, -jnp.inf)

        # Normalize attention weights per receiver neighborhood
        attention = segment_softmax(logits, receivers, n)

        # Aggregate sender features weighted by attention
        messages = x[senders] * attention[..., None]
        x = segment_sum(messages, receivers, n)

        # Concatenate heads into a flat feature vector
        x = x.reshape(n, -1)

        if self.b is not None:
            x = x + self.b

        return x


class GATv2Conv(Module):
    """Multi-head dynamic graph attention layer.

    >>> gat = GATv2Conv(16, 32, num_heads=4, key=key)
    >>> gat(x, senders, receivers)  # (n, 16) -> (n, 32)
    >>> gat(x, senders, receivers, edge_mask=mask)  # mask: bool (e,)
    """

    w_sender: Param[Float[Array, "i o"]]
    w_receiver: Param[Float[Array, "i o"]]
    att: Param[Float[Array, "h k"]]
    b: Param[Float[Array, " o"]] | None
    w_edge: Param[Float[Array, "f o"]] | None
    num_heads: int
    negative_slope: float
    edge_dim: int | None

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 1,
        edge_dim: int | None = None,
        negative_slope: float = 0.2,
        bias: bool = True,
        w_init: Initializer = glorot_uniform(),
        att_init: Initializer = glorot_uniform(),
        b_init: Initializer = zeros,
        *,
        key: PRNGKeyArray,
    ) -> None:

        if out_dim % num_heads != 0:
            raise ValueError(f"out_dim ({out_dim}) must be divisible by num_heads ({num_heads})")

        head_dim = out_dim // num_heads

        self.num_heads = num_heads
        self.negative_slope = negative_slope
        self.edge_dim = edge_dim

        key_w_s, key_w_r, key_att, key_b, key_w_e = jax.random.split(key, 5)
        self.w_sender = Param(w_init(shape=(in_dim, out_dim), key=key_w_s))
        self.w_receiver = Param(w_init(shape=(in_dim, out_dim), key=key_w_r))
        self.att = Param(att_init(shape=(num_heads, head_dim), key=key_att))
        self.b = Param(b_init(shape=(out_dim,), key=key_b)) if bias else None

        if edge_dim is not None:
            self.w_edge = Param(w_init(shape=(edge_dim, out_dim), key=key_w_e))
        else:
            self.w_edge = None

    def __call__(
        self,
        x: Float[Array, "n i"],
        senders: Int[Array, " e"],
        receivers: Int[Array, " e"],
        *,
        x_edge: Float[Array, "e f"] | None = None,
        edge_mask: Bool[Array, " e"] | None = None,
    ) -> Float[Array, "n o"]:

        # Guard silent edge-feature mismatches between construction and call
        if x_edge is None and self.edge_dim is not None:
            raise ValueError(f"edge_dim={self.edge_dim} set at init but no x_edge passed at call")
        if x_edge is not None and self.edge_dim is None:
            raise ValueError("x_edge passed at call but edge_dim not set at init")

        n = x.shape[0]

        # Project nodes, then split the output dimension into attention heads
        x_s = (x @ self.w_sender).reshape(n, self.num_heads, -1)
        x_r = (x @ self.w_receiver).reshape(n, self.num_heads, -1)

        # Combine at edge level
        edge_h = x_s[senders] + x_r[receivers]

        # Edge features go inside the LeakyReLU (unlike GATv1)
        if x_edge is not None:
            assert self.w_edge is not None
            if edge_mask is not None:
                x_edge = x_edge * edge_mask[:, None]
            edge = (x_edge @ self.w_edge).reshape(-1, self.num_heads, x_s.shape[-1])
            edge_h = edge_h + edge

        # Apply nonlinearity then dot with attention vector (GATv2 difference)
        edge_h = jax.nn.leaky_relu(edge_h, self.negative_slope)
        logits = (edge_h * self.att).sum(axis=-1)

        # Mask out edges so they receive zero attention weight
        if edge_mask is not None:
            logits = jnp.where(edge_mask[:, None], logits, -jnp.inf)

        # Normalize attention weights per receiver neighborhood
        attention = segment_softmax(logits, receivers, n)

        # Aggregate sender features weighted by attention
        messages = x_s[senders] * attention[..., None]
        x = segment_sum(messages, receivers, n)

        # Concatenate heads into a flat feature vector
        x = x.reshape(n, -1)

        if self.b is not None:
            x = x + self.b

        return x
