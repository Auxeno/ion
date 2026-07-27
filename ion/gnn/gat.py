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

    w: Param[Float[Array, "i h k"]]
    att_sender: Param[Float[Array, "h k"]]
    att_receiver: Param[Float[Array, "h k"]]
    b: Param[Float[Array, " o"]] | None
    w_edge: Param[Float[Array, "f h k"]] | None
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

        key_w, key_att_s, key_att_r, key_b, key_w_e, key_att_e = jax.random.split(key, 6)
        head_dim = out_dim // num_heads

        # Initialize projections flat so Glorot fans are (in_dim, out_dim), then split heads
        w = w_init(shape=(in_dim, out_dim), key=key_w)
        self.w = Param(w.reshape(in_dim, num_heads, head_dim))
        self.att_sender = Param(att_init(shape=(num_heads, head_dim), key=key_att_s))
        self.att_receiver = Param(att_init(shape=(num_heads, head_dim), key=key_att_r))
        self.b = Param(b_init(shape=(out_dim,), key=key_b)) if bias else None

        if edge_dim is not None:
            w_edge = w_init(shape=(edge_dim, out_dim), key=key_w_e)
            self.w_edge = Param(w_edge.reshape(edge_dim, num_heads, head_dim))
            self.att_edge = Param(att_init(shape=(num_heads, head_dim), key=key_att_e))
        else:
            self.w_edge = None
            self.att_edge = None

        self.num_heads = num_heads
        self.negative_slope = negative_slope
        self.edge_dim = edge_dim

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

        n, i = x.shape

        # Project input features into multi-head space
        x = jnp.einsum("ni, ihk -> nhk", x, self.w)

        # Compute attention scores at node level, then combine at edges
        logits_sender = jnp.einsum("nhk, hk -> nh", x, self.att_sender)
        logits_receiver = jnp.einsum("nhk, hk -> nh", x, self.att_receiver)
        logits = logits_sender[senders] + logits_receiver[receivers]

        # Add edge feature contribution to attention logits
        if x_edge is not None:
            if edge_mask is not None:
                x_edge = x_edge * edge_mask[:, None]
            edge_proj = jnp.einsum("ef, fhk -> ehk", x_edge, self.w_edge)
            logits_edge = jnp.einsum("ehk, hk -> eh", edge_proj, self.att_edge)
            logits = logits + logits_edge

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

    w_sender: Param[Float[Array, "i h k"]]
    w_receiver: Param[Float[Array, "i h k"]]
    att: Param[Float[Array, "h k"]]
    b: Param[Float[Array, " o"]] | None
    w_edge: Param[Float[Array, "f h k"]] | None
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

        key_w_s, key_w_r, key_att, key_b, key_w_e = jax.random.split(key, 5)
        head_dim = out_dim // num_heads

        # Initialize projections flat so Glorot fans are (in_dim, out_dim), then split heads
        w_sender = w_init(shape=(in_dim, out_dim), key=key_w_s)
        self.w_sender = Param(w_sender.reshape(in_dim, num_heads, head_dim))
        w_receiver = w_init(shape=(in_dim, out_dim), key=key_w_r)
        self.w_receiver = Param(w_receiver.reshape(in_dim, num_heads, head_dim))
        self.att = Param(att_init(shape=(num_heads, head_dim), key=key_att))
        self.b = Param(b_init(shape=(out_dim,), key=key_b)) if bias else None

        if edge_dim is not None:
            w_edge = w_init(shape=(edge_dim, out_dim), key=key_w_e)
            self.w_edge = Param(w_edge.reshape(edge_dim, num_heads, head_dim))
        else:
            self.w_edge = None

        self.num_heads = num_heads
        self.negative_slope = negative_slope
        self.edge_dim = edge_dim

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

        n, i = x.shape

        # Project with separate sender/receiver weights
        x_s = jnp.einsum("ni, ihk -> nhk", x, self.w_sender)
        x_r = jnp.einsum("ni, ihk -> nhk", x, self.w_receiver)

        # Combine at edge level
        edge_h = x_s[senders] + x_r[receivers]

        # Edge features go inside the LeakyReLU (unlike GATv1)
        if x_edge is not None:
            if edge_mask is not None:
                x_edge = x_edge * edge_mask[:, None]
            edge_proj = jnp.einsum("ef, fhk -> ehk", x_edge, self.w_edge)
            edge_h = edge_h + edge_proj

        # Apply nonlinearity then dot with attention vector (GATv2 difference)
        logits = jnp.einsum(
            "ehk, hk -> eh", jax.nn.leaky_relu(edge_h, self.negative_slope), self.att
        )

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
