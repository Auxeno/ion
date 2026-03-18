"""Graph Attention Network layer from Velickovic et al., 2018.

Modules:
    GATConv  Multi-head graph attention layer.

Glorot uniform weight init to match original paper, zeros for bias.
Self-loops are the caller's responsibility, see `gnn.add_self_loops`.
"""

import jax
import jax.numpy as jnp
from jax.nn.initializers import Initializer
from jaxtyping import Array, Float, Int, PRNGKeyArray

from ..nn.module import Module
from ..nn.param import Param
from .ops import segment_softmax


class GATConv(Module):
    """Multi-head graph attention layer.

    >>> gat = GATConv(16, 32, num_heads=4, key=key)
    >>> gat(x, senders, receivers)  # (n, 16) -> (n, 32)
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
        dtype: jnp.dtype = jnp.float32,
        w_init: Initializer = jax.nn.initializers.glorot_uniform(),
        att_init: Initializer = jax.nn.initializers.glorot_uniform(),
        b_init: Initializer = jax.nn.initializers.zeros,
        *,
        key: PRNGKeyArray,
    ) -> None:

        if out_dim % num_heads != 0:
            raise ValueError(f"out_dim ({out_dim}) must be divisible by num_heads ({num_heads})")

        key_w, key_att_s, key_att_r, key_b, key_w_e, key_att_e = jax.random.split(key, 6)
        head_dim = out_dim // num_heads

        self.w = Param(w_init(shape=(in_dim, num_heads, head_dim), dtype=dtype, key=key_w))
        self.att_sender = Param(att_init(shape=(num_heads, head_dim), dtype=dtype, key=key_att_s))
        self.att_receiver = Param(att_init(shape=(num_heads, head_dim), dtype=dtype, key=key_att_r))
        self.b = Param(b_init(shape=(out_dim,), dtype=dtype, key=key_b)) if bias else None

        if edge_dim is not None:
            self.w_edge = Param(
                w_init(shape=(edge_dim, num_heads, head_dim), dtype=dtype, key=key_w_e)
            )
            self.att_edge = Param(att_init(shape=(num_heads, head_dim), dtype=dtype, key=key_att_e))
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
        x_edge: Float[Array, "e f"] | None = None,
    ) -> Float[Array, "n o"]:

        num_nodes = x.shape[0]

        # Project input features into multi-head space
        x = jnp.einsum("ni, ihk -> nhk", x, self.w)

        # Compute attention scores at node level, then combine at edges
        logits_sender = jnp.einsum("nhk, hk -> nh", x, self.att_sender)
        logits_receiver = jnp.einsum("nhk, hk -> nh", x, self.att_receiver)
        logits = logits_sender[senders] + logits_receiver[receivers]

        # Add edge feature contribution to attention logits
        if x_edge is not None:
            edge_proj = jnp.einsum("ef, fhk -> ehk", x_edge, self.w_edge)
            logits_edge = jnp.einsum("ehk, hk -> eh", edge_proj, self.att_edge)
            logits = logits + logits_edge

        logits = jax.nn.leaky_relu(logits, self.negative_slope)

        # Normalize attention weights per receiver neighborhood
        attention = segment_softmax(logits, receivers, num_nodes)

        # Aggregate sender features weighted by attention
        messages = x[senders] * attention[..., None]
        x = jax.ops.segment_sum(messages, receivers, num_nodes)

        # Concatenate heads into a flat feature vector
        x = x.reshape(num_nodes, -1)

        if self.b is not None:
            x = x + self.b

        return x
