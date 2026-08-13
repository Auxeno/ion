"""Heterogeneous graph transformer layers.

Modules:
    HGTConv  Type-dependent attention over a typed graph.  (Hu et al., 2020)

Glorot uniform weight init, zeros for bias, ones for the relation prior and skip gate.
Node types are a per-node index array and relation types a per-edge one, so one
node feature matrix and one edge list carry the whole heterogeneous graph.
Attention is normalized over all of a node's incoming edges, whatever their relation.
"""

import math

import jax
import jax.numpy as jnp
from jax.nn.initializers import Initializer, glorot_uniform, zeros
from jaxtyping import Array, Float, Int, PRNGKeyArray

from ...nn.module import Module
from ...nn.param import Param
from ..ops import segment_softmax, segment_sum


class HGTConv(Module):
    """Heterogeneous graph transformer layer.

    >>> conv = HGTConv(32, 32, 3, 6, num_heads=4, key=key)
    >>> conv(x, senders, receivers, node_type=node_type, edge_type=edge_type)
    """

    w_q: Param[Float[Array, "c i o"]]
    w_k: Param[Float[Array, "c i o"]]
    w_v: Param[Float[Array, "c i o"]]
    w_att: Param[Float[Array, "r h k k"]]
    w_msg: Param[Float[Array, "r h k k"]]
    w_out: Param[Float[Array, "c o o"]]
    mu: Param[Float[Array, "r h"]]
    skip: Param[Float[Array, " c"]] | None
    b_out: Param[Float[Array, "c o"]] | None
    num_heads: int

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_node_types: int,
        num_relations: int,
        *,
        num_heads: int = 1,
        use_skip: bool = True,
        use_bias: bool = True,
        w_init: Initializer = glorot_uniform(),
        b_init: Initializer = zeros,
        key: PRNGKeyArray,
    ) -> None:

        head_dim = out_dim // num_heads
        relation_shape = (num_relations, num_heads, head_dim, head_dim)

        if out_dim % num_heads != 0:
            raise ValueError(f"out_dim ({out_dim}) must be divisible by num_heads ({num_heads})")
        if use_skip and in_dim != out_dim:
            raise ValueError(f"use_skip=True requires in_dim ({in_dim}) to equal out_dim")

        keys = jax.random.split(key, 7)
        key_q, key_k, key_v, key_att, key_msg, key_out, key_b = keys
        self.w_q = Param(w_init(shape=(num_node_types, in_dim, out_dim), key=key_q))
        self.w_k = Param(w_init(shape=(num_node_types, in_dim, out_dim), key=key_k))
        self.w_v = Param(w_init(shape=(num_node_types, in_dim, out_dim), key=key_v))
        self.w_att = Param(w_init(shape=relation_shape, key=key_att))
        self.w_msg = Param(w_init(shape=relation_shape, key=key_msg))
        self.w_out = Param(w_init(shape=(num_node_types, out_dim, out_dim), key=key_out))
        self.mu = Param(jnp.ones((num_relations, num_heads)))
        self.skip = Param(jnp.ones((num_node_types,))) if use_skip else None
        self.b_out = Param(b_init(shape=(num_node_types, out_dim), key=key_b)) if use_bias else None

        self.num_heads = num_heads

    def __call__(
        self,
        x: Float[Array, "n i"],
        senders: Int[Array, " e"],
        receivers: Int[Array, " e"],
        *,
        node_type: Int[Array, " n"],
        edge_type: Int[Array, " e"],
    ) -> Float[Array, "n o"]:

        n = x.shape[0]

        # Project every node under every type, then take the one its own type gives
        nodes = (jnp.arange(n), node_type)
        q = jnp.einsum("ni,tio->nto", x, self.w_q)[nodes].reshape(n, self.num_heads, -1)
        k = jnp.einsum("ni,tio->nto", x, self.w_k)[nodes].reshape(n, self.num_heads, -1)
        v = jnp.einsum("ni,tio->nto", x, self.w_v)[nodes].reshape(n, self.num_heads, -1)
        head_dim = q.shape[-1]

        # Each relation reweights the key and message spaces its edges send through
        edge_k = jnp.einsum("nhd,rhdf->nrhf", k, self.w_att)[senders, edge_type]
        edge_v = jnp.einsum("nhd,rhdf->nrhf", v, self.w_msg)[senders, edge_type]

        # Scaled dot product, scaled again by a learned prior on the relation
        logits = (q[receivers] * edge_k).sum(axis=-1) * self.mu[edge_type] / math.sqrt(head_dim)

        # Softmax over each receiver's incoming edges, across every relation at once
        attention = segment_softmax(logits, receivers, n)

        # Aggregate messages, then project under the receiving node's own type
        agg = segment_sum(edge_v * attention[..., None], receivers, n).reshape(n, -1)
        out = jnp.einsum("no,tof->ntf", jax.nn.gelu(agg), self.w_out)[nodes]

        if self.b_out is not None:
            out = out + self.b_out[node_type]

        # Learned per-type gate between the new representation and the input
        if self.skip is not None:
            gate = jax.nn.sigmoid(self.skip[node_type])[:, None]
            out = gate * out + (1 - gate) * x

        return out
