"""Relational graph convolutional layers.

Modules:
    RGCNConv  Per-relation neighbour transforms.            (Schlichtkrull et al., 2018)
    HGTConv   Type-dependent attention over a typed graph.  (Hu et al., 2020)

Glorot uniform weight init, zeros for bias, ones for HGTConv's relation prior and skip gate.
Relation types are a per-edge index array and HGTConv node types a per-node one, so one
node feature matrix and one edge list carry the whole heterogeneous graph.
RGCNConv takes an optional basis decomposition sharing `num_bases` matrices across relations.
HGTConv attention is normalized over all of a node's incoming edges, whatever their relation.
Self-loops are not needed: RGCNConv has a separate root term and HGTConv a skip gate.
"""

import math

import jax
import jax.numpy as jnp
from jax.nn.initializers import Initializer, glorot_uniform, zeros
from jaxtyping import Array, Float, Int, PRNGKeyArray

from ...nn.module import Module
from ...nn.param import Param
from ..ops import degree, segment_softmax, segment_sum


class RGCNConv(Module):
    """Relational graph convolutional layer.

    >>> conv = RGCNConv(16, 32, 6, key=key)
    >>> conv(x, senders, receivers, edge_type=edge_type)  # (n, 16) -> (n, 32)
    >>> conv = RGCNConv(16, 32, 6, num_bases=2, key=key)  # share two matrices
    """

    w_neigh: Param[Float[Array, "b i o"]]
    w_coeff: Param[Float[Array, "r b"]] | None
    w_root: Param[Float[Array, "i o"]]
    b: Param[Float[Array, " o"]] | None
    num_edge_types: int

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_edge_types: int,
        *,
        num_bases: int | None = None,
        use_bias: bool = True,
        w_init: Initializer = glorot_uniform(),
        b_init: Initializer = zeros,
        key: PRNGKeyArray,
    ) -> None:

        neigh_shape = (num_bases or num_edge_types, in_dim, out_dim)
        coeff_shape = (num_edge_types, num_bases or 0)

        key_neigh, key_coeff, key_root, key_b = jax.random.split(key, 4)
        self.w_neigh = Param(w_init(shape=neigh_shape, key=key_neigh))
        self.w_coeff = Param(w_init(shape=coeff_shape, key=key_coeff)) if num_bases else None
        self.w_root = Param(w_init(shape=(in_dim, out_dim), key=key_root))
        self.b = Param(b_init(shape=(out_dim,), key=key_b)) if use_bias else None

        self.num_edge_types = num_edge_types

    def __call__(
        self,
        x: Float[Array, "n i"],
        senders: Int[Array, " e"],
        receivers: Int[Array, " e"],
        *,
        edge_type: Int[Array, " e"],
    ) -> Float[Array, "n o"]:

        n = x.shape[0]

        # Each relation's transform is a learned mixture of the shared bases
        w_neigh = self.w_neigh
        if self.w_coeff is not None:
            w_neigh = jnp.einsum("rb,bio->rio", self.w_coeff, w_neigh)

        # Project every node under every relation, then take the one each edge needs
        messages = jnp.einsum("ni,rio->nro", x, w_neigh)[senders, edge_type]

        # Normalize by how many edges of the same relation reach the receiver
        segment_ids = receivers * self.num_edge_types + edge_type
        counts = degree(segment_ids, n * self.num_edge_types).astype(x.dtype)
        messages = messages / counts[segment_ids][:, None]

        aggregated = segment_sum(messages, receivers, n)
        x_out = aggregated + x @ self.w_root

        if self.b is not None:
            x_out = x_out + self.b

        return x_out


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
        num_edge_types: int,
        *,
        num_heads: int = 1,
        use_skip: bool = True,
        use_bias: bool = True,
        w_init: Initializer = glorot_uniform(),
        b_init: Initializer = zeros,
        key: PRNGKeyArray,
    ) -> None:

        head_dim = out_dim // num_heads
        relation_shape = (num_edge_types, num_heads, head_dim, head_dim)

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
        self.mu = Param(jnp.ones((num_edge_types, num_heads)))
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

        messages = edge_v * attention[..., None]
        aggregated = segment_sum(messages, receivers, n).reshape(n, -1)
        x_out = jnp.einsum("no,tof->ntf", jax.nn.gelu(aggregated), self.w_out)[nodes]

        if self.b_out is not None:
            x_out = x_out + self.b_out[node_type]

        # Learned per-type gate between the new representation and the input
        if self.skip is not None:
            gate = jax.nn.sigmoid(self.skip[node_type])[:, None]
            x_out = gate * x_out + (1 - gate) * x

        return x_out
