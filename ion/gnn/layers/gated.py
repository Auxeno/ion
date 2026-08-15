"""Gated graph convolution from Bresson and Laurent, 2017.

Modules:
    GatedGCNConv  Joint node and edge update with feature-wise edge gates.

Glorot uniform weight init, zeros for bias. The layer applies no activation,
normalization, or residual connection.
"""

import jax
from jax.nn.initializers import Initializer, glorot_uniform, zeros
from jaxtyping import Array, Float, Int, PRNGKeyArray

from ...nn.module import Module
from ...nn.param import Param
from ..ops import segment_sum


class GatedGCNConv(Module):
    """Joint node and edge update with feature-wise edge gates.

    >>> conv = GatedGCNConv(16, 32, edge_dim=8, key=key)
    >>> x, x_edge = conv(x, senders, receivers, x_edge=x_edge)
    """

    w_root: Param[Float[Array, "j o"]]
    w_neigh: Param[Float[Array, "i o"]]
    w_edge: Param[Float[Array, "f o"]]
    w_sender: Param[Float[Array, "i o"]]
    w_receiver: Param[Float[Array, "j o"]]
    b_node: Param[Float[Array, " o"]] | None
    b_edge: Param[Float[Array, " o"]] | None
    eps: float

    def __init__(
        self,
        in_dim: int | tuple[int, int],
        out_dim: int,
        edge_dim: int,
        *,
        eps: float = 1e-6,
        use_bias: bool = True,
        w_init: Initializer = glorot_uniform(),
        b_init: Initializer = zeros,
        key: PRNGKeyArray,
    ) -> None:

        in_src, in_dst = in_dim if isinstance(in_dim, tuple) else (in_dim, in_dim)

        keys = jax.random.split(key, 7)
        self.w_root = Param(w_init(shape=(in_dst, out_dim), key=keys[0]))
        self.w_neigh = Param(w_init(shape=(in_src, out_dim), key=keys[1]))
        self.w_edge = Param(w_init(shape=(edge_dim, out_dim), key=keys[2]))
        self.w_sender = Param(w_init(shape=(in_src, out_dim), key=keys[3]))
        self.w_receiver = Param(w_init(shape=(in_dst, out_dim), key=keys[4]))
        self.b_node = Param(b_init(shape=(out_dim,), key=keys[5])) if use_bias else None
        self.b_edge = Param(b_init(shape=(out_dim,), key=keys[6])) if use_bias else None
        self.eps = eps

    def __call__(
        self,
        x: Float[Array, "n i"] | tuple[Float[Array, "s i"], Float[Array, "t j"]],
        senders: Int[Array, " e"],
        receivers: Int[Array, " e"],
        *,
        x_edge: Float[Array, "e f"],
    ) -> tuple[Float[Array, "t o"], Float[Array, "e o"]]:

        x_src, x_dst = x if isinstance(x, tuple) else (x, x)
        n_dst = x_dst.shape[0]

        # Update each edge from its previous features and incident nodes
        x_edge_out = (
            x_edge @ self.w_edge
            + x_src[senders] @ self.w_sender
            + x_dst[receivers] @ self.w_receiver
        )
        if self.b_edge is not None:
            x_edge_out = x_edge_out + self.b_edge

        # Normalize feature-wise gates over each receiver's incoming edges
        gates = jax.nn.sigmoid(x_edge_out)
        gate_sum = segment_sum(gates, receivers, n_dst)

        # Gate transformed sender features, then add the central-node transform
        messages = gates * (x_src[senders] @ self.w_neigh)
        aggregated = segment_sum(messages, receivers, n_dst) / (gate_sum + self.eps)
        x_out = x_dst @ self.w_root + aggregated

        if self.b_node is not None:
            x_out = x_out + self.b_node

        return x_out, x_edge_out
