"""Graph convolutional layers.

Modules:
    GCNConv    Symmetric degree-normalized convolution.  (Kipf & Welling, 2017)
    GraphConv  Separate neighbor and root transforms.    (Morris et al., 2019)
    SAGEConv   Sample-and-aggregate with a root term.    (Hamilton et al., 2017)

Glorot uniform weight init, zeros for bias.
GCNConv self-loops are the caller's responsibility, see `gnn.add_self_loops`.
GraphConv and SAGEConv do not need self-loops because they have a separate root term.
SAGEConv neighbor aggregation is `mean`, `max`, or `sum`.
"""

from collections.abc import Callable
from typing import Literal

import jax
import jax.numpy as jnp
from jax import lax
from jax.nn.initializers import Initializer, glorot_uniform, zeros
from jaxtyping import Array, Float, Int, PRNGKeyArray

from ...nn.module import Module
from ...nn.param import Param
from ..ops import degree, segment_max, segment_mean, segment_sum


class GCNConv(Module):
    """Graph convolutional layer.

    >>> gcn = GCNConv(16, 32, key=key)
    >>> gcn(x, senders, receivers)  # (n, 16) -> (n, 32)
    """

    w: Param[Float[Array, "i o"]]
    b: Param[Float[Array, " o"]] | None

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        use_bias: bool = True,
        w_init: Initializer = glorot_uniform(),
        b_init: Initializer = zeros,
        key: PRNGKeyArray,
    ) -> None:

        key_w, key_b = jax.random.split(key)
        self.w = Param(w_init(shape=(in_dim, out_dim), key=key_w))
        self.b = Param(b_init(shape=(out_dim,), key=key_b)) if use_bias else None

    def __call__(
        self,
        x: Float[Array, "n i"],
        senders: Int[Array, " e"],
        receivers: Int[Array, " e"],
    ) -> Float[Array, "n o"]:

        n, i = x.shape

        x = x @ self.w

        # Count incoming edges at each receiver
        in_degree = degree(receivers, n).astype(x.dtype)

        # Compute symmetric normalization coefficients to stabilize hub activations
        node_norm = jnp.where(in_degree > 0, lax.rsqrt(in_degree), 0.0)
        edge_weight = node_norm[senders] * node_norm[receivers]

        # Route, scale, and accumulate features from senders to receivers
        messages = x[senders] * edge_weight[:, None]
        x = segment_sum(messages, receivers, n)

        if self.b is not None:
            x = x + self.b

        return x


class GraphConv(Module):
    """Graph convolutional layer with separate root and neighbor transforms.

    >>> conv = GraphConv(16, 32, key=key)
    >>> conv(x, senders, receivers)  # (n, 16) -> (n, 32)
    >>> conv(x, senders, receivers, edge_weight=edge_weight)
    """

    w_neigh: Param[Float[Array, "i o"]]
    w_self: Param[Float[Array, "j o"]]
    b: Param[Float[Array, " o"]] | None

    def __init__(
        self,
        in_dim: int | tuple[int, int],
        out_dim: int,
        *,
        use_bias: bool = True,
        w_init: Initializer = glorot_uniform(),
        b_init: Initializer = zeros,
        key: PRNGKeyArray,
    ) -> None:

        in_src, in_dst = in_dim if isinstance(in_dim, tuple) else (in_dim, in_dim)

        key_neigh, key_self, key_b = jax.random.split(key, 3)
        self.w_neigh = Param(w_init(shape=(in_src, out_dim), key=key_neigh))
        self.w_self = Param(w_init(shape=(in_dst, out_dim), key=key_self))
        self.b = Param(b_init(shape=(out_dim,), key=key_b)) if use_bias else None

    def __call__(
        self,
        x: Float[Array, "n i"] | tuple[Float[Array, "s i"], Float[Array, "t j"]],
        senders: Int[Array, " e"],
        receivers: Int[Array, " e"],
        *,
        edge_weight: Float[Array, " e"] | None = None,
    ) -> Float[Array, "t o"]:

        x_src, x_dst = x if isinstance(x, tuple) else (x, x)
        n_dst = x_dst.shape[0]

        # Optionally weight sender features, then sum them into each receiver
        messages = x_src[senders]
        if edge_weight is not None:
            messages = messages * edge_weight[:, None]
        neigh = segment_sum(messages, receivers, n_dst)

        # Transform neighborhood and central-node features independently
        x_out = neigh @ self.w_neigh + x_dst @ self.w_self

        if self.b is not None:
            x_out = x_out + self.b

        return x_out


class SAGEConv(Module):
    """GraphSAGE layer.

    >>> sage = SAGEConv(16, 32, key=key)
    >>> sage(x, senders, receivers)  # (n, 16) -> (n, 32)
    >>> sage = SAGEConv(16, 32, aggregator="max", normalize=True, key=key)
    """

    w_neigh: Param[Float[Array, "i o"]]
    w_self: Param[Float[Array, "j o"]] | None
    b: Param[Float[Array, " o"]] | None
    aggregate: Callable[..., Float[Array, "n i"]]
    normalize: bool

    def __init__(
        self,
        in_dim: int | tuple[int, int],
        out_dim: int,
        *,
        aggregator: Literal["mean", "max", "sum"] = "mean",
        normalize: bool = False,
        use_root_weight: bool = True,
        use_bias: bool = True,
        w_init: Initializer = glorot_uniform(),
        b_init: Initializer = zeros,
        key: PRNGKeyArray,
    ) -> None:

        aggregate = {"mean": segment_mean, "max": segment_max, "sum": segment_sum}[aggregator]
        in_src, in_dst = in_dim if isinstance(in_dim, tuple) else (in_dim, in_dim)

        key_neigh, key_self, key_b = jax.random.split(key, 3)
        self.w_neigh = Param(w_init(shape=(in_src, out_dim), key=key_neigh))
        self.w_self = (
            Param(w_init(shape=(in_dst, out_dim), key=key_self)) if use_root_weight else None
        )
        self.b = Param(b_init(shape=(out_dim,), key=key_b)) if use_bias else None

        self.aggregate = aggregate
        self.normalize = normalize

    def __call__(
        self,
        x: Float[Array, "n i"] | tuple[Float[Array, "s i"], Float[Array, "t j"]],
        senders: Int[Array, " e"],
        receivers: Int[Array, " e"],
    ) -> Float[Array, "t o"]:

        x_src, x_dst = x if isinstance(x, tuple) else (x, x)
        n_dst = x_dst.shape[0]

        # Pool sender features into each receiver's neighborhood, then transform
        neigh = self.aggregate(x_src[senders], receivers, n_dst)

        # segment_max leaves -inf at nodes with no neighbors
        if self.aggregate is segment_max:
            neigh = jnp.where(jnp.isneginf(neigh), 0.0, neigh)
        x_out = neigh @ self.w_neigh

        # Add the central node's own features through the root weight
        if self.w_self is not None:
            x_out = x_out + x_dst @ self.w_self

        if self.b is not None:
            x_out = x_out + self.b

        # L2 normalize each node embedding
        if self.normalize:
            x_out = x_out / jnp.maximum(jnp.linalg.norm(x_out, axis=-1, keepdims=True), 1e-12)

        return x_out
