"""Graph convolutional layers.

Modules:
    GCNConv    Symmetric degree-normalized convolution.  (Kipf & Welling, 2017)
    GraphConv  Separate neighbour and root transforms.   (Morris et al., 2019)
    SAGEConv   Sample-and-aggregate with a root term.    (Hamilton et al., 2017)

Glorot uniform weight init, zeros for bias.
GCNConv self-loops are the caller's responsibility, see `gnn.add_self_loops`.
GraphConv and SAGEConv do not need self-loops because they have a separate root term.
SAGEConv neighbour aggregation is `mean`, `max`, or `sum`.
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

        n = x.shape[0]

        x = x @ self.w

        # Count incoming edges at each receiver
        in_degree = degree(receivers, n).astype(x.dtype)

        # Compute symmetric normalization coefficients to stabilize hub activations
        node_norm = jnp.where(in_degree > 0, lax.rsqrt(in_degree), 0.0)
        edge_weight = node_norm[senders] * node_norm[receivers]

        # Route, scale, and accumulate features from senders to receivers
        messages = x[senders] * edge_weight[:, None]
        aggregated = segment_sum(messages, receivers, n)
        x_out = aggregated

        if self.b is not None:
            x_out = x_out + self.b

        return x_out


class GraphConv(Module):
    """Graph convolutional layer with separate root and neighbour transforms.

    >>> conv = GraphConv(16, 32, key=key)
    >>> conv(x, senders, receivers)  # (n, 16) -> (n, 32)
    >>> conv(x, senders, receivers, edge_weight=edge_weight)
    """

    w_neigh: Param[Float[Array, "i o"]]
    w_root: Param[Float[Array, "j o"]]
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

        key_neigh, key_root, key_b = jax.random.split(key, 3)
        self.w_neigh = Param(w_init(shape=(in_src, out_dim), key=key_neigh))
        self.w_root = Param(w_init(shape=(in_dst, out_dim), key=key_root))
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
        aggregated = segment_sum(messages, receivers, n_dst)

        # Transform neighbourhood and root features independently
        x_out = aggregated @ self.w_neigh + x_dst @ self.w_root

        if self.b is not None:
            x_out = x_out + self.b

        return x_out


class SAGEConv(Module):
    """GraphSAGE layer.

    >>> sage = SAGEConv(16, 32, key=key)
    >>> sage(x, senders, receivers)  # (n, 16) -> (n, 32)
    >>> sage = SAGEConv(16, 32, aggregate="max", normalize=True, key=key)
    """

    w_neigh: Param[Float[Array, "i o"]]
    w_root: Param[Float[Array, "j o"]] | None
    b: Param[Float[Array, " o"]] | None
    aggregate: Callable[..., Float[Array, "n i"]]
    normalize: bool

    def __init__(
        self,
        in_dim: int | tuple[int, int],
        out_dim: int,
        *,
        aggregate: Literal["mean", "max", "sum"] = "mean",
        normalize: bool = False,
        use_root_weight: bool = True,
        use_bias: bool = True,
        w_init: Initializer = glorot_uniform(),
        b_init: Initializer = zeros,
        key: PRNGKeyArray,
    ) -> None:

        in_src, in_dst = in_dim if isinstance(in_dim, tuple) else (in_dim, in_dim)
        root_shape = (in_dst, out_dim)

        key_neigh, key_root, key_b = jax.random.split(key, 3)
        self.w_neigh = Param(w_init(shape=(in_src, out_dim), key=key_neigh))
        self.w_root = Param(w_init(shape=root_shape, key=key_root)) if use_root_weight else None
        self.b = Param(b_init(shape=(out_dim,), key=key_b)) if use_bias else None

        aggregates = {"mean": segment_mean, "max": segment_max, "sum": segment_sum}
        self.aggregate = aggregates[aggregate]
        self.normalize = normalize

    def __call__(
        self,
        x: Float[Array, "n i"] | tuple[Float[Array, "s i"], Float[Array, "t j"]],
        senders: Int[Array, " e"],
        receivers: Int[Array, " e"],
    ) -> Float[Array, "t o"]:

        x_src, x_dst = x if isinstance(x, tuple) else (x, x)
        n_dst = x_dst.shape[0]

        messages = x_src[senders]
        aggregated = self.aggregate(messages, receivers, n_dst)

        # segment_max leaves -inf at nodes with no neighbours
        if self.aggregate is segment_max:
            aggregated = jnp.where(jnp.isneginf(aggregated), 0.0, aggregated)
        x_out = aggregated @ self.w_neigh

        if self.w_root is not None:
            x_out = x_out + x_dst @ self.w_root

        if self.b is not None:
            x_out = x_out + self.b

        # L2 normalize each node embedding
        if self.normalize:
            x_out = x_out / jnp.maximum(jnp.linalg.norm(x_out, axis=-1, keepdims=True), 1e-12)

        return x_out
