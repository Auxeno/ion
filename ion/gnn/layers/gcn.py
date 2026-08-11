"""Graph convolutional layers.

Modules:
    GCNConv    Symmetric degree-normalized convolution.  (Kipf & Welling, 2017)
    GraphConv  Separate neighbor and root transforms.    (Morris et al., 2019)

Glorot uniform weight init, zeros for bias.
GCNConv self-loops are the caller's responsibility, see `gnn.add_self_loops`.
GraphConv does not need self-loops because it has a separate root term.
"""

import jax
import jax.numpy as jnp
from jax import lax
from jax.nn.initializers import Initializer, glorot_uniform, zeros
from jaxtyping import Array, Float, Int, PRNGKeyArray

from ...nn.module import Module
from ...nn.param import Param
from ..ops import degree, segment_sum


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
    w_self: Param[Float[Array, "i o"]]
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

        key_neigh, key_self, key_b = jax.random.split(key, 3)
        self.w_neigh = Param(w_init(shape=(in_dim, out_dim), key=key_neigh))
        self.w_self = Param(w_init(shape=(in_dim, out_dim), key=key_self))
        self.b = Param(b_init(shape=(out_dim,), key=key_b)) if use_bias else None

    def __call__(
        self,
        x: Float[Array, "n i"],
        senders: Int[Array, " e"],
        receivers: Int[Array, " e"],
        *,
        edge_weight: Float[Array, " e"] | None = None,
    ) -> Float[Array, "n o"]:

        n, i = x.shape

        # Optionally weight sender features, then sum them into each receiver
        messages = x[senders]
        if edge_weight is not None:
            messages = messages * edge_weight[:, None]
        neigh = segment_sum(messages, receivers, n)

        # Transform neighborhood and central-node features independently
        x_out = neigh @ self.w_neigh + x @ self.w_self

        if self.b is not None:
            x_out = x_out + self.b

        return x_out
