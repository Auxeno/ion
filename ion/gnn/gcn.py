"""Graph Convolutional Network layer from Kipf & Welling, 2017.

Modules:
    GCNConv  Graph convolutional layer with symmetric degree normalization.

He normal weight init for ReLU activation, zeros for bias.
Self-loops are the caller's responsibility, see `gnn.add_self_loops`.
"""

import jax
import jax.numpy as jnp
from jax import lax
from jax.nn.initializers import Initializer
from jaxtyping import Array, Float, Int, PRNGKeyArray

from ..nn.module import Module
from ..nn.param import Param


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
        bias: bool = True,
        dtype: jnp.dtype = jnp.float32,
        w_init: Initializer = jax.nn.initializers.he_normal(),
        b_init: Initializer = jax.nn.initializers.zeros,
        *,
        key: PRNGKeyArray,
    ) -> None:

        key_w, key_b = jax.random.split(key)
        self.w = Param(w_init(shape=(in_dim, out_dim), dtype=dtype, key=key_w))
        self.b = Param(b_init(shape=(out_dim,), dtype=dtype, key=key_b)) if bias else None

    def __call__(
        self,
        x: Float[Array, "n i"],
        senders: Int[Array, " e"],
        receivers: Int[Array, " e"],
    ) -> Float[Array, "n o"]:

        n, i = x.shape
        (e,) = senders.shape

        x = x @ self.w

        # Compute node degrees by counting incoming edges at each receiver
        edge_counts = jnp.ones(e, dtype=x.dtype)
        degree = jax.ops.segment_sum(edge_counts, receivers, n)

        # Compute symmetric normalization coefficients to stabilize hub activations
        node_norm = jnp.where(degree > 0, lax.rsqrt(degree), 0.0)
        edge_weight = node_norm[senders] * node_norm[receivers]

        # Route, scale, and accumulate features from senders to receivers
        messages = x[senders] * edge_weight[:, None]
        x = jax.ops.segment_sum(messages, receivers, n)

        if self.b is not None:
            x = x + self.b

        return x
