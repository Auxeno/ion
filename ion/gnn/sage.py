"""GraphSAGE layer from Hamilton et al., 2017.

Modules:
    SAGEConv  Sample-and-aggregate graph convolution with a separate root term.

Glorot uniform weight init, zeros for bias.
Neighbour aggregation is `mean`, `max`, or `sum`; the central node enters through
the root weight, so self-loops are not needed.
"""

from collections.abc import Callable
from typing import Literal

import jax
import jax.numpy as jnp
from jax.nn.initializers import Initializer, glorot_uniform, zeros
from jaxtyping import Array, Float, Int, PRNGKeyArray

from ..nn.module import Module
from ..nn.param import Param
from . import segment_max, segment_mean, segment_sum


class SAGEConv(Module):
    """GraphSAGE layer.

    >>> sage = SAGEConv(16, 32, key=key)
    >>> sage(x, senders, receivers)  # (n, 16) -> (n, 32)
    >>> sage = SAGEConv(16, 32, aggregator="max", normalize=True, key=key)
    """

    w_neigh: Param[Float[Array, "i o"]]
    w_self: Param[Float[Array, "i o"]] | None
    b: Param[Float[Array, " o"]] | None
    aggregate: Callable[..., Float[Array, "n o"]]
    normalize: bool

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        aggregator: Literal["mean", "max", "sum"] = "mean",
        normalize: bool = False,
        root_weight: bool = True,
        bias: bool = True,
        w_init: Initializer = glorot_uniform(),
        b_init: Initializer = zeros,
        *,
        key: PRNGKeyArray,
    ) -> None:

        aggregators = {"mean": segment_mean, "max": segment_max, "sum": segment_sum}
        if aggregator not in aggregators:
            raise ValueError(f"aggregator must be one of {tuple(aggregators)}, got {aggregator!r}")

        key_neigh, key_self, key_b = jax.random.split(key, 3)
        self.w_neigh = Param(w_init(shape=(in_dim, out_dim), key=key_neigh))
        self.w_self = Param(w_init(shape=(in_dim, out_dim), key=key_self)) if root_weight else None
        self.b = Param(b_init(shape=(out_dim,), key=key_b)) if bias else None

        self.aggregate = aggregators[aggregator]
        self.normalize = normalize

    def __call__(
        self,
        x: Float[Array, "n i"],
        senders: Int[Array, " e"],
        receivers: Int[Array, " e"],
    ) -> Float[Array, "n o"]:

        n, i = x.shape

        # Pool sender features into each receiver's neighborhood, then transform
        neigh = self.aggregate(x[senders], receivers, n)

        # segment_max leaves -inf at nodes with no neighbors
        if self.aggregate is segment_max:
            neigh = jnp.where(jnp.isneginf(neigh), 0.0, neigh)
        x_out = neigh @ self.w_neigh

        # Add the central node's own features through the root weight
        if self.w_self is not None:
            x_out = x_out + x @ self.w_self

        if self.b is not None:
            x_out = x_out + self.b

        # L2 normalize each node embedding
        if self.normalize:
            x_out = x_out / jnp.maximum(jnp.linalg.norm(x_out, axis=-1, keepdims=True), 1e-12)

        return x_out
