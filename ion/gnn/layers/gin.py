"""Graph Isomorphism Network layers.

Modules:
    GINConv   Sum-aggregation graph convolution with an MLP update.  (Xu et al., 2019)
    GINEConv  Sum-aggregation with edge features in the messages.    (Hu et al., 2020)

The update network is passed by the caller; the layers create no weights of their
own. Self-loops are not needed: own features enter via the (1 + eps) term.
Edge features are added to sender features, so they share the node dimension.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from ...nn.module import Module
from ...nn.param import Param
from ..ops import segment_sum


class GINConv(Module):
    """Graph isomorphism layer.

    >>> gin = GINConv(MLP([16, 32, 32], key=key))
    >>> gin(x, senders, receivers)  # (n, 16) -> (n, 32)
    """

    mlp: Module
    eps: Param[Float[Array, ""]] | float

    def __init__(
        self,
        mlp: Module,
        *,
        eps: float = 0.0,
        train_eps: bool = False,
    ) -> None:

        self.mlp = mlp
        self.eps = Param(jnp.array(eps)) if train_eps else eps

    def __call__(
        self,
        x: Float[Array, "n i"],
        senders: Int[Array, " e"],
        receivers: Int[Array, " e"],
    ) -> Float[Array, "n o"]:

        n, i = x.shape

        # Sum aggregation preserves neighbor multiplicity
        agg = segment_sum(x[senders], receivers, n)

        return self.mlp((1 + self.eps) * x + agg)


class GINEConv(Module):
    """Graph isomorphism layer with edge features.

    >>> gine = GINEConv(MLP([16, 32, 32], key=key))
    >>> gine(x, senders, receivers, x_edge=x_edge)  # (n, 16) -> (n, 32)
    """

    mlp: Module
    eps: Param[Float[Array, ""]] | float

    def __init__(
        self,
        mlp: Module,
        *,
        eps: float = 0.0,
        train_eps: bool = False,
    ) -> None:

        self.mlp = mlp
        self.eps = Param(jnp.array(eps)) if train_eps else eps

    def __call__(
        self,
        x: Float[Array, "n i"],
        senders: Int[Array, " e"],
        receivers: Int[Array, " e"],
        *,
        x_edge: Float[Array, "e i"],
    ) -> Float[Array, "n o"]:

        n, i = x.shape

        # Each edge adds its features to the sender before the message nonlinearity
        agg = segment_sum(jax.nn.relu(x[senders] + x_edge), receivers, n)

        return self.mlp((1 + self.eps) * x + agg)
