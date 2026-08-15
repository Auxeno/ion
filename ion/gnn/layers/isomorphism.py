"""Graph Isomorphism Network layers.

Modules:
    GINConv   Sum-aggregation graph convolution with an MLP update.  (Xu et al., 2019)
    GINEConv  Sum-aggregation with edge features in the messages.    (Hu et al., 2020)

The update network is passed by the caller; the layers create no weights of their
own. Self-loops are not needed: own features enter via the (1 + eps) term.
Edge features are added to sender features, so they share the node dimension.
"""

from collections.abc import Callable

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

    mlp: Callable[[Array], Array]
    eps: Param[Float[Array, ""]] | float

    def __init__(
        self,
        mlp: Callable[[Array], Array],
        *,
        eps: float = 0.0,
        train_eps: bool = False,
    ) -> None:

        self.mlp = mlp
        self.eps = Param(jnp.array(eps)) if train_eps else eps

    def __call__(
        self,
        x: Float[Array, "n i"] | tuple[Float[Array, "s i"], Float[Array, "t i"]],
        senders: Int[Array, " e"],
        receivers: Int[Array, " e"],
    ) -> Float[Array, "t o"]:

        x_src, x_dst = x if isinstance(x, tuple) else (x, x)
        n_dst = x_dst.shape[0]

        # Sum aggregation preserves neighbour multiplicity
        messages = x_src[senders]
        aggregated = segment_sum(messages, receivers, n_dst)

        x_out = (1 + self.eps) * x_dst + aggregated

        return self.mlp(x_out)


class GINEConv(Module):
    """Graph isomorphism layer with edge features.

    >>> gine = GINEConv(MLP([16, 32, 32], key=key))
    >>> gine(x, senders, receivers, x_edge=x_edge)  # (n, 16) -> (n, 32)
    """

    mlp: Callable[[Array], Array]
    eps: Param[Float[Array, ""]] | float

    def __init__(
        self,
        mlp: Callable[[Array], Array],
        *,
        eps: float = 0.0,
        train_eps: bool = False,
    ) -> None:

        self.mlp = mlp
        self.eps = Param(jnp.array(eps)) if train_eps else eps

    def __call__(
        self,
        x: Float[Array, "n i"] | tuple[Float[Array, "s i"], Float[Array, "t i"]],
        senders: Int[Array, " e"],
        receivers: Int[Array, " e"],
        *,
        x_edge: Float[Array, "e i"],
    ) -> Float[Array, "t o"]:

        x_src, x_dst = x if isinstance(x, tuple) else (x, x)
        n_dst = x_dst.shape[0]

        # Each edge adds its features to the sender before the message nonlinearity
        messages = jax.nn.relu(x_src[senders] + x_edge)
        aggregated = segment_sum(messages, receivers, n_dst)

        x_out = (1 + self.eps) * x_dst + aggregated

        return self.mlp(x_out)
