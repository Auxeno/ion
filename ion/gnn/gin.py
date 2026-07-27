"""Graph Isomorphism Network layer from Xu et al., 2019.

Modules:
    GINConv  Sum-aggregation graph convolution with an MLP update.

The update network is passed by the caller; the layer creates no weights of its
own. Self-loops are not needed: own features enter via the (1 + eps) term.
"""

import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from ..nn.module import Module
from ..nn.param import Param
from .ops import segment_sum


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
