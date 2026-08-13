"""Edge update layer from Battaglia et al., 2018.

Modules:
    EdgeUpdate  Edge update with a caller-supplied network.

EdgeUpdate creates no weights of its own and applies no activation,
normalization, or residual connection.
"""

import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from ...nn.module import Module


class EdgeUpdate(Module):
    """Update edge features from their incident nodes.

    >>> update = EdgeUpdate(MLP([40, 32, 16], key=key))
    >>> update(x, senders, receivers, x_edge=x_edge)  # (e, 8) -> (e, 16)
    """

    edge_model: Module

    def __init__(self, edge_model: Module) -> None:

        self.edge_model = edge_model

    def __call__(
        self,
        x: Float[Array, "n i"] | tuple[Float[Array, "s i"], Float[Array, "t j"]],
        senders: Int[Array, " e"],
        receivers: Int[Array, " e"],
        *,
        x_edge: Float[Array, "e f"],
    ) -> Float[Array, "e o"]:

        x_src, x_dst = x if isinstance(x, tuple) else (x, x)

        edge_inputs = jnp.concatenate((x_src[senders], x_dst[receivers], x_edge), axis=-1)
        return self.edge_model(edge_inputs)
