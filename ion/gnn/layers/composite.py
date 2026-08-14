"""Composable Graph Network layers from Battaglia et al., 2018.

Modules:
    GraphNetwork  Caller-supplied edge and node updates with message aggregation.
    EdgeUpdate    Edge update with a caller-supplied network.
    NodeUpdate    Node update from aggregated edges with a caller-supplied network.

The layers create no weights of their own and apply no activation,
normalization, or residual connection.
"""

from collections.abc import Callable

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from ...nn.module import Module
from ..ops import segment_sum


class GraphNetwork(Module):
    """Update edges, aggregate them, then update nodes.

    >>> network = GraphNetwork(edge_model=edge_model, node_model=node_model)
    >>> x, x_edge = network(x, senders, receivers, x_edge=x_edge)
    """

    edge_model: Callable[[Array], Array]
    node_model: Callable[[Array], Array]
    aggregate: Callable[[Array, Int[Array, " e"], int], Array]

    def __init__(
        self,
        *,
        edge_model: Callable[[Array], Array],
        node_model: Callable[[Array], Array],
        aggregate: Callable[[Array, Int[Array, " e"], int], Array] = segment_sum,
    ) -> None:

        self.edge_model = edge_model
        self.node_model = node_model
        self.aggregate = aggregate

    def __call__(
        self,
        x: Float[Array, "n i"] | tuple[Float[Array, "s i"], Float[Array, "t j"]],
        senders: Int[Array, " e"],
        receivers: Int[Array, " e"],
        *,
        x_edge: Float[Array, "e f"] | None = None,
        edge_mask: Bool[Array, " e"] | None = None,
    ) -> tuple[Float[Array, "t o"], Float[Array, "e m"]]:

        x_src, x_dst = x if isinstance(x, tuple) else (x, x)
        n_dst = x_dst.shape[0]

        # Update each edge from its previous features and incident nodes
        edge_inputs = [x_src[senders], x_dst[receivers]]
        if x_edge is not None:
            edge_inputs.append(x_edge)
        edge_out = self.edge_model(jnp.concatenate(edge_inputs, axis=-1))

        # Aggregate updated edges at receivers, excluding masked edges
        if edge_mask is not None:
            receivers = jnp.where(edge_mask, receivers, n_dst)

        received = self.aggregate(edge_out, receivers, n_dst)
        node_inputs = jnp.concatenate((x_dst, received), axis=-1)
        node_out = self.node_model(node_inputs)

        return node_out, edge_out


class EdgeUpdate(Module):
    """Update edge features from their incident nodes.

    >>> update = EdgeUpdate(MLP([40, 32, 16], key=key))
    >>> update(x, senders, receivers, x_edge=x_edge)  # (e, 8) -> (e, 16)
    """

    edge_model: Callable[[Array], Array]

    def __init__(self, edge_model: Callable[[Array], Array]) -> None:

        self.edge_model = edge_model

    def __call__(
        self,
        x: Float[Array, "n i"] | tuple[Float[Array, "s i"], Float[Array, "t j"]],
        senders: Int[Array, " e"],
        receivers: Int[Array, " e"],
        *,
        x_edge: Float[Array, "e f"] | None = None,
    ) -> Float[Array, "e o"]:

        x_src, x_dst = x if isinstance(x, tuple) else (x, x)

        edge_inputs = [x_src[senders], x_dst[receivers]]
        if x_edge is not None:
            edge_inputs.append(x_edge)

        return self.edge_model(jnp.concatenate(edge_inputs, axis=-1))


class NodeUpdate(Module):
    """Update node features from aggregated edges.

    >>> update = NodeUpdate(MLP([24, 32, 16], key=key))
    >>> update(x, senders, receivers, x_edge=x_edge)  # (n, 16) -> (n, 16)
    """

    node_model: Callable[[Array], Array]
    aggregate: Callable[[Array, Int[Array, " e"], int], Array]

    def __init__(
        self,
        node_model: Callable[[Array], Array],
        *,
        aggregate: Callable[[Array, Int[Array, " e"], int], Array] = segment_sum,
    ) -> None:

        self.node_model = node_model
        self.aggregate = aggregate

    def __call__(
        self,
        x: Float[Array, "n i"] | tuple[Float[Array, "s i"], Float[Array, "t j"]],
        senders: Int[Array, " e"],
        receivers: Int[Array, " e"],
        *,
        x_edge: Float[Array, "e f"],
        edge_mask: Bool[Array, " e"] | None = None,
    ) -> Float[Array, "t o"]:

        x_dst = x[1] if isinstance(x, tuple) else x
        n_dst = x_dst.shape[0]

        # Aggregate edge features at receivers, excluding masked edges
        if edge_mask is not None:
            receivers = jnp.where(edge_mask, receivers, n_dst)

        received = self.aggregate(x_edge, receivers, n_dst)
        node_inputs = jnp.concatenate((x_dst, received), axis=-1)
        return self.node_model(node_inputs)
