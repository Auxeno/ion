"""Utilities for batching graphs."""

from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

__all__ = ["batch_graphs", "pad_graphs", "unbatch_graphs"]


def batch_graphs(
    xs: Sequence[Float[Array, "n d"] | np.ndarray],
    senders: Sequence[Int[Array, " e"] | np.ndarray],
    receivers: Sequence[Int[Array, " e"] | np.ndarray],
) -> tuple[
    Float[Array, "n d"],
    Int[Array, " e"],
    Int[Array, " e"],
    Int[Array, " n"],
]:
    """Pack graphs into one disconnected graph for batched message passing.

    >>> x, senders, receivers, graph_ids = batch_graphs(xs, senders_list, receivers_list)
    """
    node_counts = [x.shape[0] for x in xs]
    edge_counts = [sender_ids.shape[0] for sender_ids in senders]

    node_offsets = np.repeat(np.cumsum([0] + node_counts[:-1]), edge_counts)
    return (
        jnp.asarray(np.concatenate(xs)),
        jnp.asarray(np.concatenate(senders) + node_offsets),
        jnp.asarray(np.concatenate(receivers) + node_offsets),
        jnp.asarray(np.repeat(np.arange(len(xs)), node_counts)),
    )


def pad_graphs(
    x: Float[Array, "n d"] | np.ndarray,
    senders: Int[Array, " e"] | np.ndarray,
    receivers: Int[Array, " e"] | np.ndarray,
    graph_ids: Int[Array, " n"] | np.ndarray,
    node_capacity: int,
    edge_capacity: int,
    num_graphs: int,
) -> tuple[
    Float[Array, "n2 d"],
    Int[Array, " e2"],
    Int[Array, " e2"],
    Int[Array, " n2"],
]:
    """Pad a batched graph to fixed node, edge, and graph capacity.

    >>> x, senders, receivers, graph_ids = pad_graphs(x, s, r, graph_ids, 512, 2048, 32)
    """
    node_padding = node_capacity - x.shape[0]
    edge_padding = edge_capacity - senders.shape[0]

    return (
        jnp.asarray(np.pad(x, ((0, node_padding), (0, 0)))),
        jnp.asarray(np.pad(senders, (0, edge_padding), constant_values=node_capacity)),
        jnp.asarray(np.pad(receivers, (0, edge_padding), constant_values=node_capacity)),
        jnp.asarray(np.pad(graph_ids, (0, node_padding), constant_values=num_graphs)),
    )


def unbatch_graphs(
    x: Float[Array, "n d"] | np.ndarray,
    senders: Int[Array, " e"] | np.ndarray,
    receivers: Int[Array, " e"] | np.ndarray,
    graph_ids: Int[Array, " n"] | np.ndarray,
) -> tuple[
    list[Float[Array, "n d"]],
    list[Int[Array, " e"]],
    list[Int[Array, " e"]],
]:
    """Split a batched graph back into its component graphs.

    >>> xs, senders_list, receivers_list = unbatch_graphs(x, senders, receivers, graph_ids)
    """
    x = np.asarray(x)
    senders = np.asarray(senders)
    receivers = np.asarray(receivers)
    graph_ids = np.asarray(graph_ids)

    node_counts = np.bincount(graph_ids)
    node_offsets = np.cumsum(node_counts) - node_counts
    edge_graph_ids = graph_ids[senders]
    num_graphs = node_counts.shape[0]
    return (
        [jnp.asarray(x[graph_ids == graph_id]) for graph_id in range(num_graphs)],
        [
            jnp.asarray(senders[edge_graph_ids == graph_id] - node_offset)
            for graph_id, node_offset in enumerate(node_offsets)
        ],
        [
            jnp.asarray(receivers[edge_graph_ids == graph_id] - node_offset)
            for graph_id, node_offset in enumerate(node_offsets)
        ],
    )
