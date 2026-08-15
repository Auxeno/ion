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
    sizes = [x.shape[0] for x in xs]

    edge_offsets = np.repeat(np.cumsum([0] + sizes[:-1]), [s.shape[0] for s in senders])
    return (
        jnp.asarray(np.concatenate(xs)),
        jnp.asarray(np.concatenate(senders) + edge_offsets),
        jnp.asarray(np.concatenate(receivers) + edge_offsets),
        jnp.asarray(np.repeat(np.arange(len(xs)), sizes)),
    )


def pad_graphs(
    x: Float[Array, "n d"] | np.ndarray,
    senders: Int[Array, " e"] | np.ndarray,
    receivers: Int[Array, " e"] | np.ndarray,
    graph_ids: Int[Array, " n"] | np.ndarray,
    num_nodes: int,
    num_edges: int,
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
    pad_nodes = num_nodes - x.shape[0]
    pad_edges = num_edges - senders.shape[0]

    return (
        jnp.asarray(np.pad(x, ((0, pad_nodes), (0, 0)))),
        jnp.asarray(np.pad(senders, (0, pad_edges), constant_values=num_nodes)),
        jnp.asarray(np.pad(receivers, (0, pad_edges), constant_values=num_nodes)),
        jnp.asarray(np.pad(graph_ids, (0, pad_nodes), constant_values=num_graphs)),
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

    sizes = np.bincount(graph_ids)
    offsets = np.cumsum(sizes) - sizes
    edge_ids = graph_ids[senders]
    return (
        [jnp.asarray(x[graph_ids == i]) for i in range(sizes.shape[0])],
        [jnp.asarray(senders[edge_ids == i] - o) for i, o in enumerate(offsets)],
        [jnp.asarray(receivers[edge_ids == i] - o) for i, o in enumerate(offsets)],
    )
