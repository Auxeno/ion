"""Graph topology operations."""

from typing import Literal

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

__all__ = [
    "add_self_loops",
    "coalesce",
    "degree",
    "from_adjacency",
    "induced_subgraph",
    "k_hop_subgraph",
    "line_graph",
    "remove_self_loops",
    "to_adjacency",
    "to_undirected",
]


def add_self_loops(
    senders: Int[Array, " e"],
    receivers: Int[Array, " e"],
    num_nodes: int,
) -> tuple[Int[Array, " e2"], Int[Array, " e2"]]:
    """Append self-loop edges (i -> i) for every node.

    >>> senders, receivers = add_self_loops(senders, receivers, num_nodes)
    """
    self_indices = jnp.arange(num_nodes)
    senders = jnp.concatenate([senders, self_indices])
    receivers = jnp.concatenate([receivers, self_indices])
    return senders, receivers


def remove_self_loops(
    senders: Int[Array, " e"],
    receivers: Int[Array, " e"],
) -> tuple[Int[Array, " e2"], Int[Array, " e2"], Int[Array, " e2"]]:
    """Remove self-loop edges (i -> i).

    >>> senders, receivers, kept = remove_self_loops(senders, receivers)
    """
    kept = jnp.flatnonzero(senders != receivers)
    return senders[kept], receivers[kept], kept


def degree(
    indices: Int[Array, " e"],
    num_nodes: int,
) -> Int[Array, " n"]:
    """Count how many edges reference each node.

    >>> out_degree = degree(senders, num_nodes)
    >>> in_degree = degree(receivers, num_nodes)
    """
    return jnp.bincount(indices, length=num_nodes)


def coalesce(
    senders: Int[Array, " e"],
    receivers: Int[Array, " e"],
) -> tuple[Int[Array, " e2"], Int[Array, " e2"], Int[Array, " e2"]]:
    """Sort edges by (sender, receiver) and drop duplicates.

    >>> senders, receivers, kept = coalesce(senders, receivers)
    >>> x_edge = x_edge[kept]
    """
    edges = jnp.stack([senders, receivers], axis=1)
    _, kept = jnp.unique(edges, axis=0, return_index=True)
    return senders[kept], receivers[kept], kept


def to_undirected(
    senders: Int[Array, " e"],
    receivers: Int[Array, " e"],
) -> tuple[Int[Array, " e2"], Int[Array, " e2"], Int[Array, " e2"]]:
    """Add the reverse of every edge, then coalesce.

    >>> senders, receivers, kept = to_undirected(senders, receivers)
    >>> x_edge = jnp.concatenate([x_edge, x_edge])[kept]
    """
    return coalesce(
        jnp.concatenate([senders, receivers]),
        jnp.concatenate([receivers, senders]),
    )


def line_graph(
    senders: Int[Array, " e"],
    receivers: Int[Array, " e"],
    *,
    non_backtracking: bool = True,
) -> tuple[Int[Array, " l"], Int[Array, " l"], Int[Array, " l"]]:
    """Rebuild the graph with edges as nodes, joined head to tail.

    >>> line_senders, line_receivers, shared = line_graph(senders, receivers)
    """
    # Edges leaving a node form one contiguous block of the sender-sorted ordering
    sender_order = jnp.argsort(senders, stable=True)
    sorted_senders = senders[sender_order]

    # Every edge i -> v takes the whole block at v, one slot per successor
    successor_starts = jnp.searchsorted(sorted_senders, receivers)
    successor_ends = jnp.searchsorted(sorted_senders, receivers, side="right")
    successor_counts = successor_ends - successor_starts
    line_senders = jnp.repeat(jnp.arange(senders.shape[0]), successor_counts)
    successor_offsets = successor_starts - jnp.cumsum(successor_counts) + successor_counts
    successor_positions = successor_offsets[line_senders] + jnp.arange(line_senders.shape[0])
    line_receivers = sender_order[successor_positions]

    # Drop i -> v -> i, which sends each edge straight back down its own reverse
    if non_backtracking:
        keep = senders[line_senders] != receivers[line_receivers]
        line_senders, line_receivers = line_senders[keep], line_receivers[keep]

    return line_senders, line_receivers, receivers[line_senders]


def induced_subgraph(
    senders: Int[Array, " e"] | np.ndarray,
    receivers: Int[Array, " e"] | np.ndarray,
    node_ids: Int[Array, " k"] | np.ndarray,
    num_nodes: int,
) -> tuple[
    Int[Array, " e2"],
    Int[Array, " e2"],
    Int[Array, " k"],
    Int[Array, " e2"],
]:
    """Build the node-induced subgraph over a selected set of nodes.

    >>> senders, receivers, node_ids, edge_ids = induced_subgraph(senders, receivers, selected, n)
    """
    senders = np.asarray(senders)
    receivers = np.asarray(receivers)
    node_ids = np.asarray(node_ids)

    if np.unique(node_ids).shape[0] != node_ids.shape[0]:
        raise ValueError("node_ids must not contain duplicates")

    relabel = np.full(num_nodes, -1, dtype=senders.dtype)
    relabel[node_ids] = np.arange(node_ids.shape[0], dtype=senders.dtype)

    keep = (relabel[senders] >= 0) & (relabel[receivers] >= 0)
    edge_ids = np.flatnonzero(keep)

    return (
        jnp.asarray(relabel[senders[edge_ids]]),
        jnp.asarray(relabel[receivers[edge_ids]]),
        jnp.asarray(node_ids),
        jnp.asarray(edge_ids),
    )


def k_hop_subgraph(
    senders: Int[Array, " e"] | np.ndarray,
    receivers: Int[Array, " e"] | np.ndarray,
    node_ids: Int[Array, " s"] | np.ndarray,
    num_hops: int,
    num_nodes: int,
    *,
    direction: Literal["in", "out", "both"] = "in",
) -> tuple[
    Int[Array, " e2"],
    Int[Array, " e2"],
    Int[Array, " k"],
    Int[Array, " e2"],
]:
    """Build the node-induced subgraph within a number of hops of selected nodes.

    >>> senders, receivers, node_ids, edge_ids = k_hop_subgraph(senders, receivers, selected, 2, n)
    """
    if num_hops < 0:
        raise ValueError("num_hops must be non-negative")
    if direction not in ("in", "out", "both"):
        raise ValueError("direction must be 'in', 'out', or 'both'")

    senders = np.asarray(senders)
    receivers = np.asarray(receivers)
    node_ids = np.asarray(node_ids)

    if np.unique(node_ids).shape[0] != node_ids.shape[0]:
        raise ValueError("node_ids must not contain duplicates")

    seen = np.zeros(num_nodes, dtype=bool)
    seen[node_ids] = True
    frontier = seen.copy()
    nodes_by_hop = [node_ids]

    for _ in range(num_hops):
        candidates = []
        if direction in ("in", "both"):
            candidates.append(senders[frontier[receivers]])
        if direction in ("out", "both"):
            candidates.append(receivers[frontier[senders]])

        new_node_ids = np.unique(np.concatenate(candidates))
        new_node_ids = new_node_ids[~seen[new_node_ids]]
        if new_node_ids.shape[0] == 0:
            break

        seen[new_node_ids] = True
        frontier = np.zeros(num_nodes, dtype=bool)
        frontier[new_node_ids] = True
        nodes_by_hop.append(new_node_ids)

    return induced_subgraph(senders, receivers, np.concatenate(nodes_by_hop), num_nodes)


def to_adjacency(
    senders: Int[Array, " e"],
    receivers: Int[Array, " e"],
    num_nodes: int,
) -> Float[Array, " n n"]:
    """Scatter edges into a dense adjacency matrix.

    >>> adjacency = to_adjacency(senders, receivers, num_nodes)
    """
    return jnp.zeros((num_nodes, num_nodes)).at[senders, receivers].set(1.0)


def from_adjacency(
    adjacency: Float[Array, " n n"],
    edge_capacity: int | None = None,
) -> tuple[Int[Array, " e"], Int[Array, " e"]]:
    """Gather the nonzero entries of a dense adjacency matrix into edges.

    >>> senders, receivers = from_adjacency(adjacency)
    >>> senders, receivers = from_adjacency(adjacency, edge_capacity=42)  # jit friendly
    """
    senders, receivers = jnp.nonzero(adjacency, size=edge_capacity, fill_value=adjacency.shape[0])
    return senders, receivers
