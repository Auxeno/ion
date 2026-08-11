"""Graph topology operations."""

import jax.numpy as jnp
from jaxtyping import Array, Int

__all__ = [
    "add_self_loops",
    "coalesce",
    "degree",
    "line_graph",
    "remove_self_loops",
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
) -> tuple[Int[Array, " e2"], Int[Array, " e2"]]:
    """Remove self-loop edges (i -> i).

    >>> senders, receivers = remove_self_loops(senders, receivers)
    """
    keep = senders != receivers
    return senders[keep], receivers[keep]


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
    order = jnp.argsort(senders, stable=True)
    sorted_senders = senders[order]

    # Every edge i -> v takes the whole block at v, one slot per successor
    starts = jnp.searchsorted(sorted_senders, receivers)
    successors = jnp.searchsorted(sorted_senders, receivers, side="right") - starts
    line_senders = jnp.repeat(jnp.arange(senders.shape[0]), successors)
    offsets = starts - jnp.cumsum(successors) + successors
    line_receivers = order[offsets[line_senders] + jnp.arange(line_senders.shape[0])]

    # Drop i -> v -> i, which sends each edge straight back down its own reverse
    if non_backtracking:
        keep = senders[line_senders] != receivers[line_receivers]
        line_senders, line_receivers = line_senders[keep], line_receivers[keep]

    return line_senders, line_receivers, receivers[line_senders]
