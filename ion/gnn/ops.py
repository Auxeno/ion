"""Graph operations.

Functions:
    segment_sum      Sum reduction within segments. (Re-exported from jax.ops.)
    segment_max      Maximum reduction within segments. (Re-exported from jax.ops.)
    segment_min      Minimum reduction within segments. (Re-exported from jax.ops.)
    segment_prod     Product reduction within segments. (Re-exported from jax.ops.)
    segment_softmax  Softmax normalized within segments (e.g. per-node neighborhoods).
    segment_mean     Mean reduction within segments.
    mean_pool        Average node features within each graph (graph-level readout).
    sum_pool         Sum node features within each graph.
    max_pool         Maximum node features within each graph.
    batch_graphs     Pack graphs into one disconnected graph for batched message passing.
    add_self_loops   Append identity edges so every node sends a message to itself.

`segment_sum`, `segment_max`, `segment_min` and `segment_prod` are re-exported
from `jax.ops`, so all graph layers and public segment reductions share this
module as their canonical namespace.
"""

from collections.abc import Sequence

import jax.numpy as jnp
from jax.ops import segment_max, segment_min, segment_prod, segment_sum
from jaxtyping import Array, Float, Int

__all__ = [
    "add_self_loops",
    "batch_graphs",
    "max_pool",
    "mean_pool",
    "segment_max",
    "segment_mean",
    "segment_min",
    "segment_prod",
    "segment_softmax",
    "segment_sum",
    "sum_pool",
]


def segment_softmax(
    data: Float[Array, "e ..."],
    segment_ids: Int[Array, " e"],
    num_segments: int,
) -> Float[Array, "e ..."]:
    """Softmax normalized within each segment.

    >>> weights = segment_softmax(logits, receivers, num_nodes)
    """
    # Subtract per-segment max for numerical stability
    maxes = segment_max(data, segment_ids, num_segments)
    maxes = jnp.where(jnp.isinf(maxes), 0.0, maxes)
    data = jnp.exp(data - maxes[segment_ids])

    # Normalize by per-segment sum; guard empty segments (never indexed) against 0 / 0
    sums = segment_sum(data, segment_ids, num_segments)
    return data / jnp.where(sums == 0, 1.0, sums)[segment_ids]


def segment_mean(
    data: Float[Array, "e ..."],
    segment_ids: Int[Array, " e"],
    num_segments: int,
) -> Float[Array, "s ..."]:
    """Mean of data within each segment; empty segments give zeros.

    >>> means = segment_mean(messages, receivers, num_nodes)
    """
    sums = segment_sum(data, segment_ids, num_segments)

    # Count segment members; guard empty segments against 0 / 0
    counts = segment_sum(jnp.ones_like(segment_ids, dtype=data.dtype), segment_ids, num_segments)
    counts = jnp.maximum(counts, 1)
    return sums / counts.reshape(-1, *(1,) * (data.ndim - 1))


def mean_pool(
    x: Float[Array, "n d"],
    graph_ids: Int[Array, " n"],
    num_graphs: int,
) -> Float[Array, "g d"]:
    """Average node features within each graph.

    >>> g = mean_pool(x, graph_ids, num_graphs)  # (n, d) -> (g, d)
    """
    return segment_mean(x, graph_ids, num_graphs)


def sum_pool(
    x: Float[Array, "n d"],
    graph_ids: Int[Array, " n"],
    num_graphs: int,
) -> Float[Array, "g d"]:
    """Sum node features within each graph.

    >>> g = sum_pool(x, graph_ids, num_graphs)  # (n, d) -> (g, d)
    """
    return segment_sum(x, graph_ids, num_graphs)


def max_pool(
    x: Float[Array, "n d"],
    graph_ids: Int[Array, " n"],
    num_graphs: int,
) -> Float[Array, "g d"]:
    """Maximum of node features within each graph; empty graphs give zeros.

    >>> g = max_pool(x, graph_ids, num_graphs)  # (n, d) -> (g, d)
    """
    maxes = segment_max(x, graph_ids, num_graphs)

    # segment_max fills empty segments with -inf
    return jnp.where(jnp.isneginf(maxes), 0.0, maxes)


def batch_graphs(
    xs: Sequence[Float[Array, "_ d"]],
    senders: Sequence[Int[Array, " _"]],
    receivers: Sequence[Int[Array, " _"]],
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
    offsets = jnp.cumsum(jnp.array([0] + sizes[:-1]))
    graph_ids = jnp.repeat(jnp.arange(len(xs)), jnp.array(sizes))
    return (
        jnp.concatenate(xs),
        jnp.concatenate([s + o for s, o in zip(senders, offsets)]),
        jnp.concatenate([r + o for r, o in zip(receivers, offsets)]),
        graph_ids,
    )


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
