"""Graph-level readout operations."""

import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from .segment import segment_max, segment_mean, segment_sum

__all__ = ["max_pool", "mean_pool", "sum_pool"]


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
