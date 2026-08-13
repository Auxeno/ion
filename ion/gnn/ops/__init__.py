"""Graph reductions, pooling, batching, and topology operations."""

from .batching import batch_graphs, unbatch_graphs
from .graph import (
    add_self_loops,
    coalesce,
    degree,
    from_adjacency,
    line_graph,
    remove_self_loops,
    to_adjacency,
    to_undirected,
)
from .pool import max_pool, mean_pool, min_pool, sum_pool
from .segment import (
    segment_max,
    segment_mean,
    segment_min,
    segment_prod,
    segment_softmax,
    segment_std,
    segment_sum,
    segment_var,
)

__all__ = [
    "add_self_loops",
    "batch_graphs",
    "coalesce",
    "degree",
    "from_adjacency",
    "line_graph",
    "max_pool",
    "mean_pool",
    "min_pool",
    "remove_self_loops",
    "segment_max",
    "segment_mean",
    "segment_min",
    "segment_prod",
    "segment_softmax",
    "segment_std",
    "segment_sum",
    "segment_var",
    "sum_pool",
    "to_adjacency",
    "to_undirected",
    "unbatch_graphs",
]
