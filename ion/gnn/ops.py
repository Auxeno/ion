"""Graph operations.

Functions:
    segment_softmax  Softmax normalized within segments (e.g. per-node neighborhoods).
    add_self_loops   Append identity edges so every node sends a message to itself.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int


def segment_softmax(
    data: Float[Array, "e ..."],
    segment_ids: Int[Array, " e"],
    num_segments: int,
) -> Float[Array, "e ..."]:
    """Softmax normalized within each segment.

    >>> weights = segment_softmax(logits, receivers, num_nodes)
    """
    # Subtract per-segment max for numerical stability
    maxes = jax.ops.segment_max(data, segment_ids, num_segments)
    data = jnp.exp(data - maxes[segment_ids])

    # Normalize by per-segment sum
    sums = jax.ops.segment_sum(data, segment_ids, num_segments)
    return data / (sums[segment_ids] + 1e-6)


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
