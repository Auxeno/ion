"""Segment reduction operations.

Functions:
    segment_sum        Sum reduction within segments with float32 accumulation.
    segment_max        Maximum reduction within segments. (Re-exported from jax.ops.)
    segment_min        Minimum reduction within segments. (Re-exported from jax.ops.)
    segment_prod       Product reduction within segments. (Re-exported from jax.ops.)
    segment_softmax    Softmax normalized within segments (e.g. per-node neighborhoods).
    segment_mean       Mean reduction within segments.
    segment_var        Population variance within segments.
    segment_std        Population standard deviation within segments.
`segment_max`, `segment_min` and `segment_prod` are re-exported from `jax.ops`.
`segment_sum` wraps the JAX operation with float32 accumulation for floating-point data.
"""

from typing import Any

import jax
import jax.numpy as jnp
from jax.ops import segment_max, segment_min, segment_prod
from jaxtyping import Array, Float, Int

__all__ = [
    "segment_max",
    "segment_mean",
    "segment_min",
    "segment_prod",
    "segment_softmax",
    "segment_std",
    "segment_sum",
    "segment_var",
]


def segment_sum(
    data: Array,
    segment_ids: Int[Array, " e"],
    num_segments: int | None = None,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    bucket_size: int | None = None,
    mode: Any = None,
) -> Array:
    """Sum values within each segment.

    >>> sums = segment_sum(messages, receivers, num_nodes)
    """
    dtype = data.dtype
    if jnp.issubdtype(dtype, jnp.floating):
        data = data.astype(jnp.float32)

    return jax.ops.segment_sum(
        data,
        segment_ids,
        num_segments,
        indices_are_sorted=indices_are_sorted,
        unique_indices=unique_indices,
        bucket_size=bucket_size,
        mode=mode,
    ).astype(dtype)


def segment_softmax(
    data: Float[Array, "e ..."],
    segment_ids: Int[Array, " e"],
    num_segments: int,
) -> Float[Array, "e ..."]:
    """Softmax normalized within each segment.

    >>> weights = segment_softmax(logits, receivers, num_nodes)
    """
    dtype = data.dtype
    data = data.astype(jnp.float32)

    # Subtract per-segment max for numerical stability
    maxes = segment_max(data, segment_ids, num_segments)
    maxes = jnp.where(jnp.isinf(maxes), 0.0, maxes)
    data = jnp.exp(data - maxes[segment_ids])

    # Normalize by per-segment sum
    sums = segment_sum(data, segment_ids, num_segments)
    return (data / jnp.where(sums == 0, 1.0, sums)[segment_ids]).astype(dtype)


def segment_mean(
    data: Float[Array, "e ..."],
    segment_ids: Int[Array, " e"],
    num_segments: int,
) -> Float[Array, "s ..."]:
    """Mean of data within each segment; empty segments give zeros.

    >>> means = segment_mean(messages, receivers, num_nodes)
    """
    dtype = data.dtype
    data = data.astype(jnp.float32)

    sums = segment_sum(data, segment_ids, num_segments)

    # Count segment members
    counts = jnp.maximum(jnp.bincount(segment_ids, length=num_segments), 1)
    return (sums / counts.reshape(-1, *(1,) * (data.ndim - 1))).astype(dtype)


def segment_var(
    data: Float[Array, "e ..."],
    segment_ids: Int[Array, " e"],
    num_segments: int,
) -> Float[Array, "s ..."]:
    """Population variance of data within each segment; empty segments give zeros.

    >>> variances = segment_var(messages, receivers, num_nodes)
    """
    dtype = data.dtype
    data = data.astype(jnp.float32)

    # Two-pass: deviations from the segment mean, rather than E[x^2] - E[x]^2
    means = segment_mean(data, segment_ids, num_segments)
    deviations = jnp.square(data - means[segment_ids])
    return jnp.maximum(segment_mean(deviations, segment_ids, num_segments), 0.0).astype(dtype)


def segment_std(
    data: Float[Array, "e ..."],
    segment_ids: Int[Array, " e"],
    num_segments: int,
) -> Float[Array, "s ..."]:
    """Population standard deviation within each segment; empty segments give zeros.

    >>> stds = segment_std(messages, receivers, num_nodes)
    """
    return jnp.sqrt(segment_var(data, segment_ids, num_segments))
