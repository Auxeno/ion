"""Utilities for batching graphs."""

from collections.abc import Sequence

import jax.numpy as jnp
from jaxtyping import Array, Float, Int

__all__ = ["batch_graphs"]


def batch_graphs(
    xs: Sequence[Float[Array, "n d"]],
    senders: Sequence[Int[Array, " e"]],
    receivers: Sequence[Int[Array, " e"]],
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
