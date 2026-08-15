"""Graph normalization layers.

Modules:
    GraphNorm  Per-graph normalization with a learnable mean scale.  (Cai et al., 2021)

Statistics are computed independently over the nodes of each graph.
"""

import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float, Int

from ...nn.module import Module
from ...nn.param import Param
from ..ops import segment_mean


class GraphNorm(Module):
    """Graph normalization with a learnable mean scale.

    >>> norm = GraphNorm(64)
    >>> norm(x)  # (n, 64) -> (n, 64)
    """

    scale: Param[Float[Array, " d"]]
    b: Param[Float[Array, " d"]] | None
    mean_scale: Param[Float[Array, " d"]]
    eps: float

    def __init__(self, dim: int, *, eps: float = 1e-5, use_bias: bool = True) -> None:

        self.scale = Param(jnp.ones(dim))
        self.b = Param(jnp.zeros(dim)) if use_bias else None
        self.mean_scale = Param(jnp.ones(dim))

        self.eps = eps

    def __call__(
        self,
        x: Float[Array, "n d"],
        graph_ids: Int[Array, " n"] | None = None,
        num_graphs: int | None = None,
    ) -> Float[Array, "n d"]:

        dtype = x.dtype
        x = x.astype(jnp.float32)

        if graph_ids is None:
            if num_graphs is not None:
                raise ValueError("num_graphs requires graph_ids")

            mean = jnp.mean(x, axis=0)
            x = x - self.mean_scale * mean
            var = jnp.mean(jnp.square(x), axis=0)
            x = x * lax.rsqrt(var + self.eps)
        else:
            if num_graphs is None:
                raise ValueError("num_graphs is required with graph_ids")

            mean = segment_mean(x, graph_ids, num_graphs)
            x = x - self.mean_scale * mean[graph_ids]

            var = segment_mean(jnp.square(x), graph_ids, num_graphs)
            x = x * lax.rsqrt(var[graph_ids] + self.eps)

        x = x * self.scale
        if self.b is not None:
            x = x + self.b

        return x.astype(dtype)
