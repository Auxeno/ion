"""Graph pooling layers.

Modules:
    GlobalAttentionPool         Learned attention-weighted graph readout.
    MultiHeadAttentionPool      Learned seed queries attending over each graph.  (Lee et al., 2019)

GlobalAttentionPool takes its score and optional value callables from the caller.
MultiHeadAttentionPool learns its seeds directly as queries, so it needs no query
projection, and returns one row per seed. Glorot uniform weight init, zeros for bias.
"""

from collections.abc import Callable
import math

import jax
import jax.numpy as jnp
from jax.nn.initializers import Initializer, glorot_uniform, zeros
from jaxtyping import Array, Float, Int, PRNGKeyArray

from ...nn.module import Module
from ...nn.param import Param
from ..ops import segment_softmax, segment_sum


class GlobalAttentionPool(Module):
    """Global attention pooling.

    >>> pool = GlobalAttentionPool(score=Linear(16, 1, use_bias=False, key=key))
    >>> pool(x, graph_ids, num_graphs=4)  # (n, 16) -> (4, 16)
    """

    score: Callable[[Array], Array]
    value: Callable[[Array], Array] | None

    def __init__(
        self,
        score: Callable[[Array], Array],
        *,
        value: Callable[[Array], Array] | None = None,
    ) -> None:

        self.score = score
        self.value = value

    def __call__(
        self,
        x: Float[Array, "n i"],
        graph_ids: Int[Array, " n"],
        num_graphs: int,
    ) -> Float[Array, "g o"]:

        logits = self.score(x)
        if logits.shape != (x.shape[0], 1):
            raise ValueError(f"score must return shape ({x.shape[0]}, 1), got {logits.shape}")

        attention = segment_softmax(logits, graph_ids, num_graphs)
        values = x if self.value is None else self.value(x)

        return segment_sum(attention * values, graph_ids, num_graphs)


class MultiHeadAttentionPool(Module):
    """Pooling by multi-head attention over learned seeds.

    >>> pool = MultiHeadAttentionPool(16, 32, num_seeds=4, num_heads=4, key=key)
    >>> pool(x, graph_ids, num_graphs=2)  # (n, 16) -> (2, 4, 32)
    """

    seeds: Param[Float[Array, "s o"]]
    w_k: Param[Float[Array, "i o"]]
    w_v: Param[Float[Array, "i o"]]
    w_out: Param[Float[Array, "o o"]]
    b_out: Param[Float[Array, " o"]] | None
    num_seeds: int
    num_heads: int

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        num_seeds: int = 1,
        num_heads: int = 1,
        use_bias: bool = True,
        w_init: Initializer = glorot_uniform(),
        seed_init: Initializer = glorot_uniform(),
        b_init: Initializer = zeros,
        key: PRNGKeyArray,
    ) -> None:

        if out_dim % num_heads != 0:
            raise ValueError(f"out_dim ({out_dim}) must be divisible by num_heads ({num_heads})")

        key_seeds, key_k, key_v, key_out, key_b = jax.random.split(key, 5)
        self.seeds = Param(seed_init(shape=(num_seeds, out_dim), key=key_seeds))
        self.w_k = Param(w_init(shape=(in_dim, out_dim), key=key_k))
        self.w_v = Param(w_init(shape=(in_dim, out_dim), key=key_v))
        self.w_out = Param(w_init(shape=(out_dim, out_dim), key=key_out))
        self.b_out = Param(b_init(shape=(out_dim,), key=key_b)) if use_bias else None

        self.num_seeds = num_seeds
        self.num_heads = num_heads

    def __call__(
        self,
        x: Float[Array, "n i"],
        graph_ids: Int[Array, " n"],
        num_graphs: int,
    ) -> Float[Array, "g s o"]:

        n = x.shape[0]

        # Seeds are the queries
        q = jnp.reshape(self.seeds, (self.num_seeds, self.num_heads, -1))
        k = (x @ self.w_k).reshape(n, self.num_heads, -1)
        v = (x @ self.w_v).reshape(n, self.num_heads, -1)
        head_dim = q.shape[-1]

        # Scaled dot product between every seed and every node
        logits = jnp.einsum("shd,nhd->nsh", q, k) / math.sqrt(head_dim)

        # Softmax over the nodes of each graph, separately per seed and head
        attention = segment_softmax(logits, graph_ids, num_graphs)

        # Aggregate node values weighted by attention
        out = segment_sum(attention[..., None] * v[:, None], graph_ids, num_graphs)
        out = out.reshape(num_graphs, self.num_seeds, -1) @ self.w_out

        if self.b_out is not None:
            out = out + self.b_out

        return out
