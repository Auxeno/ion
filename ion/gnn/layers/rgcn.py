"""Relational graph convolutional layers.

Modules:
    RGCNConv  Per-relation neighbor transforms.  (Schlichtkrull et al., 2018)

Glorot uniform weight init, zeros for bias.
Relation types are a per-edge index array, so one edge list carries every relation.
Optional basis decomposition shares `num_bases` matrices across the relations.
Self-loops are not needed because the layer has a separate root term.
"""

import jax
import jax.numpy as jnp
from jax.nn.initializers import Initializer, glorot_uniform, zeros
from jaxtyping import Array, Float, Int, PRNGKeyArray

from ...nn.module import Module
from ...nn.param import Param
from ..ops import degree, segment_sum


class RGCNConv(Module):
    """Relational graph convolutional layer.

    >>> conv = RGCNConv(16, 32, 6, key=key)
    >>> conv(x, senders, receivers, edge_type=edge_type)  # (n, 16) -> (n, 32)
    >>> conv = RGCNConv(16, 32, 6, num_bases=2, key=key)  # share two matrices
    """

    w_neigh: Param[Float[Array, "b i o"]]
    w_coeff: Param[Float[Array, "r b"]] | None
    w_self: Param[Float[Array, "i o"]]
    b: Param[Float[Array, " o"]] | None
    num_relations: int

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_relations: int,
        *,
        num_bases: int | None = None,
        use_bias: bool = True,
        w_init: Initializer = glorot_uniform(),
        b_init: Initializer = zeros,
        key: PRNGKeyArray,
    ) -> None:

        key_neigh, key_coeff, key_self, key_b = jax.random.split(key, 4)
        self.w_neigh = Param(
            w_init(shape=(num_bases or num_relations, in_dim, out_dim), key=key_neigh)
        )
        self.w_coeff = (
            Param(w_init(shape=(num_relations, num_bases), key=key_coeff))
            if num_bases is not None
            else None
        )
        self.w_self = Param(w_init(shape=(in_dim, out_dim), key=key_self))
        self.b = Param(b_init(shape=(out_dim,), key=key_b)) if use_bias else None

        self.num_relations = num_relations

    def __call__(
        self,
        x: Float[Array, "n i"],
        senders: Int[Array, " e"],
        receivers: Int[Array, " e"],
        *,
        edge_type: Int[Array, " e"],
    ) -> Float[Array, "n o"]:

        n = x.shape[0]

        # Each relation's transform is a learned mixture of the shared bases
        w_neigh = self.w_neigh
        if self.w_coeff is not None:
            w_neigh = jnp.einsum("rb,bio->rio", self.w_coeff, w_neigh)

        # Project every node under every relation, then take the one each edge needs
        messages = jnp.einsum("ni,rio->nro", x, w_neigh)[senders, edge_type]

        # Normalize by how many edges of the same relation reach the receiver
        segments = receivers * self.num_relations + edge_type
        count = degree(segments, n * self.num_relations).astype(x.dtype)
        messages = messages / count[segments][:, None]

        # Accumulate relation messages, then add the central-node transform
        x_out = segment_sum(messages, receivers, n) + x @ self.w_self

        if self.b is not None:
            x_out = x_out + self.b

        return x_out
