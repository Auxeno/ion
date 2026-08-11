"""Graph readout layers.

Modules:
    GlobalAttentionPool  Learned attention-weighted graph readout.

The score and optional value modules are supplied by the caller.
"""

from jaxtyping import Array, Float, Int

from ...nn.module import Module
from ..ops import segment_softmax, segment_sum


class GlobalAttentionPool(Module):
    """Global attention pooling.

    >>> pool = GlobalAttentionPool(score=Linear(16, 1, use_bias=False, key=key))
    >>> pool(x, graph_ids, num_graphs=4)  # (n, 16) -> (4, 16)
    """

    score: Module
    value: Module | None

    def __init__(
        self,
        score: Module,
        *,
        value: Module | None = None,
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
