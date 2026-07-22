Low-rank adaptation of a linear layer ([Hu et al., 2021](https://arxiv.org/abs/2106.09685)).

Wraps a frozen `Linear` with a trainable low-rank update `x @ A @ B`, scaled by `alpha / rank`. Only the small `A` and `B` factors receive gradients, so a large base layer can be adapted cheaply.

Parameters
----------
linear : Linear
    Base layer to adapt. It is frozen on construction, so its weights receive
    no gradients.
rank : int, default=8
    Rank of the low-rank update. Must be >= 1.
alpha : float | None, default=None
    Scaling factor; the update is multiplied by `alpha / rank`. Defaults to
    `rank`, giving neutral (unit) scaling.
a_init : Initializer
    Initializer for `A`. He normal by default.
b_init : Initializer
    Initializer for `B`. Zeros by default, so the update starts at zero and the
    wrapped layer is unchanged at initialization.
key : jax.Array
    RNG key for parameter initialization. Keyword-only.

Attributes
----------
linear : Linear
    The frozen base layer.
a : Param
    Down-projection of shape `(in_dim, rank)`.
b : Param
    Up-projection of shape `(rank, out_dim)`.

Notes
-----
Pairs with the optimizer's [auto-partitioning](../../core/optimizer.md), which allocates no state for the frozen base and updates only `A` and `B`.

Examples
--------
>>> lora = nn.LoRALinear(nn.Linear(64, 128, key=keys[0]), rank=8, key=keys[1])
>>> y = lora(x)  # (*, 64) -> (*, 128)
