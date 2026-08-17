Drop the input for whole samples.

Parameters
----------
x : jax.Array["b ...", float]
    Input with one leading batch dimension.
training : bool
    Whether to apply stochastic depth. Evaluation returns the input unchanged.
key : jax.Array | None, default=None
    RNG key for the drop mask. Required during training when `p > 0`.

Returns
-------
jax.Array["b ...", float]
    Masked and rescaled output, or `x` unchanged during evaluation. The shape
    and dtype match the input. Each sample is kept or dropped as a whole.

Info
----
Exactly one leading batch dimension; use `jax.vmap` for extra batch dimensions,
mapping over `key` as well so each slice draws its own mask.
