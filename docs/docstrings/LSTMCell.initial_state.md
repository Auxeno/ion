Zero initial hidden and cell states for one sequence.

Returns
-------
tuple[jax.Array["h", float], jax.Array["h", float]]
    Unbatched `(h, c)` pair, each with shape `(hidden_dim,)` and the same dtype
    as the recurrent weights.
