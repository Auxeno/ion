Process a sequence in both directions.

Parameters
----------
x : jax.Array["b t i", float]
    Input sequence in batch-first layout.
hx : tuple[PyTree, PyTree] | None, default=None
    Initial state for each direction. `None` uses each layer's default state.

Returns
-------
tuple[jax.Array["b t o", float], tuple[PyTree, PyTree]]
    Combined outputs in the original time order and both final states.
