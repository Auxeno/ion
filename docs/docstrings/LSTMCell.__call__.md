Advance the state by one step.

Parameters
----------
x : jax.Array["... i", float]
    Input at the current timestep.
hx : tuple[jax.Array["... h", float], jax.Array["... h", float]]
    Previous `(hidden, cell)` state.

Returns
-------
tuple[jax.Array["... h", float], jax.Array["... h", float]]
    New `(hidden, cell)` state.
