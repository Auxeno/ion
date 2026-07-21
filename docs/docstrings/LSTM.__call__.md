Run the LSTM over a sequence.

Parameters
----------
x : jax.Array["b t i", float]
    Input sequence of `t` timesteps.
hx : tuple[jax.Array["b h", float], jax.Array["b h", float]] | None, default=None
    Initial `(hidden, cell)` state. Defaults to zeros.

Returns
-------
tuple[jax.Array["b t h", float], tuple[jax.Array["b h", float], jax.Array["b h", float]]]
    Per-timestep outputs and the final `(hidden, cell)` state.
