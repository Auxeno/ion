Run the RNN over a sequence.

Parameters
----------
x : jax.Array["b t i", float]
    Input sequence of `t` timesteps.
hx : jax.Array["b h", float] | None, default=None
    Initial hidden state. Defaults to zeros.

Returns
-------
tuple[jax.Array["b t h", float], jax.Array["b h", float]]
    Per-timestep outputs and the final hidden state.
