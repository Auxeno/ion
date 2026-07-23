Run S4D over a sequence with a parallel scan.

Parameters
----------
x : jax.Array["b t i", float]
    Input sequence of `t` timesteps.
hx : jax.Array["b i h", complex] | None, default=None
    Initial complex state. Defaults to zeros.

Returns
-------
tuple[jax.Array["b t i", float], jax.Array["b i h", complex]]
    Per-timestep outputs (dimension `in_dim`) and the final complex state.

Info
----
Input is `(batch, time, features)`. Pass `hx` to override the default zero initial state.
