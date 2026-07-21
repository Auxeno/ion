Run the LSTM over a sequence.

Parameters
----------
x : Float[Array, "b t i"]
    Input sequence of `t` timesteps.
hx : tuple[Float[Array, "b h"], Float[Array, "b h"]] | None, default=None
    Initial `(hidden, cell)` state. Defaults to zeros.

Returns
-------
tuple[Float[Array, "b t h"], tuple[Float[Array, "b h"], Float[Array, "b h"]]]
    Per-timestep outputs and the final `(hidden, cell)` state.
