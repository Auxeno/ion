Run the GRU over a sequence.

Parameters
----------
x : Float[Array, "b t i"]
    Input sequence of `t` timesteps.
hx : Float[Array, "b h"] | None, default=None
    Initial hidden state. Defaults to zeros.

Returns
-------
tuple[Float[Array, "b t h"], Float[Array, "b h"]]
    Per-timestep outputs and the final hidden state.
