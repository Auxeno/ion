Run the LRU over a sequence with a parallel scan.

Parameters
----------
x : Float[Array, "b t i"]
    Input sequence of `t` timesteps.
hx : Complex[Array, "b h"] | None, default=None
    Initial complex state. Defaults to zeros.

Returns
-------
tuple[Float[Array, "b t i"], Complex[Array, "b h"]]
    Per-timestep outputs (dimension `in_dim`) and the final complex state.
