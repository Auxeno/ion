Advance the state by one step.

Parameters
----------
x : Float[Array, "... i"]
    Input at the current timestep.
hx : tuple[Float[Array, "... h"], Float[Array, "... h"]]
    Previous `(hidden, cell)` state.

Returns
-------
tuple[Float[Array, "... h"], Float[Array, "... h"]]
    New `(hidden, cell)` state.
