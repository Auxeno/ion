Advance the state by one step.

Parameters
----------
x : Float[Array, "... i"]
    Input at the current timestep.
h : Complex[Array, "... i h"]
    Previous complex state, one bank per input feature.

Returns
-------
tuple[Float[Array, "... i"], Complex[Array, "... i h"]]
    Output at this step and the new complex state.
