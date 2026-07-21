Advance the state by one step.

Parameters
----------
x : jax.Array["... i", float]
    Input at the current timestep.
h : jax.Array["... i h", complex]
    Previous complex state, one bank per input feature.

Returns
-------
tuple[jax.Array["... i", float], jax.Array["... i h", complex]]
    Output at this step and the new complex state.
