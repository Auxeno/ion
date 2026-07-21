Advance the hidden state by one step.

Parameters
----------
x : jax.Array["... i", float]
    Input at the current timestep.
h : jax.Array["... h", float]
    Previous hidden state.

Returns
-------
jax.Array["... h", float]
    New hidden state.
