Apply batch normalization.

Parameters
----------
x : jax.Array["... d", float]
    Input with features last.
training : bool
    Whether to use batch statistics and update the running values.

Returns
-------
jax.Array["... d", float]
    Normalized output.

Info
----
Training updates the running statistics in place, so the same layer instance
carries them into the next call.
