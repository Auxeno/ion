Apply the wrapped module with its normalized parameter.

Parameters
----------
x : Float[Array, "..."]
    Input to the wrapped module.
training : bool
    Whether to update the power-iteration vectors.

Returns
-------
Float[Array, "..."]
    Wrapped module output.

Info
----
Training refines the power-iteration vectors in place, so the same layer
instance carries them into the next call.
