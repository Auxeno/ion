Apply the wrapped module with its normalized parameter.

Parameters
----------
x : Any
    Input to the wrapped module.
training : bool
    Whether to update the power-iteration vectors.

Returns
-------
Any
    Wrapped module output.

Info
----
Training refines the power-iteration vectors in place, so the same layer
instance carries them into the next call.
