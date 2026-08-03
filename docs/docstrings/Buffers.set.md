Return a new buffer collection with one module's value replaced.

Parameters
----------
module : BufferModule
    Module that owns the value.
value : PyTree
    New non-empty pytree of arrays.

Returns
-------
Buffers
    Copy containing the new value.

Info
----
The new value must preserve the initialized pytree structure, leaf shapes, and
leaf dtypes.
