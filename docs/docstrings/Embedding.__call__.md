Look up embedding vectors by integer id.

Parameters
----------
x : jax.Array["...", int]
    Integer ids in `[0, num_embeddings)`.

Returns
-------
jax.Array["... d", float]
    Embedding vectors, with a new trailing dimension of size `dim`.

Info
----
Ids may have any shape; the output appends a trailing `dim` axis.

Warning
-------
Ids outside `[0, num_embeddings)` are silently clamped to the nearest valid row by JAX rather than raising, so validate inputs upstream.
