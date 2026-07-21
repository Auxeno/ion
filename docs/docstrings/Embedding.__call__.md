Look up embedding vectors by integer id.

Parameters
----------
x : jax.Array["...", int]
    Integer ids in `[0, num_embeddings)`, any shape.

Returns
-------
jax.Array["... d", float]
    Embedding vectors, with a new trailing dimension of size `dim`.
