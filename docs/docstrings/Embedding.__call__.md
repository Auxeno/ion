Look up embedding vectors by integer id.

Parameters
----------
x : Int[Array, "..."]
    Integer ids in `[0, num_embeddings)`, any shape.

Returns
-------
Float[Array, "... d"]
    Embedding vectors, with a new trailing dimension of size `dim`.
