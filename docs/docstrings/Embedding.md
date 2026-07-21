Token embedding lookup table.

Maps integer ids to dense vectors by indexing a learnable weight matrix. Each id in `[0, num_embeddings)` selects one row.

Parameters
----------
num_embeddings : int
    Size of the vocabulary (number of rows in the table).
dim : int
    Embedding vector dimension.
w_init : Initializer
    Weight initializer. Truncated normal (std 0.02) by default, the GPT-2/BERT
    convention.
key : PRNGKeyArray
    RNG key for parameter initialization. Keyword-only.

Attributes
----------
w : Param
    Embedding table of shape `(num_embeddings, dim)`.

Examples
--------
>>> embed = nn.Embedding(1000, 64, key=key)
>>> y = embed(ids)  # (*,) int -> (*, 64)
