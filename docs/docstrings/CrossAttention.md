Multi-head cross-attention (Vaswani et al., 2017).

Queries come from one sequence and keys/values from a separate context, so a decoder can attend into an encoder's output. Projects the concatenated heads back to `dim`.

Parameters
----------
dim : int
    Query and output dimension. Must be divisible by `num_heads`.
num_heads : int, default=1
    Number of attention heads.
context_dim : int | None, default=None
    Feature dimension of the context (key/value) sequence. Defaults to `dim`.
bias : bool, default=False
    Whether the output projection includes a bias term.
w_init : Initializer
    Projection initializer. Truncated normal (std 0.02) by default.
b_init : Initializer
    Bias initializer. Zeros by default.
key : PRNGKeyArray
    RNG key for parameter initialization. Keyword-only.

Attributes
----------
w_q : Param
    Query projection of shape `(dim, dim)`.
w_k, w_v : Param
    Key and value projections of shape `(context_dim, dim)`.
w_out : Param
    Output projection of shape `(dim, dim)`.
b_out : Param | None
    Output bias of shape `(dim,)`. `None` when `bias=False`.

Notes
-----
The call-time `mask` matches the query-key dimensions; see [Conventions](../conventions.md#attention-masking) for shapes.

Examples
--------
>>> attn = nn.CrossAttention(64, num_heads=8, key=key)
>>> y = attn(x, context)  # (b, s, 64), (b, t, 64) -> (b, s, 64)
