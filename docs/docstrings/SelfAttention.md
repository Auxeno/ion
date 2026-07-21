Multi-head self-attention (Vaswani et al., 2017).

Projects the input to queries, keys, and values, attends within a single sequence, and projects the concatenated heads back to `dim`. Supports grouped-query and multi-query attention, causal masking, and sliding-window attention.

Parameters
----------
dim : int
    Model dimension. Must be divisible by `num_heads`.
num_heads : int, default=1
    Number of query heads.
num_kv_heads : int | None, default=None
    Number of key/value heads. Fewer than `num_heads` gives grouped-query
    attention (`1` gives multi-query); `num_heads` must be divisible by it.
    Defaults to `num_heads` (standard multi-head attention).
bias : bool, default=False
    Whether the output projection includes a bias term.
causal : bool, default=False
    If `True`, each position attends only to itself and earlier positions.
window : int | tuple[int, int] | None, default=None
    Sliding-window attention. An int gives a symmetric window; a `(left, right)`
    tuple sets each side. `None` attends over the full sequence.
w_init : Initializer
    Projection initializer. Truncated normal (std 0.02) by default.
b_init : Initializer
    Bias initializer. Zeros by default.
key : jax.Array
    RNG key for parameter initialization. Keyword-only.

Attributes
----------
w_q : Param
    Query projection of shape `(dim, num_heads * head_dim)`.
w_k, w_v : Param
    Key and value projections of shape `(dim, num_kv_heads * head_dim)`.
w_out : Param
    Output projection of shape `(num_heads * head_dim, dim)`.
b_out : Param | None
    Output bias of shape `(dim,)`. `None` when `bias=False`.

Notes
-----
`head_dim` is `dim // num_heads`. Weights are stored flat 2D and reshaped into heads in the forward pass, so a custom variance-scaling `w_init` sees the true fan sizes. `causal` and `window` compose with an explicit call-time `mask`; see [Reference](../reference.md#attention-masking) for mask shapes.

Examples
--------
>>> attn = nn.SelfAttention(64, num_heads=8, key=key)
>>> y = attn(x)                                  # (b, s, 64) -> (b, s, 64)
>>> attn = nn.SelfAttention(64, num_heads=8, num_kv_heads=2, causal=True, key=key)
>>> y = attn(x)                                  # grouped-query, causal
