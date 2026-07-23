Multi-head cross-attention ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)).

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
    Projection initializer. Glorot uniform by default.
b_init : Initializer
    Bias initializer. Zeros by default.
key : jax.Array
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

Example
-------
```python
batch, query_seq, context_seq, dim = 4, 16, 32, 64
attn = nn.CrossAttention(dim, num_heads=8, key=key)
x = jnp.ones((batch, query_seq, dim))
context = jnp.ones((batch, context_seq, dim))
y = attn(x, context)  # (4, 16, 64), (4, 32, 64) -> (4, 16, 64)

x_batched = jnp.ones((5, batch, query_seq, dim))  # extra batch dim
context_batched = jnp.ones((5, batch, context_seq, dim))
y_batched = jax.vmap(attn)(x_batched, context_batched)  # (5, 4, 16, 64), (5, 4, 32, 64) -> (5, 4, 16, 64)
```
