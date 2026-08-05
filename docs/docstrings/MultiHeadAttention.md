Multi-head attention ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)).

Projects the input to queries, keys, and values, and projects the concatenated heads back to `dim`. Keys and values come from the input itself (self-attention) or from a separate context sequence passed as `x_kv` (cross-attention), so one layer covers both. Supports grouped-query attention ([Ainslie et al., 2023](https://arxiv.org/abs/2305.13245)), multi-query attention ([Shazeer, 2019](https://arxiv.org/abs/1911.02150)), sliding-window attention ([Beltagy et al., 2020](https://arxiv.org/abs/2004.05150)), and causal masking.

Parameters
----------
dim : int
    Query and output dimension. Must be divisible by `num_heads`.
num_heads : int, default=1
    Number of query heads.
num_kv_heads : int | None, default=None
    Number of key/value heads. Fewer than `num_heads` gives grouped-query
    attention (`1` gives multi-query); `num_heads` must be divisible by it.
    Defaults to `num_heads` (standard multi-head attention).
kv_dim : int | None, default=None
    Feature dimension of the key/value sequence. Defaults to `dim`, which is
    what self-attention and same-width cross-attention both need.
use_bias : bool, default=False
    Whether the output projection includes a bias term.
causal : bool, default=False
    If `True`, each query attends only to itself and earlier positions. With a
    context this applies over the `(s, t)` query-key grid.
window : int | tuple[int, int] | None, default=None
    Sliding-window attention. An int gives a symmetric window; a `(left, right)`
    tuple sets each side. `None` attends over the full sequence.
w_init : Initializer
    Projection initializer. Glorot uniform by default.
b_init : Initializer
    Bias initializer. Zeros by default.
attention_fn : Callable[..., Array], default=jax.nn.dot_product_attention
    Function applied to the projected query, key, and value arrays. It receives
    JAX-compatible `mask`, `is_causal`, and `local_window_size` keywords.
key : jax.Array
    RNG key for parameter initialization. Keyword-only.

Attributes
----------
w_q : Param
    Query projection of shape `(dim, num_heads * head_dim)`.
w_k, w_v : Param
    Key and value projections of shape `(kv_dim, num_kv_heads * head_dim)`.
w_out : Param
    Output projection of shape `(num_heads * head_dim, dim)`.
b_out : Param | None
    Output bias of shape `(dim,)`. `None` when `use_bias=False`.

Example
-------
```python
batch, seq, dim = 4, 16, 64
attn = nn.MultiHeadAttention(dim, num_heads=8, key=key)
x = jnp.ones((batch, seq, dim))
y = attn(x)  # (4, 16, 64) -> (4, 16, 64)

x_batched = jnp.ones((5, batch, seq, dim))  # extra batch dim
y_batched = jax.vmap(attn)(x_batched)  # (5, 4, 16, 64) -> (5, 4, 16, 64)

# GQA, causal, with a sliding window of length 5
attn_gqa = nn.MultiHeadAttention(dim, num_heads=8, num_kv_heads=2, causal=True, window=5, key=key)
y = attn_gqa(x)  # (4, 16, 64) -> (4, 16, 64)

# Cross-attention into a wider context sequence
context_seq, context_dim = 32, 128
attn_cross = nn.MultiHeadAttention(dim, num_heads=8, kv_dim=context_dim, key=key)
x_kv = jnp.ones((batch, context_seq, context_dim))
y = attn_cross(x, x_kv)  # (4, 16, 64), (4, 32, 128) -> (4, 16, 64)

# cuDNN FlashAttention-2 with bfloat16 on supported hardware
flash_attn = nn.MultiHeadAttention(
    dim,
    num_heads=8,
    attention_fn=functools.partial(jax.nn.dot_product_attention, implementation="cudnn"),
    key=key,
)
flash_attn = flash_attn.astype(jnp.bfloat16)
y = flash_attn(x.astype(jnp.bfloat16))
```
