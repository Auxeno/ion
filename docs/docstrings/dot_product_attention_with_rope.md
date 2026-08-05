Dot-product attention with rotary embeddings applied to query and key.

Parameters
----------
query : jax.Array["b s h k", float]
    Query vectors, one per head.
key : jax.Array["b t j k", float]
    Key vectors, one per key/value head.
value : jax.Array["b t j k", float]
    Value vectors, one per key/value head. Left unrotated.
rope : RoPE
    Rotary embedding applied to `query` and `key`. Its `axis` must match the
    layout of those arrays.
**kwargs : Any
    Forwarded to `jax.nn.dot_product_attention`.

Returns
-------
jax.Array["b s h k", float]
    Attention output, one vector per query head.

Example
-------
```python
rope = nn.RoPE()
attn = nn.MultiHeadAttention(
    768,
    num_heads=12,
    attention_fn=functools.partial(nn.dot_product_attention_with_rope, rope=rope),
    key=key,
)
x = jnp.ones((4, 128, 768))
y = attn(x)  # (4, 128, 768) -> (4, 128, 768)
```
