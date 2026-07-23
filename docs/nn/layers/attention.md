# Attention

Multi-head attention. `SelfAttention` attends within one sequence (optionally causal); `CrossAttention` attends from a query sequence into a separate context.

::: ion.nn.SelfAttention

::: ion.nn.CrossAttention

---

## Masking

`SelfAttention` and `CrossAttention` accept an optional boolean `mask`, where
`True` means attend and `False` means ignore. A self-attention mask may have
shape `(s, s)`, `(b, s, s)`, or `(b, h, s, s)`.

```python
attn = nn.SelfAttention(64, num_heads=8, causal=True, key=key)
attn(x)  # applies a lower-triangular mask

valid = jnp.arange(seq_len)[None, :] < lengths[:, None]
attn(x, mask=valid[:, None, :] & valid[:, :, None])

ids = jnp.arange(seq_len)
window = jnp.abs(ids[:, None] - ids[None, :]) <= 32
attn(x, mask=window)
```

For `CrossAttention`, the last two mask dimensions are query and context:

```python
mask = jnp.ones((src_len, tgt_len), dtype=bool)
cross_attn(x, context, mask=mask)
```
