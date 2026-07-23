ALiBi linear position bias ([Press et al., 2022](https://arxiv.org/abs/2108.12409)).

Builds a per-head bias that penalizes attention between distant positions by a distance proportional to a fixed, head-specific slope. Add the result to attention logits before the softmax; it holds no parameters and needs no `key`.

Parameters
----------
seq_len : int
    Sequence length.
num_heads : int
    Number of attention heads. Each head gets a distinct geometric slope.
dtype : jnp.dtype, default=jnp.float32
    Dtype of the returned array.

Returns
-------
jax.Array["num_heads seq_len seq_len", float]
    Additive attention bias, one matrix per head.

Example
-------
```python
bias = nn.alibi(128, 8)  # (8, 128, 128)
logits = jnp.ones((4, 8, 128, 128))
logits = logits + bias  # (4, 8, 128, 128), broadcast across the batch
```
