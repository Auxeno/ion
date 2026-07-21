ALiBi linear position bias (Press et al., 2022).

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
Float[Array, "num_heads seq_len seq_len"]
    Additive attention bias, one matrix per head.

Notes
-----
A parameter-free alternative to positional embeddings that extrapolates to longer sequences than seen in training.

Examples
--------
>>> bias = nn.alibi(128, 8)  # (8, 128, 128)
>>> logits = logits + bias   # add to (b, 8, 128, 128) attention logits
