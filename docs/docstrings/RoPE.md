Rotary positional embeddings ([Su et al., 2021](https://arxiv.org/abs/2104.09864)).

Rotates adjacent pairs of features by an angle proportional to their position, applied to query and key before attention. Relative position falls out of the dot product, and there are no learnable parameters.

For position \(p\), each adjacent feature pair is rotated at its own frequency:

\[
\begin{bmatrix}x'_{2k}\\x'_{2k+1}\end{bmatrix}
=
\begin{bmatrix}
\cos\theta_{p,k} & -\sin\theta_{p,k}\\
\sin\theta_{p,k} & \cos\theta_{p,k}
\end{bmatrix}
\begin{bmatrix}x_{2k}\\x_{2k+1}\end{bmatrix},
\qquad
\theta_{p,k}=p\,\theta^{-2k/d}.
\]

Positions form a 1D sequence by default. Passing `shape` lays them out on an N-dimensional lattice instead, giving each axis its own section of the head dimension ([Heo et al., 2024](https://arxiv.org/abs/2403.13298)).

Parameters
----------
shape : tuple[int, ...] | None, default=None
    Extents of the position lattice, row-major, one entry per axis. `None`
    treats positions as a flat sequence. `head_dim` must be divisible by
    `2 * len(shape)`.
num_prefix_tokens : int, default=0
    Number of leading tokens held at position 0, where the rotation is the
    identity. Use this for CLS or register tokens.
axis : int, default=-3
    Axis holding the sequence positions. The default matches the
    `(batch, sequence, heads, head_dim)` queries and keys that
    `MultiHeadAttention` builds. Pass `-2` for a head-first layout.
theta : float, default=10000.0
    Base wavelength controlling the rotation frequencies across feature pairs.

Example
-------
```python
rope = nn.RoPE()
q = jnp.ones((4, 16, 8, 32))  # (batch, sequence, heads, head_dim)
k = jnp.ones((4, 16, 8, 32))
q = rope(q)  # (4, 16, 8, 32) -> (4, 16, 8, 32)
k = rope(k)  # (4, 16, 8, 32) -> (4, 16, 8, 32)

# Bound into an attention layer, which applies it to query and key
def attention_with_rope(q, k, v, *, rope, **kwargs):
    return jax.nn.dot_product_attention(rope(q), rope(k), v, **kwargs)

attn = nn.MultiHeadAttention(
    768,
    num_heads=12,
    attention_fn=functools.partial(attention_with_rope, rope=rope),
    key=key,
)
x = jnp.ones((4, 128, 768))
y = attn(x)  # (4, 128, 768) -> (4, 128, 768)

# 2D lattice over a 14x14 patch grid, behind a single CLS token
rope_2d = nn.RoPE(shape=(14, 14), num_prefix_tokens=1)
q = jnp.ones((4, 197, 8, 32))
q = rope_2d(q)  # (4, 197, 8, 32) -> (4, 197, 8, 32)
```
