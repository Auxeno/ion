Rotary positional embeddings ([Su et al., 2021](https://arxiv.org/abs/2104.09864)).

Encodes position by rotating pairs of features by an angle proportional to their position, applied to query and key vectors before attention. Relative position falls out of the dot product, and there are no learnable parameters.

Positions form a 1D sequence by default. Passing `shape` lays them out on an N-dimensional lattice instead, splitting the head dimension evenly across its axes so each axis rotates its own contiguous section. This is axial RoPE, as used for images by [RoPE-ViT](https://arxiv.org/abs/2403.13298). Coordinates are enumerated row-major, so `shape=(height, width)` varies width fastest.

Feature pairs are adjacent, so a head rotates `(x0, x1)`, `(x2, x3)`, and so on. This is the convention of the paper and of `SinusoidalPositionalEmbedding`, and it differs from the half-split some checkpoints use, which matters only when porting weights.

Parameters
----------
shape : tuple[int, ...] | None, default=None
    Extents of the position lattice, one entry per axis. `None` treats positions
    as a flat sequence. `head_dim` must be divisible by `2 * len(shape)`, which
    for a typical `head_dim` of 64 admits one or two axes but not three.
num_prefix_tokens : int, default=0
    Number of leading tokens held at position 0, where the rotation is the
    identity. Use this for CLS or register tokens that sit in front of the
    lattice and should stay position-invariant. They share position 0 with the
    first lattice site, which is also unrotated.
axis : int, default=-3
    Axis holding the sequence positions. The default matches the projected
    `(batch, sequence, heads, head_dim)` queries and keys that
    `MultiHeadAttention` builds and `jax.nn.dot_product_attention` consumes.
    Pass `-2` for a head-first `(batch, heads, sequence, head_dim)` layout, or
    for a single unbatched head shaped `(sequence, head_dim)`.
theta : float, default=10000.0
    Base wavelength controlling the rotation frequencies across feature pairs.

Notes
-----
`shape`, `num_prefix_tokens`, and `axis` are static configuration, so a RoPE is fixed to one
lattice and one tensor layout. That is what lets it be bound into a `MultiHeadAttention` layer's
`attention_fn` at construction time.

Example
-------
```python
rope = nn.RoPE()
q = jnp.ones((4, 16, 8, 32))  # (batch, sequence, heads, head_dim)
k = jnp.ones((4, 16, 8, 32))
q = rope(q)  # (4, 16, 8, 32) -> (4, 16, 8, 32)
k = rope(k)  # (4, 16, 8, 32) -> (4, 16, 8, 32)

# Head-first layout, or a single unbatched head
rope_head_first = nn.RoPE(axis=-2)
q = jnp.ones((4, 8, 16, 32))
q = rope_head_first(q)  # (4, 8, 16, 32) -> (4, 8, 16, 32)

# 2D lattice over a 14x14 patch grid, behind a single CLS token
rope_2d = nn.RoPE(shape=(14, 14), num_prefix_tokens=1)
q = jnp.ones((4, 197, 8, 32))
q = rope_2d(q)  # (4, 197, 8, 32) -> (4, 197, 8, 32)

# 3D lattice over frames and space, needing head_dim divisible by 6
rope_3d = nn.RoPE(shape=(8, 14, 14))
q = jnp.ones((4, 1568, 8, 48))
q = rope_3d(q)  # (4, 1568, 8, 48) -> (4, 1568, 8, 48)

# Bound into an attention layer, which applies it to query and key
from functools import partial

attn = nn.MultiHeadAttention(
    768,
    num_heads=12,
    attention_fn=partial(nn.dot_product_attention_with_rope, rope=rope_2d),
    key=key,
)
x = jnp.ones((4, 197, 768))
y = attn(x)  # (4, 197, 768) -> (4, 197, 768)
```
