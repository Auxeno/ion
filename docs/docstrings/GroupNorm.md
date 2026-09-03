Group normalization ([Wu & He, 2018](https://arxiv.org/abs/1803.08494)).

Splits the channel dimension into `num_groups` groups and normalizes each group over its channels and a configurable number of trailing spatial dimensions, then applies a learnable per-channel scale and bias.

Parameters
----------
dim : int
    Number of channels. Must be divisible by `num_groups`.
num_groups : int
    Number of groups to split channels into. `num_groups=1` recovers
    LayerNorm over all channels; `num_groups=dim` gives instance norm.
num_spatial_dims : int
    Number of trailing spatial dimensions included in the group statistics.
    Match the data's spatial rank for images (the standard GroupNorm);
    `0` normalizes each position over its channel groups only.
eps : float, default=1e-5
    Positive constant added to the variance for numerical stability.
use_bias : bool, default=True
    Whether to learn a per-channel shift.

Attributes
----------
scale : Param
    Per-channel scale of shape `(dim,)`, initialized to ones.
b : Param | None
    Per-channel bias of shape `(dim,)`, initialized to zeros. `None` when `use_bias=False`.

Example
-------
```python
batch, height, width, channels = 8, 32, 32, 64
norm = nn.GroupNorm(channels, num_groups=8, num_spatial_dims=2)
x = jnp.ones((batch, height, width, channels))
y = norm(x)  # (8, 32, 32, 64) -> (8, 32, 32, 64)

norm = nn.GroupNorm(channels, num_groups=8, num_spatial_dims=0)
x = jnp.ones((batch, channels))
y = norm(x)  # (8, 64) -> (8, 64), per-position over channel groups

x_batched = jnp.ones((5, batch, height, width, channels))  # extra batch dim
grouped = nn.GroupNorm(channels, num_groups=8, num_spatial_dims=2)
y_batched = jax.vmap(grouped)(x_batched)  # (5, 8, 32, 32, 64) -> (5, 8, 32, 32, 64)
```

Note
-------
`num_spatial_dims` must match the number of trailing spatial dimensions in the input. A mismatch reduces over the wrong axes and silently produces wrong statistics rather than raising.
