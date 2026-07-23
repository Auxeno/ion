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
    Constant added to the variance for numerical stability.

Attributes
----------
scale : Param
    Per-channel scale of shape `(dim,)`, initialized to ones.
b : Param
    Per-channel bias of shape `(dim,)`, initialized to zeros.

Info
----
Channels are last. Takes no `key`: scale and bias are initialized deterministically.

Warning
-------
`num_spatial_dims` must match the number of trailing spatial dimensions in the input. A mismatch reduces over the wrong axes and silently produces wrong statistics rather than raising.

Example
-------
```python
norm = nn.GroupNorm(64, num_groups=8, num_spatial_dims=2)
y = norm(x)  # (b, h, w, 64) -> (b, h, w, 64)

norm = nn.GroupNorm(64, num_groups=8, num_spatial_dims=0)
y = norm(x)  # (b, 64) -> (b, 64), per-position over channel groups
```
