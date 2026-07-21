Root mean square normalization over the last dimension (Zhang & Sennrich, 2019).

Rescales each input vector by its root mean square without subtracting the mean, then applies a learnable elementwise scale. Cheaper than `LayerNorm` and common in modern transformers.

Parameters
----------
dim : int
    Size of the feature dimension to normalize.
eps : float, default=1e-5
    Constant added inside the square root for numerical stability.

Attributes
----------
scale : Param
    Elementwise scale of shape `(dim,)`, initialized to ones.

Notes
-----
No mean centering and no bias term, the two differences from `LayerNorm`. Operates on the last dimension only, so any number of leading batch dimensions is supported.

Examples
--------
>>> norm = nn.RMSNorm(64)
>>> y = norm(x)  # (*, 64) -> (*, 64)
