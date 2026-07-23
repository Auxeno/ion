Layer normalization over the last dimension ([Ba et al., 2016](https://arxiv.org/abs/1607.06450)).

Normalizes each input vector to zero mean and unit variance across its features, then applies a learnable elementwise scale and bias.

Parameters
----------
dim : int
    Size of the feature dimension to normalize.
eps : float, default=1e-5
    Constant added to the variance for numerical stability.

Attributes
----------
scale : Param
    Elementwise scale of shape `(dim,)`, initialized to ones.
b : Param
    Elementwise bias of shape `(dim,)`, initialized to zeros.

Info
----
Operates on the last dimension only, so any number of leading batch dimensions is supported.

Example
-------
```python
norm = nn.LayerNorm(64)
y = norm(x)  # (*, 64) -> (*, 64)
```
