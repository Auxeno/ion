Root mean square normalization over the last dimension ([Zhang & Sennrich, 2019](https://arxiv.org/abs/1910.07467)).

Rescales each input vector by its root mean square without subtracting the mean, then applies a learnable elementwise scale. Cheaper than `LayerNorm` and common in modern transformers.

\[
\operatorname{RMSNorm}(x)
= \gamma \odot \frac{x}{\sqrt{\frac{1}{d}\sum_{k=1}^d x_k^2+\epsilon}}.
\]

Parameters
----------
dim : int
    Size of the feature dimension to normalize.
eps : float, default=1e-5
    Positive constant added inside the square root for numerical stability.

Attributes
----------
scale : Param
    Elementwise scale of shape `(dim,)`, initialized to ones.

Example
-------
```python
norm = nn.RMSNorm(64)
x = jnp.ones((4, 16, 64))
y = norm(x)  # (4, 16, 64) -> (4, 16, 64)
```
