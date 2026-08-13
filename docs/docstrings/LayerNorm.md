Layer normalization over the last dimension ([Ba et al., 2016](https://arxiv.org/abs/1607.06450)).

Normalizes each input vector to zero mean and unit variance across its features, then applies a learnable elementwise scale and bias.

\[
\operatorname{LayerNorm}(x)
= \gamma \odot \frac{x-\mu}{\sqrt{\sigma^2+\epsilon}} + \beta,
\qquad
\mu=\frac{1}{d}\sum_{k=1}^d x_k.
\]

Parameters
----------
dim : int
    Size of the feature dimension to normalize.
eps : float, default=1e-5
    Positive constant added to the variance for numerical stability.
use_bias : bool, default=True
    Whether to learn an elementwise shift.

Attributes
----------
scale : Param
    Elementwise scale of shape `(dim,)`, initialized to ones.
b : Param | None
    Elementwise bias of shape `(dim,)`, initialized to zeros. `None` when `use_bias=False`.

Example
-------
```python
norm = nn.LayerNorm(64)
x = jnp.ones((4, 16, 64))
y = norm(x)  # (4, 16, 64) -> (4, 16, 64)
```
