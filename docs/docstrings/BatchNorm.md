Batch normalization ([Ioffe & Szegedy, 2015](https://arxiv.org/abs/1502.03167)).

Normalizes each feature over all preceding dimensions. Training updates the
running mean and variance; evaluation uses them. These statistics are buffers,
not trainable parameters.

Parameters
----------
dim : int
    Size of the feature dimension to normalize.
momentum : float, default=0.1
    Weight given to the current batch statistics.
eps : float, default=1e-5
    Value added to the variance.
bias : bool, default=True
    Whether to use a bias.

Attributes
----------
scale : Param
    Scale of shape `(dim,)`.
b : Param | None
    Bias of shape `(dim,)`, or `None` when disabled.
running_mean : Buffer
    Running mean of shape `(dim,)`.
running_var : Buffer
    Running variance of shape `(dim,)`.

Example
-------
```python
batch, dim = 8, 64
norm = nn.BatchNorm(dim)

x = jnp.ones((batch, dim))
y = norm(x, training=True)  # (8, 64) -> (8, 64), running statistics updated
y = norm(x, training=False)  # (8, 64), running statistics
```

Note
----
Running statistics are stored in float32. The normalized values return to the
input dtype before the learned affine transform. Buffers are not changed by
`Module.astype` or the optimizer.
