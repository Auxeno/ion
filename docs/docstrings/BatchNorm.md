Batch normalization ([Ioffe & Szegedy, 2015](https://arxiv.org/abs/1502.03167)).

Normalizes each feature over all preceding dimensions. Training updates the
running mean and variance; evaluation uses them. These statistics are buffer
values, not trainable parameters.

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

Example
-------
```python
batch, dim = 8, 64
norm = nn.BatchNorm(dim)
buffers = norm.init_buffers()

x = jnp.ones((batch, dim))
y, buffers = norm(x, buffers, training=True)  # (8, 64) -> (8, 64), buffers updated
y, _ = norm(x, buffers, training=False)  # (8, 64), running statistics
```

Note
----
Running statistics are stored in float32. The normalized values return to the
input dtype before the learned affine transform. Buffers are not changed by
`Module.astype` or the optimizer.
