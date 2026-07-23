Fully connected linear layer computing `x @ w + b`.

Parameters
----------
in_dim : int
    Input feature dimension.
out_dim : int
    Output feature dimension.
bias : bool, default=True
    Whether to include a learnable bias term.
w_init : Initializer
    Weight initializer. Glorot uniform by default.
b_init : Initializer
    Bias initializer. Zeros by default.
key : jax.Array
    RNG key for parameter initialization. Keyword-only.

Attributes
----------
w : Param
    Weight matrix of shape `(in_dim, out_dim)`.
b : Param | None
    Bias vector of shape `(out_dim,)`. `None` when `bias=False`.

Example
-------
```python
batch, in_dim, out_dim = 10, 3, 16
linear = nn.Linear(in_dim, out_dim, key=key)

x = jnp.ones((batch, in_dim))
y = linear(x)  # (10, 3) -> (10, 16)

x = jnp.ones((5, batch, in_dim))  # extra batch dim
y = linear(x)  # (5, 10, 3) -> (5, 10, 16)
```
