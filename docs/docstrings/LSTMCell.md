Single-step LSTM cell ([Hochreiter & Schmidhuber, 1997](https://doi.org/10.1162/neco.1997.9.8.1735)).

Computes one timestep of the input, forget, cell, and output gates, returning the new hidden and cell states `(h, c)`. Use `LSTM` to scan a whole sequence.

Parameters
----------
in_dim : int
    Input feature dimension.
hidden_dim : int
    Hidden state dimension. The four gates share a `4 * hidden_dim` projection.
bias : bool, default=True
    Whether to include a learnable bias term.
w_i_init : Initializer
    Input-to-hidden initializer. Glorot uniform by default.
w_h_init : Initializer
    Hidden-to-hidden initializer. Orthogonal by default.
b_init : Initializer
    Bias initializer. Zeros by default, except the forget-gate slice, which is
    set to ones to encourage remembering early in training.
key : jax.Array
    RNG key for parameter initialization. Keyword-only.

Attributes
----------
w_i : Param
    Input-to-gates weight of shape `(in_dim, 4 * hidden_dim)`.
w_h : Param
    Hidden-to-gates weight of shape `(hidden_dim, 4 * hidden_dim)`.
b : Param | None
    Gate bias of shape `(4 * hidden_dim,)`. `None` when `bias=False`.

Example
-------
```python
batch, in_dim, hidden_dim = 4, 3, 16
cell = nn.LSTMCell(in_dim, hidden_dim, key=key)
x = jnp.ones((batch, in_dim))
hx = (jnp.zeros((batch, hidden_dim)), jnp.zeros((batch, hidden_dim)))
h, c = cell(x, hx)  # (4, 3), ((4, 16), (4, 16)) -> ((4, 16), (4, 16))
```
