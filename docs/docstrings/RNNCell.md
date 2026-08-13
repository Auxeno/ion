Single-step vanilla RNN cell.

Computes one recurrent update:

\[
h_t = \tanh(x_tW_i + h_{t-1}W_h + b).
\]

Use `RNN` to scan a whole sequence; use the cell directly for custom loops or
when threading state across chunks.

Parameters
----------
in_dim : int
    Input feature dimension.
hidden_dim : int
    Hidden state dimension.
use_bias : bool, default=True
    Whether to include a learnable bias term.
w_i_init : Initializer
    Input-to-hidden initializer. Glorot uniform by default.
w_h_init : Initializer
    Hidden-to-hidden initializer. Orthogonal by default, which preserves
    gradient norms across time steps.
b_init : Initializer
    Bias initializer. Zeros by default.
key : jax.Array
    RNG key for parameter initialization. Keyword-only.

Attributes
----------
w_i : Param
    Input-to-hidden weight of shape `(in_dim, hidden_dim)`.
w_h : Param
    Hidden-to-hidden weight of shape `(hidden_dim, hidden_dim)`.
b : Param | None
    Bias of shape `(hidden_dim,)`. `None` when `use_bias=False`.

Example
-------
```python
batch, in_dim, hidden_dim = 4, 3, 16
cell = nn.RNNCell(in_dim, hidden_dim, key=key)
x = jnp.ones((batch, in_dim))
h0 = jnp.zeros((batch, hidden_dim))
h = cell(x, h0)  # (4, 3), (4, 16) -> (4, 16)
```
