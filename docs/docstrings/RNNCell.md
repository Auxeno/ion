Single-step vanilla RNN cell.

Computes one timestep, `h' = tanh(x W_i + h W_h + b)`. Use `RNN` to scan a whole sequence; use the cell directly for custom loops or when threading state across chunks.

Parameters
----------
in_dim : int
    Input feature dimension.
hidden_dim : int
    Hidden state dimension.
bias : bool, default=True
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
    Bias of shape `(hidden_dim,)`. `None` when `bias=False`.

Info
----
The `initial_state` property returns a zero hidden state of the right shape.

Example
-------
```python
cell = nn.RNNCell(3, 16, key=key)
h = cell(x, cell.initial_state)  # (*, 3), (*, 16) -> (*, 16)
```
