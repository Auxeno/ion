GRU over a full sequence ([Cho et al., 2014](https://arxiv.org/abs/1406.1078)).

Scans a `GRUCell` across the time axis, returning the output at every step and the final hidden state.

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
    Hidden-to-hidden initializer. Orthogonal by default.
b_init : Initializer
    Bias initializer. Zeros by default.
key : jax.Array
    RNG key for parameter initialization. Keyword-only.

Attributes
----------
cell : GRUCell
    The wrapped single-step cell holding the weights.

Notes
-----
Pass `hx` to override the default zero initial state. See [Reference](../reference.md#recurrent-state).

Examples
--------
>>> gru = nn.GRU(3, 16, key=key)
>>> outputs, h = gru(x)          # (b, t, 3) -> (b, t, 16), (b, 16)
>>> outputs, h = gru(x, hx=h0)   # custom initial state
