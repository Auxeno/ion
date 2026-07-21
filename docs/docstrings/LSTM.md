LSTM over a full sequence (Hochreiter & Schmidhuber, 1997).

Scans an `LSTMCell` across the time axis, returning the output at every step and the final `(h, c)` state.

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
    Bias initializer. Zeros by default, except the forget-gate slice, set to ones.
key : jax.Array
    RNG key for parameter initialization. Keyword-only.

Attributes
----------
cell : LSTMCell
    The wrapped single-step cell holding the weights.

Notes
-----
Pass `hx` as an `(h, c)` tuple to override the default zero initial state. See [Reference](../reference.md#recurrent-state).

Examples
--------
>>> lstm = nn.LSTM(3, 16, key=key)
>>> outputs, (h, c) = lstm(x)                # (b, t, 3) -> (b, t, 16), ((b, 16), (b, 16))
>>> outputs, (h, c) = lstm(x, hx=(h0, c0))   # custom initial state
