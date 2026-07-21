Vanilla RNN over a full sequence.

Scans an `RNNCell` across the time axis, returning the output at every step and the final hidden state.

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
key : PRNGKeyArray
    RNG key for parameter initialization. Keyword-only.

Attributes
----------
cell : RNNCell
    The wrapped single-step cell holding the weights.

Notes
-----
Pass `hx` to override the default zero initial state, for example when continuing a sequence across chunks. See [Conventions](../conventions.md#recurrent-state).

Examples
--------
>>> rnn = nn.RNN(3, 16, key=key)
>>> outputs, h = rnn(x)          # (b, t, 3) -> (b, t, 16), (b, 16)
>>> outputs, h = rnn(x, hx=h0)   # custom initial state
