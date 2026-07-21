Single-step GRU cell (Cho et al., 2014).

Computes one timestep of the reset, update, and candidate gates, returning the new hidden state. Use `GRU` to scan a whole sequence.

Parameters
----------
in_dim : int
    Input feature dimension.
hidden_dim : int
    Hidden state dimension. The three gates share a `3 * hidden_dim` projection.
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
w_i : Param
    Input-to-gates weight of shape `(in_dim, 3 * hidden_dim)`.
w_h : Param
    Hidden-to-gates weight of shape `(hidden_dim, 3 * hidden_dim)`.
b : Param | None
    Input-side gate bias of shape `(3 * hidden_dim,)`. `None` when `bias=False`.
b_h : Param | None
    Hidden-side gate bias, kept separate so the candidate gate matches the
    reference GRU formulation exactly. `None` when `bias=False`.

Notes
-----
The `initial_state` property returns a zero hidden state of the right shape.

Examples
--------
>>> cell = nn.GRUCell(3, 16, key=key)
>>> h = cell(x, cell.initial_state)  # (*, 3), (*, 16) -> (*, 16)
