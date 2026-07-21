Linear Recurrent Unit over a full sequence (Orvieto et al., 2023).

Runs an `LRUCell` over the time axis with an associative parallel scan, so the whole sequence is processed in parallel rather than stepped serially. Returns the output at every step and the final complex state.

Parameters
----------
in_dim : int
    Input and output feature dimension.
hidden_dim : int
    Number of complex state eigenvalues.
r_min : float, default=0.0
    Lower bound on the initial eigenvalue magnitudes.
r_max : float, default=1.0
    Upper bound on the initial eigenvalue magnitudes.
max_phase : float, default=2 * pi
    Upper bound on the initial eigenvalue phases.
w_init : Initializer
    Initializer for the `B` and `C` projections. Glorot normal by default.
d_init : Initializer
    Initializer for the `D` skip term. Zeros by default.
key : PRNGKeyArray
    RNG key for parameter initialization. Keyword-only.

Attributes
----------
cell : LRUCell
    The wrapped single-step cell holding the parameters.

Notes
-----
The output dimension equals `in_dim`. Pass `hx` to override the default zero initial state. See [Conventions](../conventions.md#ssm).

Examples
--------
>>> lru = nn.LRU(3, 16, key=key)
>>> outputs, h = lru(x)          # (b, t, 3) -> (b, t, 3), (b, 16)
>>> outputs, h = lru(x, hx=h0)   # custom initial state
