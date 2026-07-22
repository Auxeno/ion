Single-step Linear Recurrent Unit cell ([Orvieto et al., 2023](https://arxiv.org/abs/2303.06349)).

Applies one step of a complex diagonal linear recurrence, `h' = diag(lambda) h + B x`, with output `y = Re(C h) + D x`. Eigenvalues `lambda` are parameterized in the log domain so their magnitudes stay in the stable range during training. Use `LRU` to scan a whole sequence.

Parameters
----------
in_dim : int
    Input and output feature dimension.
hidden_dim : int
    Number of complex state eigenvalues. Stored independently, without the
    conjugate-pair symmetry used by `S4D`/`S5`.
r_min : float, default=0.0
    Lower bound on the initial eigenvalue magnitudes.
r_max : float, default=1.0
    Upper bound on the initial eigenvalue magnitudes.
max_phase : float, default=2 * pi
    Upper bound on the initial eigenvalue phases.
w_init : Initializer
    Initializer for the `B` and `C` projections. Glorot uniform by default.
d_init : Initializer
    Initializer for the `D` skip term. Zeros by default.
key : jax.Array
    RNG key for parameter initialization. Keyword-only.

Attributes
----------
nu_log, theta_log : Param
    Log-domain magnitude and phase of the diagonal eigenvalues, each `(hidden_dim,)`.
gamma_log : Param
    Log-domain input normalization, `(hidden_dim,)`.
B, C : Param
    Complex input and output projections.
D : Param
    Real per-feature skip connection.

Notes
-----
The output dimension equals `in_dim`. State is complex; `initial_state` returns a zero complex vector of shape `(hidden_dim,)`. See [Reference](../reference.md#ssm).

Examples
--------
>>> cell = nn.LRUCell(3, 16, key=key)
>>> y, h = cell(x, cell.initial_state)  # (*, 3), (*, 16) -> (*, 3), (*, 16)
