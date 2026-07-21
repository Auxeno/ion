Single-step S5 cell (Smith et al., 2023).

A diagonal state space model with a single state shared across all features (multi-input multi-output), in contrast to `S4D`'s per-feature states. One complex diagonal recurrence mixes every input channel through `B` and reads out through `C`. Use `S5` to scan a whole sequence.

Parameters
----------
in_dim : int
    Input and output feature dimension.
state_dim : int
    Shared state size. Stored with conjugate-pair symmetry, so `state_dim` is
    represented by `state_dim // 2` complex eigenvalues.
dt_min : float, default=0.001
    Lower bound of the initial timestep range (sampled log-uniform).
dt_max : float, default=0.1
    Upper bound of the initial timestep range.
w_init : Initializer
    Initializer for the `B` and `C` projections. Glorot normal by default.
d_init : Initializer
    Initializer for the `D` skip term. Zeros by default.
key : jax.Array
    RNG key for parameter initialization. Keyword-only.

Attributes
----------
A_log_re, A_im : Param
    Log-domain real part and imaginary part of the diagonal state matrix.
B, C : Param
    Complex input and output projections coupling all features to the state.
D : Param
    Real per-feature skip connection.
log_dt : Param
    Log-domain discretization timestep.

Notes
-----
The output dimension equals `in_dim`. State is complex with shape `(state_dim // 2,)`; `initial_state` returns it zeroed. See [Reference](../reference.md#ssm).

Examples
--------
>>> cell = nn.S5Cell(3, 8, key=key)
>>> y, h = cell(x, cell.initial_state)  # (*, 3), (*, 4) -> (*, 3), (*, 4)
