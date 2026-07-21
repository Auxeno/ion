Single-step S4D cell (Gu et al., 2022).

A diagonal state space model applied independently per input feature (single-input single-output). Each feature runs its own complex diagonal recurrence with a learnable timestep, discretized from continuous-time parameters. Use `S4D` to scan a whole sequence.

Parameters
----------
in_dim : int
    Input and output feature dimension.
state_dim : int
    State size per feature. Stored with conjugate-pair symmetry, so `state_dim`
    is represented by `state_dim // 2` complex eigenvalues.
dt_min : float, default=0.001
    Lower bound of the initial timestep range (sampled log-uniform).
dt_max : float, default=0.1
    Upper bound of the initial timestep range.
w_init : Initializer
    Initializer for the `C` output projection. Glorot normal by default.
d_init : Initializer
    Initializer for the `D` skip term. Zeros by default.
key : PRNGKeyArray
    RNG key for parameter initialization. Keyword-only.

Attributes
----------
A_log_re, A_im : Param
    Log-domain real part and imaginary part of the diagonal state matrix.
C : Param
    Complex output projection.
D : Param
    Real per-feature skip connection.
log_dt : Param
    Log-domain discretization timestep, one per feature.

Notes
-----
The output dimension equals `in_dim`. State is complex with shape `(in_dim, state_dim // 2)`; `initial_state` returns it zeroed. See [Conventions](../conventions.md#ssm).

Examples
--------
>>> cell = nn.S4DCell(3, 8, key=key)
>>> y, h = cell(x, cell.initial_state)  # (*, 3), (*, 3, 4) -> (*, 3), (*, 3, 4)
