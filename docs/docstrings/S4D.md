S4D over a full sequence ([Gu et al., 2022](https://arxiv.org/abs/2206.11893)).

Runs an `S4DCell` over the time axis with an associative parallel scan. Each input feature is processed by its own diagonal state space model (single-input single-output). Returns the output at every step and the final complex state.

Parameters
----------
in_dim : int
    Input and output feature dimension.
state_dim : int
    State size per feature, stored as `state_dim // 2` conjugate-pair eigenvalues.
dt_min : float, default=0.001
    Lower bound of the initial timestep range (sampled log-uniform).
dt_max : float, default=0.1
    Upper bound of the initial timestep range.
w_init : Initializer
    Initializer for the `C` output projection. Glorot uniform by default.
d_init : Initializer
    Initializer for the `D` skip term. Zeros by default.
key : jax.Array
    RNG key for parameter initialization. Keyword-only.

Attributes
----------
cell : S4DCell
    The wrapped single-step cell holding the parameters.

Notes
-----
The output dimension equals `in_dim`. Pass `hx` to override the default zero initial state. See [Reference](../reference.md#ssm).

Examples
--------
>>> s4d = nn.S4D(3, 8, key=key)
>>> outputs, h = s4d(x)          # (b, t, 3) -> (b, t, 3), (b, 3, 4)
>>> outputs, h = s4d(x, hx=h0)   # custom initial state
