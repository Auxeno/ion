Single-step Linear Recurrent Unit cell ([Orvieto et al., 2023](https://arxiv.org/abs/2303.06349)).

Applies one step of a complex diagonal linear recurrence, \(h' = \operatorname{diag}(\lambda)h + Bx\), with output \(y = \operatorname{Re}(Ch) + Dx\). Eigenvalues \(\lambda\) are parameterized in the log domain so their magnitudes stay in the stable range during training. Use `LRU` to scan a whole sequence.

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

Info
----
The output dimension equals `in_dim`.

Example
-------
```python
batch, in_dim, hidden_dim = 4, 3, 16
cell = nn.LRUCell(in_dim, hidden_dim, key=key)
x = jnp.ones((batch, in_dim))
h0 = jnp.zeros((batch, hidden_dim), dtype=jnp.complex64)
y, h = cell(x, h0)  # (4, 3), (4, 16) -> (4, 3), (4, 16)
```
