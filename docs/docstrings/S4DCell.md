Single-step S4D cell ([Gu et al., 2022](https://arxiv.org/abs/2206.11893)).

A diagonal state space model applied independently per input feature (single-input single-output). Each feature runs its own complex diagonal recurrence with a learnable timestep, discretized from continuous-time parameters. Use `S4D` to scan a whole sequence.

With diagonal \(A\), zero-order-hold discretization and the recurrent update are

\[
\begin{gathered}
\bar A = e^{A\Delta}, \qquad
\bar B = A^{-1}(\bar A-I), \qquad
h_t = \bar A \odot h_{t-1} + \bar B \odot x_t,
\\[4pt]
y_t = 2\operatorname{Re}\!\left(\sum_k C_k h_{t,k}\right) + D \odot x_t.
\end{gathered}
\]

The factor of two recovers the contribution from the conjugate eigenvalues
that are not stored explicitly. S4D applies this recurrence independently to
every input feature; `S4D` evaluates the same update with an associative scan.

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
    Initializer for the `C` output projection. Glorot uniform by default.
d_init : Initializer
    Initializer for the `D` skip term. Zeros by default.
key : jax.Array
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

Example
-------
```python
batch, in_dim, state_dim = 4, 3, 8
cell = nn.S4DCell(in_dim, state_dim, key=key)
x = jnp.ones((batch, in_dim))
h0 = jnp.zeros((batch, in_dim, state_dim // 2), dtype=jnp.complex64)
y, h = cell(x, h0)  # (4, 3), (4, 3, 4) -> (4, 3), (4, 3, 4)
```

Note
----
The output dimension equals `in_dim`.
