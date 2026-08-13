Single-step GRU cell ([Cho et al., 2014](https://arxiv.org/abs/1406.1078)).

Computes one timestep of the reset, update, and candidate gates, returning the new hidden state. Use `GRU` to scan a whole sequence.

Split the input and hidden projections into reset, update, and candidate parts:

\[
\begin{gathered}
(r_x,z_x,n_x)=x_tW_i+b_i, \qquad
(r_h,z_h,n_h)=h_{t-1}W_h+b_h,
\\[4pt]
r_t=\sigma(r_x+r_h), \qquad z_t=\sigma(z_x+z_h), \qquad
n_t=\tanh(n_x+r_t\odot n_h),
\\[4pt]
h_t=(1-z_t)\odot n_t+z_t\odot h_{t-1}.
\end{gathered}
\]

Parameters
----------
in_dim : int
    Input feature dimension.
hidden_dim : int
    Hidden state dimension. The three gates share a `3 * hidden_dim` projection.
use_bias : bool, default=True
    Whether to include a learnable bias term.
w_i_init : Initializer
    Input-to-hidden initializer. Glorot uniform by default.
w_h_init : Initializer
    Hidden-to-hidden initializer. Orthogonal by default.
b_init : Initializer
    Bias initializer. Zeros by default.
key : jax.Array
    RNG key for parameter initialization. Keyword-only.

Attributes
----------
w_i : Param
    Input-to-gates weight of shape `(in_dim, 3 * hidden_dim)`.
w_h : Param
    Hidden-to-gates weight of shape `(hidden_dim, 3 * hidden_dim)`.
b : Param | None
    Input-side gate bias of shape `(3 * hidden_dim,)`. `None` when `use_bias=False`.
b_h : Param | None
    Hidden-side gate bias, kept separate so the candidate gate matches the
    reference GRU formulation exactly. `None` when `use_bias=False`.

Example
-------
```python
batch, in_dim, hidden_dim = 4, 3, 16
cell = nn.GRUCell(in_dim, hidden_dim, key=key)
x = jnp.ones((batch, in_dim))
h0 = jnp.zeros((batch, hidden_dim))
h = cell(x, h0)  # (4, 3), (4, 16) -> (4, 16)
```
