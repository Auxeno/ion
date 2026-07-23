Linear Recurrent Unit over a full sequence ([Orvieto et al., 2023](https://arxiv.org/abs/2303.06349)).

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
    Initializer for the `B` and `C` projections. Glorot uniform by default.
d_init : Initializer
    Initializer for the `D` skip term. Zeros by default.
key : jax.Array
    RNG key for parameter initialization. Keyword-only.

Attributes
----------
cell : LRUCell
    The wrapped single-step cell holding the parameters.

Example
-------
```python
batch, time, in_dim, hidden_dim = 4, 10, 3, 16
lru = nn.LRU(in_dim, hidden_dim, key=key)
x = jnp.ones((batch, time, in_dim))
outputs, h = lru(x)  # (4, 10, 3) -> (4, 10, 3), (4, 16)

x_batched = jnp.ones((5, batch, time, in_dim))  # extra batch dim
outputs, h = jax.vmap(lru)(x_batched)  # (5, 4, 10, 3) -> (5, 4, 10, 3), (5, 4, 16)
```
