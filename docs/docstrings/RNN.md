Vanilla RNN over a full sequence.

Scans an `RNNCell` across the time axis, returning the output at every step and the final hidden state.

Parameters
----------
in_dim : int
    Input feature dimension.
hidden_dim : int
    Hidden state dimension.
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
cell : RNNCell
    The wrapped single-step cell holding the weights.

Example
-------
```python
batch, time, in_dim, hidden_dim = 4, 10, 3, 16
rnn = nn.RNN(in_dim, hidden_dim, key=key)
x = jnp.ones((batch, time, in_dim))
outputs, h = rnn(x)  # (4, 10, 3) -> (4, 10, 16), (4, 16)

x_batched = jnp.ones((5, batch, time, in_dim))  # extra batch dim
outputs, h = jax.vmap(rnn)(x_batched)  # (5, 4, 10, 3) -> (5, 4, 10, 16), (5, 4, 16)
```
