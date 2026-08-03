Spectral normalization ([Miyato et al., 2018](https://arxiv.org/abs/1802.05957)).

Divides a module parameter by its largest singular value, estimated with power
iteration.

Parameters
----------
module : Module
    Module to wrap.
parameter : str, default='w'
    `Param` field to normalize. Must have rank 2 or greater.
power_iterations : int, default=1
    Power iterations per training call.
eps : float, default=1e-12
    Minimum norm and singular value.
key : jax.Array
    Random key for the initial power-iteration vectors.

Attributes
----------
module : Module
    Wrapped module.
u : Buffer
    Left power-iteration vector.
v : Buffer
    Right power-iteration vector.

Example
-------
```python
batch, in_dim, out_dim = 8, 64, 128
key_linear, key_vectors = jax.random.split(key)

layer = nn.SpectralNorm(nn.Linear(in_dim, out_dim, key=key_linear), key=key_vectors)
x = jnp.ones((batch, in_dim))
y = layer(x, training=True)  # (8, 64) -> (8, 128), vectors refined
y = layer(x, training=False)  # (8, 128), vectors held fixed
```

Note
----
The final parameter dimension is the output dimension. Earlier dimensions are
flattened, supporting linear and convolutional weights. Power-iteration vectors
use the wrapped parameter dtype at construction. The normalized parameter
returns to its original dtype before calling the wrapped module.
