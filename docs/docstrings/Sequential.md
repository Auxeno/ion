Chain layers and callables, applying them in order.

Each layer's output feeds the next. `training`, `buffers`, and split random keys
are forwarded to layers that accept them.

Parameters
----------
*layers : Callable
    The layers or callables to apply in order. Any callable taking one value
    and returning one value works, not only `Module`s.

Attributes
----------
layers : tuple[Callable, ...]
    The chained layers, in order.

Example
-------
```python
batch, in_dim, hidden_dim, out_dim = 32, 3, 16, 1
key_1, key_2, key_dropout = jax.random.split(key, 3)

model = nn.Sequential(
    nn.Linear(in_dim, hidden_dim, key=key_1),
    nn.Dropout(0.1),
    nn.Linear(hidden_dim, out_dim, key=key_2),
)
x = jnp.ones((batch, in_dim))
y = model(x, training=True, key=key_dropout)  # (32, 3) -> (32, 1)
```
