Chain layers and callables, applying them in order.

Each layer's output feeds the next. `training` and split random keys are
forwarded to layers that accept them.

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
keys = jax.random.split(key, 2)
model = nn.Sequential(
    nn.Linear(3, 16, key=keys[0]),
    jax.nn.relu,
    nn.Linear(16, 1, key=keys[1]),
)
x = jnp.ones((32, 3))
y = model(x)  # (32, 3) -> (32, 1)
```

Layers that need a training mode or a random key receive them through the call:

```python
keys = jax.random.split(key, 3)
model = nn.Sequential(
    nn.Conv(3, 16, kernel_shape=(3, 3), padding=1, key=keys[0]),
    jax.nn.leaky_relu,
    nn.Dropout(0.1),
    lambda x: x.mean(axis=(1, 2)),
    nn.Linear(16, 10, key=keys[1]),
)
x = jnp.ones((8, 28, 28, 3))
y = model(x, training=True, key=keys[2])  # (8, 28, 28, 3) -> (8, 10)
```
