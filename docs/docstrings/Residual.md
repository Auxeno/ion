Add a layer's output to its input.

Parameters
----------
layer : Callable
    Layer or callable whose output has the same shape as its input.

Attributes
----------
layer : Callable
    Wrapped layer or callable.

Example
-------
```python
model = nn.Residual(nn.Sequential(
    nn.Linear(16, 16, key=key),
    jax.nn.relu,
))
x = jnp.ones((32, 16))
y = model(x)  # (32, 16)
```

A ResNet-style block composes naturally from nested `Sequential` modules:

```python
keys = jax.random.split(key, 2)
block = nn.Sequential(
    nn.Residual(nn.Sequential(
        nn.Conv(16, 16, kernel_shape=(3, 3), padding=1, key=keys[0]),
        nn.BatchNorm(16),
        jax.nn.relu,
        nn.Conv(16, 16, kernel_shape=(3, 3), padding=1, key=keys[1]),
        nn.BatchNorm(16),
    )),
    jax.nn.relu,
)
x = jnp.ones((8, 32, 32, 16))
y = block(x, training=True)  # (8, 32, 32, 16)
```
