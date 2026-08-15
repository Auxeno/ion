Build and evaluate independently initialized modules together.

Parameters
----------
factory : Callable[[jax.Array], Module]
    Function that constructs one module from one RNG key.
size : int
    Number of ensemble members.
reduction : str | None, default=None
    Output reduction: `None`, `"mean"`, or `"sum"`. `None` retains the leading member axis.
key : jax.Array
    RNG key split across member construction. Keyword-only.

Attributes
----------
models : Module
    Module with parameters stacked over the leading member axis.
size : int
    Number of ensemble members.
reduction : str | None
    Output reduction.

Example
-------
```python
ensemble = nn.Ensemble(
    lambda key: nn.MLP([3, 64, 1], key=key),
    4,
    key=key,
)
x = jnp.ones((32, 3))
y = ensemble(x)  # (4, 32, 1)
```

Note
----
Member construction and evaluation use `jax.vmap`. Factories must therefore build stateless modules without `Buffer` fields.
