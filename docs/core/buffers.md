# Buffer

`Buffer` holds non-trainable array state that a module mutates during a forward pass, such as BatchNorm running statistics. This lets stateful modules use the same call and training interfaces as stateless ones.

::: ion.nn.Buffer
    options:
      members:
        - value
        - set

---

!!! important
    Unlike a PyTorch buffer, an Ion `Buffer` is not general-purpose storage for non-trainable arrays. Use a normal JAX array unless the forward pass must mutate the value.

## Using stateful layers

Stateful layers construct and manage their own buffers:

```python
model = nn.Sequential(
    nn.Linear(3, 64, key=key),
    nn.BatchNorm(64),
    jax.nn.relu,
)

y = model(x, training=True)   # updates running statistics
y = model(x, training=False)  # reads running statistics
```

There is no separate state argument or return value, so the ordinary [training step](../nn/guide.md#training-the-model) works unchanged. Read a buffer through `.value`:

```python
running_mean = model[1].running_mean.value
```

## Defining a buffer

```python
class Counter(nn.Module):
    count: nn.Buffer

    def __init__(self):
        self.count = nn.Buffer(jnp.array(0))

    def __call__(self, x):
        self.count.set(self.count.value + 1)
        return x
```

Buffers have a fixed shape and dtype, contribute no pytree leaves, and are not cast or updated by an optimizer. `set` also applies `stop_gradient`.

Because buffers are mutable, ordinary pytree copies share their state. Use `model.clone()` for an independent copy. See [Sharp edges](../sharp-edges.md#models-with-buffers-are-not-plain-values) for copying and JAX transform constraints.
