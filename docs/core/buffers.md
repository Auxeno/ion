# Buffers

Some layers, such as `BatchNorm`, update running values during a forward pass.
A `Buffer` holds one of those values inside the model and updates it in place.

::: ion.nn.Buffer
    options:
      show_docstring_description: false
      members:
        - value
        - set

---

## Using buffers

Buffers need no setup. A stateful layer builds its own, and is called exactly
like a stateless one:

```python
model = nn.Sequential(
    nn.Linear(3, 64, key=key),
    nn.BatchNorm(64),
    jax.nn.relu,
)

y = model(x, training=True)
y = model(x, training=False)
```

To inspect or use a buffer in a computation, read its `.value`. The buffer
itself is a state wrapper, not an array:

```python
running_mean = model[1].running_mean.value
```

Training updates the running values, evaluation reads them. Because a buffer
contributes no pytree leaves, `jax.grad` returns a tree with no buffer entries,
the optimizer allocates no state for them, and `model.astype` leaves their
dtype alone.

## Writing a stateful layer

Give the layer a `Buffer` field, read it with `.value`, and replace the stored
value with `.set`:

```python
class MeanCenter(nn.Module):
    running_mean: nn.Buffer
    momentum: float

    def __init__(self, dim, momentum=0.1):
        self.running_mean = nn.Buffer(jnp.zeros(dim))
        self.momentum = momentum

    def __call__(self, x, *, training):
        if training:
            axes = tuple(range(x.ndim - 1))
            mean = jnp.mean(x, axis=axes)
            self.running_mean.set(
                (1.0 - self.momentum) * self.running_mean.value
                + self.momentum * mean
            )
        else:
            mean = self.running_mean.value

        return x - mean
```

`set` applies `jax.lax.stop_gradient`, so a buffer update never contributes to
parameter gradients. A buffer therefore always reads back as a constant. Use the
value you computed in the maths the layer returns, not a read-back of the buffer
you just wrote, or gradients through it silently become zero. `BatchNorm`
normalizes with its local `mean`, not with `self.running_mean.value`, so
gradients still flow through the batch statistics.

A buffer keeps the dtype it was built with, and `astype` never casts it. This
lets a layer hold numerically sensitive state in float32 while the rest of the
model runs in `bfloat16`, but that internal dtype should not unexpectedly
promote the layer's output. Convert values back to the dtype the layer would
normally return at the boundary of the buffered calculation. Note that writes
are strict about dtype: a buffer's dtype is fixed for its lifetime.

## Buffers are mutable

A buffer is the one part of a model that changes in place, which means a model
owning one is not a plain value. Copies made with `jax.tree.map` share their
buffers with the original:

```python
copy = jax.tree.map(lambda leaf: leaf, model)  # shares running statistics
independent = model.clone()                    # owns its running statistics
```

`clone`, `freeze`, `unfreeze` and `load` give the model they return its own
buffers. `astype` is the exception: it shares them, which is what lets the
mixed-precision workflow cast inside the loss and still update the master
model's running statistics. `Optimizer.update` shares them too, since a step
continues one model rather than copying it.

`Module.clone` handles a model; `ion.clone` handles any pytree. `ion.is_buffer`
is the matching predicate for custom tree traversals:

::: ion.clone
    options:
      heading_level: 3

::: ion.is_buffer
    options:
      heading_level: 3

See [Stateful training](../workflows.md#stateful-training) for a complete
training step, and [Sharp edges](../sharp-edges.md) for the cases where
mutability shows through.
