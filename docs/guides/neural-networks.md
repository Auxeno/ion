# Neural Networks

A walkthrough of building and training a model with `ion.nn`. For the cross-cutting rules that every layer follows (input format, shape labels, initialization) see the [NN Reference](../nn/reference.md); for the individual layers see [Layers](../nn/index.md).

## Define a model from scratch

A model is a [`Module`](../core/module.md): declare fields as class annotations, assign them in `__init__`, and implement `__call__`. Weights are [`Param`](../core/param.md)s, which behave like arrays in the forward pass:

```python
import jax
import jax.numpy as jnp
import ion.nn as nn

class Linear(nn.Module):
    w: nn.Param
    b: nn.Param

    def __init__(self, in_dim, out_dim, *, key):
        self.w = nn.Param(jax.random.normal(key, (in_dim, out_dim)))
        self.b = nn.Param(jnp.zeros((out_dim,)))

    def __call__(self, x):
        return x @ self.w + self.b

model = Linear(3, 16, key=jax.random.key(0))
y = model(jnp.ones((32, 3)))  # (32, 3) -> (32, 16)
```

Inheriting from `Module` registers the class as a JAX pytree, the model is immutable after `__init__`, and `@`/`+` work transparently on Params through the `__jax_array__` protocol.

## Use built-in layers

In practice you compose the shipped layers rather than writing every weight by hand:

```python
class MLP(nn.Module):
    layer_1: nn.Linear
    layer_2: nn.Linear

    def __init__(self, *, key):
        keys = jax.random.split(key, 2)
        self.layer_1 = nn.Linear(784, 128, key=keys[0])
        self.layer_2 = nn.Linear(128, 10, key=keys[1])

    def __call__(self, x):
        return self.layer_2(jax.nn.relu(self.layer_1(x)))

model = MLP(key=jax.random.key(0))
```

`nn.MLP` and `nn.Sequential` assemble common stacks for you when you do not need a custom `__call__`.

## Inspect it

```python
>>> model
MLP(
  layer_1=Linear(
    w=Param(f32[784, 128], trainable=True),
    b=Param(f32[128], trainable=True),
  ),
  layer_2=Linear(
    w=Param(f32[128, 10], trainable=True),
    b=Param(f32[10], trainable=True),
  ),
)

>>> model.num_params
101770
```

In notebooks, `ion.enable_treescope()` gives an interactive colour-coded view (on by default).

## Train it

The loss takes the model first so `jax.grad` differentiates with respect to it. Wrap an optax transform in an [`Optimizer`](../core/optimizer.md); it returns the updated model and a new optimizer with an incremented step:

```python
import optax
import ion

def loss_fn(model, x, y):
    logits = model(x)
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

@jax.jit
def train_step(model, optimizer, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(model, x, y)
    model, optimizer = optimizer.update(model, grads)
    return model, optimizer, loss

model = MLP(key=jax.random.key(0))
optimizer = ion.Optimizer(optax.adam(3e-4), model)

for x, y in data:
    model, optimizer, loss = train_step(model, optimizer, x, y)
```

Both the model and optimizer are pytrees, so the whole step flows through `jax.jit` naturally.

## Common variations

**Stochastic layers** such as [Dropout](../nn/layers/dropout.md) take an RNG key at call time, so thread one through the loss:

```python
def loss_fn(model, x, y, key):
    logits = model(x, key=key)
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
```

**Gradient clipping** and other transforms compose with `optax.chain`:

```python
optimizer = ion.Optimizer(
    optax.chain(optax.clip_by_global_norm(1.0), optax.adam(3e-4)),
    model,
)
```

**Fully compiled loops** (for example RL inner loops) use `jax.lax.scan`, since the model and optimizer are valid scan carries:

```python
def scan_step(carry, batch):
    model, optimizer = carry
    x, y = batch
    loss, grads = jax.value_and_grad(loss_fn)(model, x, y)
    model, optimizer = optimizer.update(model, grads)
    return (model, optimizer), loss

(model, optimizer), losses = jax.lax.scan(scan_step, (model, optimizer), batches)
```

## Next

- [Freezing](freezing.md) to fine-tune part of a model or use LoRA.
- [Mixed Precision](mixed-precision.md) for bfloat16 training and inference.
- [Checkpointing](checkpointing.md) to save and load models.
- [Examples](../examples/index.md) for complete projects.
