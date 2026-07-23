# Freezing

Ion's trainability system makes it straightforward to freeze parts of a model for fine-tuning or transfer learning. Freezing is a property of each [`Param`](../core/param.md): a frozen param still participates in the forward pass, but `jax.grad` returns a zero gradient for it and the [`Optimizer`](../core/optimizer.md) skips it entirely.

## Basic freezing

Freeze or unfreeze every param in a model. Both return a new model and leave the original untouched.

```python
frozen = model.freeze()
trainable = model.unfreeze()
```

## Selective freezing

Target a submodule with [`at`](../core/module.md), then set it back:

```python
# Freeze just the encoder
model = model.at.encoder.set(model.encoder.freeze())

# Freeze everything except the head
model = model.freeze()
model = model.at.head.set(model.head.unfreeze())
```

## How it works end to end

1. **Forward pass:** frozen params compute as normal, but `stop_gradient` marks them as constants.
2. **Backward pass:** `jax.grad` returns zero gradients at frozen positions, which XLA optimises away.
3. **Update step:** the `Optimizer` partitions frozen params with `optax.set_to_zero`, so no momentum or variance buffers are allocated for them.

```python
# Fine-tune only the classifier
model = model.freeze()
model = model.at.classifier.set(model.classifier.unfreeze())

# Only classifier params get optimizer state
optimizer = ion.Optimizer(optax.adam(3e-4), model)

@jax.jit
def train_step(model, optimizer, x, y):
    grads = jax.grad(loss_fn)(model, x, y)
    model, optimizer = optimizer.update(model, grads)
    return model, optimizer
```

## Change trainability before building the optimizer

Freeze or unfreeze first, then construct the optimizer, never the reverse. The `Optimizer` bakes the frozen/trainable partition into its state at construction, so a later `freeze()`/`unfreeze()` invalidates that state and `update()` raises:

```python
optimizer = ion.Optimizer(optax.adam(3e-4), model)
model = model.unfreeze()
optimizer.update(model, grads)
# ValueError: Model structure or trainability changed, create a new Optimizer.
```

The fix is one line: build a fresh optimizer from the updated model. This resets momentum buffers, which is the correct semantics, since newly unfrozen params have no gradient history and staged-unfreezing schedules conventionally restart optimizer state at each stage.

```python
model = model.unfreeze()
optimizer = ion.Optimizer(optax.adam(3e-4), model)
```

## LoRA fine-tuning

[`LoRALinear`](../nn/layers/lora.md) wraps a linear layer and freezes its base weights automatically, so only the low-rank `A` and `B` matrices train:

```python
lora = nn.LoRALinear(base_linear, rank=8, key=key)

# Optimizer auto-partitions: only the LoRA matrices get optimizer state
optimizer = ion.Optimizer(optax.adam(3e-4), lora)
```

## Tree-level functions

The same operations exist as standalone functions that work on any pytree, not just modules. Use the predicates with `jax.tree` utilities to inspect trainability.

::: ion.freeze

::: ion.unfreeze

::: ion.is_param

::: ion.is_trainable_param
