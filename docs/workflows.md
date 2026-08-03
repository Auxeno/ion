# Workflows

Common workflows that apply across models and layer families:

- [Checkpointing](#checkpointing)
- [Stateful training](#stateful-training)
- [Mixed precision](#mixed-precision)
- [Freezing](#freezing)
- [Inspecting models](#inspecting-models)

## Checkpointing

`save` and `load` persist any pytree to an `.ion` file:

```python
ion.save("model.ion", model)
model = ion.load("model.ion", model)
```

Save the model and optimizer together to resume training:

```python
ion.save("checkpoint.ion", (model, optimizer))
model, optimizer = ion.load("checkpoint.ion", (model, optimizer))
```

Include buffers when checkpointing a stateful model:

```python
ion.save("checkpoint.ion", (model, buffers, optimizer))

reference = MyModel(key=jax.random.key(0))
reference_buffers = reference.init_buffers(key=jax.random.key(1))
reference_optimizer = ion.Optimizer(optax.adam(1e-3), reference)
model, buffers, optimizer = ion.load(
    "checkpoint.ion", (reference, reference_buffers, reference_optimizer)
)
```

`load` takes a reference pytree that supplies the structure and non-array
fields. Array values and `Param.trainable` flags come from the checkpoint:

```python
reference = MyModel(key=jax.random.key(0))
model = ion.load("model.ion", reference)
```

Checkpoint tensor names use the same tree paths as `Module.at`, such as
`blocks[2].attn.w_q`. Shape or structure mismatches raise `ValueError`; dtype
mismatches warn and preserve the saved dtype. See [Sharp
edges](sharp-edges.md) for how static configuration is handled.

::: ion.save
    options:
      heading_level: 3

::: ion.load
    options:
      heading_level: 3

## Stateful training

Stateful layers return updated [`Buffers`](core/buffers.md) alongside their
output. Carry those buffers through the loss as auxiliary data so gradients are
still taken only with respect to the model:

```python
key_linear, key_output = jax.random.split(key)
model = nn.Sequential(
    nn.Linear(4, 16, key=key_linear),
    nn.BatchNorm(16),
    jax.nn.relu,
    nn.Dropout(0.1),
    nn.Linear(16, 3, key=key_output),
)
buffers = model.init_buffers()
optimizer = ion.Optimizer(optax.adam(1e-3), model)

def loss_fn(model, buffers, x, labels, key):
    logits, buffers = model(x, buffers, training=True, key=key)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    return loss, buffers

@jax.jit
def train_step(model, buffers, optimizer, x, labels, key):
    (loss, buffers), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        model, buffers, x, labels, key
    )
    model, optimizer = optimizer.update(model, grads)
    return model, buffers, optimizer, loss
```

Evaluation uses the latest buffers without replacing them:

```python
logits, _ = model(x, buffers, training=False)
```

Buffers are pytrees with a fixed structure, so passing updated values through a
compiled step does not itself cause recompilation.

## Mixed precision

Layer constructors use JAX's default floating dtype and do not take a `dtype`
argument. For mixed-precision training, keep float32 master parameters and cast
the model inside the loss:

```python
def loss_fn(model, x, y):
    model = model.astype(jnp.bfloat16)
    x = x.astype(jnp.bfloat16)
    logits = model(x)
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
```

The cast is differentiable, so gradients return in float32 to match the master
parameters and optimizer state. Only the forward and backward computation uses
bfloat16.

Stateful layers choose their buffer precision independently. Numerically
sensitive values can remain in float32 without promoting the layer output, so
buffers do not need to be cast alongside the model.

For low-precision inference, cast the model once:

```python
model = model.astype(jnp.bfloat16)
predictions = model(x.astype(jnp.bfloat16))
```

Cast the inputs as well as the model. Float32 inputs otherwise promote the
computation back to float32. See [Sharp edges](sharp-edges.md) for details.

`Module.astype` casts a model; `ion.astype` casts any pytree, so it also reaches
optimizer state and plain arrays:

::: ion.astype
    options:
      heading_level: 3

## Freezing

A frozen parameter still participates in the forward pass, but receives a zero
gradient and is skipped by the optimizer. Freeze or unfreeze a complete model:

```python
frozen_model = model.freeze()
trainable_model = frozen_model.unfreeze()
```

Use `at` to change one part of a model:

```python
# Freeze just the encoder
model = model.at.encoder.set(model.encoder.freeze())

# Freeze everything except the classifier
model = model.freeze()
model = model.at.classifier.set(model.classifier.unfreeze())
```

Change trainability before constructing the optimizer. Its state is created
only for the parameters that are trainable at construction:

```python
model = model.freeze()
model = model.at.classifier.set(model.classifier.unfreeze())

# Only the classifier receives optimizer state
optimizer = ion.Optimizer(optax.adam(3e-4), model)
```

Changing trainability later changes the model's
[pytree](https://docs.jax.dev/en/latest/pytrees.html) structure, so construct
a new optimizer from the updated model:

```python
model = model.unfreeze()
optimizer = ion.Optimizer(optax.adam(3e-4), model)
```

[`LoRALinear`](nn/layers/lora.md) freezes its base weights automatically:

```python
lora = nn.LoRALinear(base_linear, rank=8, key=key)
optimizer = ion.Optimizer(optax.adam(3e-4), lora)
```

Only the low-rank `A` and `B` parameters receive optimizer state.

`Module.freeze` and `Module.unfreeze` act on a model; `ion.freeze` and
`ion.unfreeze` do the same for any pytree. The `is_param` and
`is_trainable_param` predicates filter leaves when building custom masks:

::: ion.freeze
    options:
      heading_level: 3

::: ion.unfreeze
    options:
      heading_level: 3

::: ion.is_param
    options:
      heading_level: 3

::: ion.is_trainable_param
    options:
      heading_level: 3

## Inspecting models

A model prints as a tree, with each `Param` showing its dtype, shape, and
trainability:

```text
MLP(
  layer_1=Linear(
    w=Param(f32[4, 16], trainable=True),
    b=Param(f32[16], trainable=True),
  ),
  layer_2=Linear(
    w=Param(f32[16, 3], trainable=True),
    b=Param(f32[3], trainable=True),
  ),
  activation=relu,
)
```

In IPython and Jupyter, [Treescope](https://github.com/google-deepmind/treescope)
renders the same tree interactively, with collapsible nodes and array
visualizations. It is enabled on import and covers Ion modules, params, buffers,
and JAX arrays:

```python
ion.enable_treescope()                 # Ion types and arrays (default)
ion.enable_treescope(everything=True)  # every type treescope supports
ion.disable_treescope()                # fall back to plain text
```

::: ion.enable_treescope
    options:
      heading_level: 3

::: ion.disable_treescope
    options:
      heading_level: 3
