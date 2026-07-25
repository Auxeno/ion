# Workflows

Common workflows that apply across models and layer families:

- [Inspecting models](#inspecting-models)
- [Freezing](#freezing)
- [Mixed precision](#mixed-precision)
- [Checkpointing](#checkpointing)

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
visualizations. It is enabled on import and covers Ion modules, params, and JAX
arrays:

```python
ion.enable_treescope()                 # Ion modules, params, and arrays (default)
ion.enable_treescope(everything=True)  # every type treescope supports
ion.disable_treescope()                # fall back to plain text
```

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

For low-precision inference, cast the model once:

```python
model = model.astype(jnp.bfloat16)
predictions = model(x.astype(jnp.bfloat16))
```

Cast the inputs as well as the model. Float32 inputs otherwise promote the
computation back to float32. See [Sharp edges](sharp-edges.md) for details.

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
