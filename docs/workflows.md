# Workflows

Common workflows that apply across models and layer families:

- [Checkpointing](#checkpointing)
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

A stateful model needs nothing extra. Buffers live in the model, so running statistics are written and restored along with the parameters.

`load` takes a reference pytree that supplies the structure and non-array fields. Array values and `Param.trainable` flags come from the checkpoint:

```python
reference = MyModel(key=jax.random.key(0))
model = ion.load("model.ion", reference)
```

Checkpoint tensor names use the same tree paths as `Module.at`, such as `blocks[2].attn.w_q`. Shape or structure mismatches raise `ValueError`; dtype mismatches warn and preserve the saved dtype. See [Sharp edges](sharp-edges.md) for how static configuration is handled.

::: ion.save
    options:
      heading_level: 3

::: ion.load
    options:
      heading_level: 3

## Mixed precision

Layer constructors use JAX's default floating dtype and do not take a `dtype` argument. For mixed-precision training, keep float32 master parameters and cast the model inside the loss:

```python
def loss_fn(model, x, y):
    model = model.astype(jnp.bfloat16)
    x = x.astype(jnp.bfloat16)
    logits = model(x)
    return optax.softmax_cross_entropy_with_integer_labels(
        logits.astype(jnp.float32), y
    ).mean()
```

The casts are differentiable, so gradients return in float32 to match the master parameters and optimizer state. The model's forward and backward computation uses bfloat16, while cross-entropy stays in float32 for numerical stability.

Buffers are not cast, and the cast model shares them with the master rather than copying them, so a stateful layer keeps float32 running statistics and keeps updating the master's copy of them.

For low-precision inference, cast the model once:

```python
model = model.astype(jnp.bfloat16)
predictions = model(x.astype(jnp.bfloat16))
```

Cast the inputs as well as the model. Float32 inputs otherwise promote the computation back to float32. See [Sharp edges](sharp-edges.md) for details.

`Module.astype` casts a model; `ion.astype` casts any pytree, so it also reaches optimizer state and plain arrays:

::: ion.astype
    options:
      heading_level: 3

## Freezing

A frozen parameter still participates in the forward pass, but receives a zero gradient and is skipped by the optimizer. Freeze or unfreeze a complete model:

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

Change trainability before constructing the optimizer. Its state is created only for the parameters that are trainable at construction:

```python
model = model.freeze()
model = model.at.classifier.set(model.classifier.unfreeze())

# Only the classifier receives optimizer state
optimizer = ion.Optimizer(optax.adam(3e-4), model)
```

Changing trainability later changes the model's [pytree](https://docs.jax.dev/en/latest/pytrees.html) structure, so construct a new optimizer from the updated model:

```python
model = model.unfreeze()
optimizer = ion.Optimizer(optax.adam(3e-4), model)
```

`Module.freeze` and `Module.unfreeze` act on a model; `ion.freeze` and `ion.unfreeze` do the same for any pytree. The `is_param` and `is_trainable_param` predicates filter leaves when building custom masks:

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

A model prints as a tree, grouped into config, parameters, buffers and child modules, with parameter counts and sizes on each line. Frozen params are marked `frozen`. In a color terminal each layer is highlighted with the same color Treescope gives it, carried on both of its brackets. Layers sharing a mechanism share a hue, so every convolution is one color and every normalization another, and numbers, strings, constants and dtypes take the colors the documentation gives them in code blocks. Output captured to a pipe or file drops the color, leaving plain text:

```text
MLP(  # 131 params, 524 B, 80 frozen
  activation=relu, final_activation=None,
  # Modules:
  (0): Linear(  # 80 params, 320 B, 80 frozen
    # Parameters:
    w=Param(float32(4, 16), frozen),
    b=Param(float32(16,), frozen),
  ),
  (1): Linear(  # 51 params, 204 B
    # Parameters:
    w=Param(float32(16, 3)),
    b=Param(float32(3,)),
  ),
)
```

Printing or evaluating a model adds a distribution histogram and moments to each parameter, aligned in a column so layers can be compared by eye:

```text
Linear(  # 4,608 params, 18 KB
  # Parameters:
  w=Param(float32(8, 512)),  █▇▇▇▇▇▇▇▇▇█  μ=-0.00036 σ=0.062
  b=Param(float32(512,)),    ▁▁▁▁▁█▁▁▁▁▁  μ=0 σ=0
)
```

Every summary comes from at most 16,384 evenly spaced values. A parameter no larger than that sample is described exactly; a larger one is described from the sample and marks both moments `≈` so no approximation reads as measured. The histogram bins between the 1st and 99th percentile of the sample, so a few outliers cannot flatten the bars. A constant parameter has no width, so its mass sits in the middle bucket with a zero deviation.

The whole tree costs one device synchronization, and a large parameter costs no more than a small one. Where a model is printed often enough for that to matter, in logs or an exception handler, turn the descriptions off and `repr` renders structure alone:

```python
ion.statistics = False
```

Inside `jax.jit` and friends parameters are tracers rather than concrete arrays, so a model printed there renders its structure with no descriptions either way.

In IPython and Jupyter, [Treescope](https://github.com/google-deepmind/treescope) renders the same tree interactively, with collapsible nodes and array visualizations. Ion ships the rendering hooks but does not install or activate it, so pull it in and turn it on the usual way:

```bash
pip install treescope
```

```python
import treescope

treescope.basic_interactive_setup()
```

Modules, params, buffers and optimizers then render as folding trees, collapsed down to shapes and expanding to array visualizations on click.

## Measuring cost

`Module.cost` traces and compiles a call without executing it, then explains its static work and memory layer by layer:

```python
report = model.cost(jnp.ones((8, 128), jnp.int32))
print(report)
```

```text
TransformerBlock · input float32(8, 128, 384) · GPU/XLA

3.86GFLOP · 1,771,008 params (6.76 MB) · 85 ops → 34 fused
21.8 MB total memory = (1.5 MB + 6.76 MB) input + 12 MB intermediate + 1.5 MB output

layer                             FLOPs              share  ops  output
TransformerBlock                  3.86G  ██████████ 100.0%   85  float32(8, 128, 384)
  attention MultiHeadAttention    1.42G  ███▊        36.8%   30  float32(8, 128, 384)
  attention_norm LayerNorm        2.37M               0.1%   16  float32(8, 128, 384)
  mlp_norm LayerNorm              2.37M               0.1%   16  float32(8, 128, 384)
  mlp_in Linear                   1.21G      ██▉     31.3%    1  float32(8, 128, 1536)
  mlp_out Linear                  1.21G         ███  31.3%    1  float32(8, 128, 384)
```

The tree and colours match the model repr. FLOPs are inclusive: a parent's value contains its descendants. `share` is simply that value divided by the whole call's FLOPs. Sibling bars tile their parent, leaving any work done directly by the parent as an unfilled segment.

`ops` counts traced JAX operations once in the static graph. A scan body therefore keeps the same op count at every sequence length and carries a `loop xT` suffix, while its FLOPs scale by `T`. `fused` counts executable operations left in the optimized top-level graph after fusion and other compiler simplification. It is not a count of device kernel launches.

The output column records the logical shape and dtype produced by each module during the trace. Fusion may prevent that value from becoming a physical device buffer.

`Module.cost` analyses a model's own call; `ion.cost` does the same for any callable taking a model, so a loss, gradient evaluation or whole step is analysed the same way:

```python
ion.cost(loss, model, x, y)
ion.cost(jax.grad(loss), model, x, y)
model.cost(jax.ShapeDtypeStruct((8192, 256), jnp.float32))
```

Concrete array pytrees are abstractified automatically, while a `jax.ShapeDtypeStruct` avoids allocating the input in the first place. Results are accessible directly by tree path:

```python
report.layers["layers[3]"].flops
report.layers["layers[3]"].output
report.total_memory
```

The memory line is the compiler's buffer plan. Input is all array arguments, including model parameters; the parentheses separate non-parameter inputs from parameter storage. Intermediate is temporary storage allocated for the call, and output is the returned buffers. Reused or donated storage is subtracted when the compiler reports aliases. This is not observed process or allocator memory.

FLOPs use the conventional two operations for a multiply-add. Matmul and convolution counts are precise under that convention; elementwise counts are indicative. Dynamic `cond` and unknown-trip `while` control flow are rejected because they have no single factual static cost.

::: ion.cost
    options:
      heading_level: 3
