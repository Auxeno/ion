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

Echoing a model at an interactive prompt adds a distribution histogram and moments to each
parameter, aligned in a column so layers can be compared by eye:

```text
Linear(  # 4,608 params, 18 KB
  # Parameters:
  w=Param(float32(8, 512)),  █▇▇▇▇▇▇▇▇▆▇▇█  μ=-0.00036 σ=0.062
  b=Param(float32(512,)),    ▁▁▁▁▁▁█▁▁▁▁▁▁  μ=0 σ=0
)
```

Only the echo path pays for this. `repr` does no reductions, so logging, debuggers and
exception messages stay cheap on models of any size. The histogram bins between the 1st and
99th percentile of a subsample, while the moments are exact over the whole array. A constant
parameter has no width, so its mass sits in the middle bucket with a zero deviation.

In IPython and Jupyter, [Treescope](https://github.com/google-deepmind/treescope) renders the same tree interactively, with collapsible nodes and array visualizations. It is enabled on import and covers Ion modules, params, buffers, and JAX arrays:

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

## Measuring cost

`ion.cost` traces and compiles one call, then reports what it costs a layer at a time:

```python
report = ion.cost(model, jnp.ones((8, 128), jnp.int32))
print(report)
```

```text
20.6GFLOP · 486 MB transferred · 65.5 MB peak memory
16,145,152 params · 146 ops → 71 kernels · balance 250 FLOP/byte

layer                             FLOPs              share  ceiling    memory   transfer  dtype
Sequential                        20.6G  ██████████ 100.0%      16%    128 MB     486 MB  float32
  (0) Embedding                    397K  ▋            5.2%       0%    1.5 MB    25.5 MB  float32
  (1) Sequential                  3.84G   █▉         22.5%      13%   13.5 MB     109 MB  float32
  (2) Sequential                  3.84G     ██▏      22.5%      13%   13.5 MB     109 MB  float32
  (3) Linear                      12.9G       ████▌  44.7%      23%    128 MB     218 MB  float32
```

The tree is the one the model repr prints, so the two line up row for row, nested rows and
all; only the top level is shown above. Everything the repr already shows is left out:
parameter counts, shapes, stored dtypes and distributions stay there, and the table carries
only what a call reveals.

**share** estimates the fraction of the whole call a layer accounts for, from `max(flops /
peak, bytes / bandwidth)` normalised across the model. Only the ratio of those two device
figures survives the normalisation, which is the `balance` argument, so the ranking holds on
any accelerator with a similar ratio. Below the ridge it reduces to a share of memory
traffic, and above it to a share of arithmetic.

The bars read differently from the numbers, deliberately. Each set of siblings tiles its
parent's width, so a bar starts where the previous one ended and its length is what that
layer took of its parent, not of the model. A level that leaves the bar short of full has
spent the remainder on work the parent does itself, such as the activations between an MLP's
layers. Bars also fade as they nest, so a level never reads as louder than the one above it.
Read down the numbers for magnitude, across the bars for structure.

**ceiling** is the most of the device's compute a layer can reach before bandwidth limits it.
It is a ceiling, not a measurement: a layer at 5% cannot exceed 5% however well its kernels
are written, because the arithmetic units are idle while its bytes move. Reaching 100% means
the memory system keeps up and only arithmetic is left to limit it.

Reading those two together is the point. The embedding above does 0.0% of the arithmetic and
still takes 7.0% of the call, at a 0% ceiling, which no FLOP count would show. The head does
most of the arithmetic in the model and is the only part running anywhere near efficiently.

Any callable taking a model works, so a backward pass or a whole step is measured the same way:

```python
ion.cost(loss, model, x, y)
ion.cost(jax.grad(loss), model, x, y)
ion.cost(model, jax.ShapeDtypeStruct((8192, 256), jnp.float32))
```

The numbers are also reachable directly, keyed by tree path:

```python
report.layers["layers[3]"].transferred  # 218 MB, in bytes
report.layers["layers[3]"].intensity    # arithmetic per byte moved
report.peak_memory                      # what the compiler reserves for intermediates
```

**memory** is a high-water mark, not a total: values die as soon as their last reader has
run, and their buffers are counted as free from that point. A sixteen-layer MLP therefore
holds no more at once than a four-layer one, and a scan reuses its carry so only its stacked
output grows with the sequence.

### What is measured and what is modelled

`params`, `ops`, `kernels` and `peak_memory` are counted, the last straight from the
compiler. `transferred` is read off the compiled HLO and agrees with XLA's own accounting to
within a thousandth across matmul, convolution, attention, normalisation and embedding
models.

`flops` is exact for matmuls and convolutions, where an operation count is unambiguous, and
conventional for elementwise chains, where it is not. Counting one operation per output
element puts a lone `LayerNorm` about a quarter under what XLA charges. Since real models
concentrate their arithmetic in matmuls, the total is close and the elementwise rows are
indicative.

`memory` is a liveness bound taken before fusion. Where the compiler has to materialise
buffers it lands within a couple of percent of `peak_memory`; where it fuses a chain into
registers it overstates, and a lone `LayerNorm` reads megabytes against a compiler peak of
tens of kilobytes. The header figure is the compiler's own and is the one that decides
whether a batch size fits.

`share` and `ceiling` are the modelled figures, and neither is a measurement. Both come from
the roofline: each kernel is weighed as `max(flops / peak, bytes / bandwidth)`, and a layer
sums the kernels inside it, since kernels run one after another. That keeps shares additive,
so children can never outweigh a parent even when some are compute limited and others
bandwidth limited.

Per-layer attribution is exact wherever XLA keeps a layer's work in its own kernels. Where a
fusion spans a boundary it carries one scope, so the work lands on whichever layer the
compiler named it after: a `Linear`'s bias add fused into the following activation is charged
to the parent that applies the activation.

::: ion.cost
    options:
      heading_level: 3
