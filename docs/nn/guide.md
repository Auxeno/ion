# Neural networks

For neural network layers and their APIs, see the [NN layer reference](layers/index.md).

Every Ion layer is a pytree of parameters. Stateless layers map arrays to
arrays, while stateful layers update declared `Buffer` fields in place. This
guide covers the two shape contracts those functions use, how layers compose
into a model, and where state and randomness enter.

- [Feature axes](#feature-axes)
- [Structured axes](#structured-axes)
- [Adding leading axes](#adding-leading-axes)
- [Inside a layer](#inside-a-layer)
- [Composing layers](#composing-layers)
- [Stateful layers](#stateful-layers)
- [Carrying state](#carrying-state)
- [Randomness](#randomness)
- [Training the model](#training-the-model)
- [Array conventions](#array-conventions)
- [Further examples](#further-examples)

## Feature axes

A `Linear` layer holds a weight matrix and transforms the last axis of its
input:

```python
import jax
import jax.numpy as jnp

from ion import nn

key = jax.random.key(0)
linear = nn.Linear(3, 16, key=key)
```

The layer never sees a batch size. It only requires that the final axis has
length 3, and it leaves every axis before that untouched:

```python
linear(jnp.ones((3,))).shape        # (16,)
linear(jnp.ones((32, 3))).shape     # (32, 16)
linear(jnp.ones((8, 32, 3))).shape  # (8, 32, 16)
```

One vector, a batch of vectors, and a batch of sequences of vectors all work,
with no reshaping and no `vmap`. The layer reference writes this as
`(*, 3) -> (*, 16)`, where `*` stands for any number of leading axes, including
none.

Layers that act on a feature axis and nothing else all share this contract:
`Linear`, `LayerNorm`, `RMSNorm`, `Embedding`, `Dropout`, `LoRALinear`,
`Identity`, and `MLP`. Whatever the leading axes mean is the caller's business.

## Structured axes

The remaining layers cannot be so relaxed. A convolution slides a kernel across
spatial axes, a recurrent layer scans along time, and attention compares
positions with each other. Each needs to know exactly which axis is which, so
it fixes the rank of its input:

```python
conv = nn.Conv(3, 16, kernel_shape=(3, 3), padding=1, key=key)
conv(jnp.ones((8, 28, 28, 3))).shape  # (8, 28, 28, 16)
```

`Conv` takes `(b, *spatial, c)`. Images are channels-last, and the number of
spatial axes follows `kernel_shape`, so the same layer covers 1D, 2D, and 3D
convolution. `MaxPool` and `AvgPool` use the same layout.

An input without a batch axis is an error rather than a single unbatched
example:

```python
conv(jnp.ones((28, 28, 3)))  # ValueError
```

Sequence layers fix rank the same way, with time as the second axis:

```python
lstm = nn.LSTM(3, 16, key=key)
outputs, (h, c) = lstm(jnp.ones((8, 20, 3)))

outputs.shape  # (8, 20, 16), one hidden state per timestep
h.shape        # (8, 16), the final hidden state
```

`RNN`, `GRU`, `LSTM`, `S4D`, `S5`, and `LRU` all take `(b, t, d)`.
`SelfAttention` and `CrossAttention` take `(b, s, d)`, where `s` is the
sequence length being attended over.

| Layer | Input | Fixed axes |
|---|---|---|
| `Linear`, `LayerNorm`, `RMSNorm`, `Embedding`, `Dropout` | `(*, d)` | Feature axis only |
| `BatchNorm` | `(b, ..., d)` | One or more leading reduction axes |
| `Conv`, `ConvTranspose`, `MaxPool`, `AvgPool` | `(b, *spatial, c)` | Batch, spatial, channels |
| `GroupNorm` | `(*, *spatial, c)` | Spatial axes per `num_spatial_dims` |
| `RNN`, `LSTM`, `GRU`, `S4D`, `S5`, `LRU` | `(b, t, d)` | Batch, time, features |
| `SelfAttention`, `CrossAttention` | `(b, s, d)` | Batch, sequence, features |

## Adding leading axes

Fixed rank is not a limit on what can be expressed, because `jax.vmap` maps any
layer over an extra axis. An attention layer rejects a fourth axis:

```python
attn = nn.SelfAttention(64, num_heads=8, key=key)
attn(jnp.ones((2, 4, 16, 64)))  # ValueError
```

Wrapping the call restores it:

```python
jax.vmap(attn)(jnp.ones((2, 4, 16, 64))).shape  # (2, 4, 16, 64)
```

Ion modules are pytrees, so `vmap` treats the layer as data and maps over the
input alone. Nothing about the layer needs to change. The same mapping handles a
single unbatched image, though adding the axis with `x[None]` is usually
simpler.

`vmap` is also how one model becomes many. Mapping construction over a batch of
keys produces an ensemble that runs in a single call:

```python
keys = jax.random.split(key, 8)
ensemble = jax.vmap(lambda key: nn.MLP([3, 64, 1], key=key))(keys)
preds = jax.vmap(lambda model: model(jnp.ones((32, 3))))(ensemble)

preds.shape  # (8, 32, 1)
```

These examples do not update buffers. Evaluation calls may read buffers under
`vmap`, but a training call cannot update one buffer concurrently from multiple
mapped lanes. `BatchNorm` already reduces over all leading axes, so pass the
whole batch directly instead. See
[Sharp edges](../sharp-edges.md#buffer-mutation-and-jax-transforms).

## Inside a layer

A layer is a `Module`, and trainable arrays in that pytree are `Param` objects.
Bare array fields remain ordinary model data, while buffers contribute no
pytree leaves. Constructing a layer creates its parameters, and printing it
shows the structure without printing any values:

```python
mlp = nn.MLP([3, 64, 64, 1], key=key)
print(mlp)
```

```text
MLP(
  layers=(
    Linear(
      w=Param(f32[3, 64], trainable=True),
      b=Param(f32[64], trainable=True),
    ),
    Linear(
      w=Param(f32[64, 64], trainable=True),
      b=Param(f32[64], trainable=True),
    ),
    Linear(
      w=Param(f32[64, 1], trainable=True),
      b=Param(f32[1], trainable=True),
    ),
  ),
  activation=relu,
  final_activation=None,
)
```

Every leaf of that tree is a `Param`, and everything else is structure that JAX
transformations pass through untouched.

The `activation` field holds `jax.nn.relu` itself. It is a field like any other,
but not a leaf, so it becomes part of the pytree structure rather than something
`jax.grad` differentiates. [Param](../core/param.md) and
[Module](../core/module.md) cover this split in full.

```python
mlp.num_params  # 4481
```

## Composing layers

`Sequential` chains callables. Layers, plain functions, and lambdas all qualify:

```python
keys = jax.random.split(key, 2)
model = nn.Sequential(
    nn.Linear(3, 64, key=keys[0]),
    jax.nn.relu,
    nn.Linear(64, 1, key=keys[1]),
)
```

`MLP` is the same idea specialized to alternating linear layers and one
activation, so `nn.MLP([3, 64, 1], key=key)` replaces the above.

Anything with branching, multiple inputs, or intermediate values worth naming is
written as a `Module` instead. Declare the submodules as class annotations,
assign them in `__init__`, and write the forward pass in `__call__`:

```python
class SequenceClassifier(nn.Module):
    embed: nn.Embedding
    lstm: nn.LSTM
    norm: nn.LayerNorm
    head: nn.Linear

    def __init__(self, vocab_size, dim, num_classes, *, key):
        keys = jax.random.split(key, 3)
        self.embed = nn.Embedding(vocab_size, dim, key=keys[0])
        self.lstm = nn.LSTM(dim, dim, key=keys[1])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes, key=keys[2])

    def __call__(self, ids):
        x = self.embed(ids)
        x, (h, c) = self.lstm(x)
        x = self.norm(h)
        return self.head(x)

model = SequenceClassifier(1000, 64, 4, key=key)
logits = model(jnp.zeros((8, 20), dtype=jnp.int32))

logits.shape  # (8, 4)
```

The constructor splits one key into as many as it has parameterized submodules.
`LayerNorm` needs no key because its parameters are fixed to ones and zeros.

Both contracts appear in that forward pass. `Embedding` accepts integer ids of
any shape and appends a feature axis, `LSTM` requires exactly `(b, t, d)`, and
the closing `LayerNorm` and `Linear` act on `(b, 64)` without caring that the
time axis is gone:

```python
ids              # (8, 20), integer token ids
self.embed(ids)  # (8, 20, 64)
self.lstm(x)     # (8, 20, 64), ((8, 64), (8, 64))
self.norm(h)     # (8, 64)
self.head(x)     # (8, 4)
```

## Stateful layers

Some layers update non-trainable values during the forward pass. `BatchNorm`
tracks a running mean and variance, while `SpectralNorm` tracks singular-vector
estimates. These values are [`Buffer`](../core/buffers.md) fields, updated in
place, so a stateful layer is called like any other:

```python
model = nn.Sequential(
    nn.Linear(3, 64, key=key),
    nn.BatchNorm(64),
    jax.nn.relu,
)

y = model(x, training=True)   # running statistics updated
y = model(x, training=False)  # running statistics used
```

Buffers contribute no pytree leaves, so `jax.grad`, the optimizer, and
`Module.astype` all leave them alone. They can be read and updated inside
`jax.jit` and `jax.lax.scan`. Do not put a buffer update inside `jax.checkpoint`
or map several writes to one buffer with `jax.vmap`. See
[Stateful training](../workflows.md#stateful-training) for a complete update
step and [Sharp edges](../sharp-edges.md#buffer-mutation-and-jax-transforms) for
the transform boundaries.

## Carrying state

`LSTM` returns its final state alongside the outputs. Passing that state back in
continues the sequence, which is how a long sequence is processed in chunks:

```python
outputs_1, state = lstm(chunk_1)
outputs_2, state = lstm(chunk_2, state)
```

This recurrent state belongs to one input sequence, and is passed explicitly.
Buffers instead belong to the model and usually persist across training batches.

Sequence layers run the scan internally. To control the recurrence directly, use
the cell that each one wraps. Cells take a single timestep and follow the
feature-axis contract, so they accept any leading axes:

```python
cell = nn.LSTMCell(3, 16, key=key)
h, c = cell(jnp.ones((8, 3)), (jnp.zeros((8, 16)), jnp.zeros((8, 16))))
```

`cell.initial_state` supplies zeros of the right shape for one example, which
broadcast to a batch. Driving the cell with `lax.scan` reproduces `LSTM`, with
the time axis moved to the front:

```python
from jax import lax

def step(state, x_t):
    state = cell(x_t, state)
    return state, state[0]

h0 = jax.tree.map(lambda s: jnp.broadcast_to(s, (8, 16)), cell.initial_state)
state, outputs = lax.scan(step, h0, jnp.moveaxis(x, 1, 0))
```

Writing the loop makes room for anything the packaged layer does not do, such as
resetting state at episode boundaries or feeding each output back in as the next
input. `RNNCell`, `GRUCell`, `S4DCell`, `S5Cell`, and `LRUCell` behave the same
way.

## Randomness

Parameter initialization takes a key at construction, and it is always
keyword-only:

```python
model = nn.MLP([3, 64, 1], key=key)
```

`Dropout` needs randomness on every call instead, so it takes a key there:

```python
dropout = nn.Dropout(0.1)
y = dropout(x, training=True, key=key)
y = dropout(x, training=False)
```

Ion holds no global RNG and no hidden counter. Any key reaching a layer was
split by the caller, which is what keeps a training step reproducible and safe
to `jit`. `Sequential` splits the key it receives and passes one to each layer
that accepts a `key` argument:

```python
model = nn.Sequential(nn.Linear(3, 64, key=keys[0]), nn.Dropout(0.1))
y = model(x, training=True, key=key)
```

There is no `model.train()` or `model.eval()`. Modules are immutable, so
training mode is passed explicitly to layers that behave differently during
training and evaluation:

```python
y = model(x, training=True, key=key)
y = model(x, training=False)
```

Evaluation makes `Dropout` the identity and makes stateful layers read buffers
without updating them. It needs no call-time random key unless another layer
uses one during evaluation.

## Training the model

Training uses native JAX transformations. The loss takes the model as its first
argument so `jax.grad` differentiates with respect to its parameters, and
[`ion.Optimizer`](../core/optimizer.md) wraps any optax transform:

```python
import optax

import ion

model = SequenceClassifier(1000, 64, 4, key=key)
optimizer = ion.Optimizer(optax.adam(1e-3), model)

def loss_fn(model, ids, labels):
    logits = model(ids)
    return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

@jax.jit
def train_step(model, optimizer, ids, labels):
    loss, grads = jax.value_and_grad(loss_fn)(model, ids, labels)
    model, optimizer = optimizer.update(model, grads)
    return model, optimizer, loss
```

The whole step compiles because the model is a pytree: JAX flattens it into
arrays on the way in and rebuilds it on the way out. Ion adds no training loop,
data loader, or callbacks, so iterating over batches is ordinary Python.
Models with stateful layers also carry their buffers through the loss; see
[Stateful training](../workflows.md#stateful-training).

## Array conventions

| Convention | Rule |
|---|---|
| Feature axis | Last, always |
| Leading axes | Free for feature-axis layers, fixed for structured ones |
| Images | Channels-last, `(b, h, w, c)` |
| Sequences | Time or sequence second, `(b, t, d)` |
| Batching | Explicit in the array, or added with `jax.vmap` |
| Keys | Keyword-only, used for parameter and buffer initialization and stochastic calls |
| Training mode | Explicit on layers that differ between training and evaluation |
| Dtypes | No `dtype` argument on layers, cast with `model.astype(...)` |

Casting a whole model is covered in
[Mixed precision](../workflows.md#mixed-precision).

### Shape labels

| Label | Meaning |
|---|---|
| `b` | Batch size |
| `t`, `s` | Time or sequence length |
| `d` | General feature dimension |
| `i`, `o` | Input and output feature dimensions |
| `c` | Channels |
| `h`, `k` | Attention heads and per-head dimension |
| `v` | Vocabulary size |
| `*` | Any number of leading axes |

Each layer page defines how these labels apply to its inputs, parameters, and
outputs.

## Further examples

- [CNN on MNIST](../examples/cnn-mnist.md) trains an image classifier on
  channels-last batches.
- [RNN on MNIST](../examples/rnn-mnist.md) reads the same images as sequences of
  rows.
- [GPT on TinyStories](../examples/gpt-tinystories.ipynb) stacks attention and
  normalization into a transformer.
- [SSM on Pathfinder](../examples/ssm-pathfinder.ipynb) runs state space layers
  over long sequences.
