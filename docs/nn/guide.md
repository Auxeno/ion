# Neural networks

This guide builds and trains a neural network with JAX and Ion. For individual
layers and their APIs, see the [NN layer reference](layers/index.md).

Follow the guide from the beginning for a walkthrough, or jump directly to a
topic:

- [Feature arrays](#feature-arrays)
- [Layers and nonlinearities](#layers-and-nonlinearities)
- [Building a model](#building-a-model)
- [Losses and predictions](#losses-and-predictions)
- [Gradients and updates](#gradients-and-updates)
- [Compiling the training step](#compiling-the-training-step)
- [Array conventions](#array-conventions)
- [Further examples](#further-examples)

## Feature arrays

The final axis of an input array stores features. Here there are four items and
three features per item:

```python
import jax
import jax.numpy as jnp
import optax

import ion
from ion import nn

x = jnp.array([
    [1.0, 0.2, -0.4],
    [0.3, 0.8, 0.1],
    [-0.2, 0.5, 0.9],
    [0.7, -0.1, 0.4],
])

x.shape  # (4 items, 3 features)
```

The leading axis can represent a batch, a sequence, or another collection of
items. The meaning depends on the layer and task. The final axis is the feature
dimension transformed by a fully connected layer.

## Layers and nonlinearities

A `Linear` layer applies the same learned transformation to every row:

```python
linear = nn.Linear(
    in_dim=3,
    out_dim=8,
    key=jax.random.key(0),
)

h = linear(x)
h.shape  # (4, 8)
```

The layer computes `x @ w + b`. Its weight has shape `(3, 8)`, so the final
dimension changes from three input features to eight output features. The
leading dimension is preserved.

An activation function is applied between learned transformations:

```python
h = jax.nn.relu(linear(x))
```

Without the activation, several linear layers in sequence still reduce to one
linear transformation. The activation lets successive layers represent
nonlinear functions.

## Building a model

`MLP` constructs a sequence of linear layers with an activation between them.
The final layer has no activation by default, so its outputs can be used as
class logits:

```python
model = nn.MLP(
    dims=[3, 16, 16, 2],
    key=jax.random.key(1),
)

logits = model(x)
logits.shape  # (4, 2)
```

The dimensions describe the complete path through the network:

```text
3 input features -> 16 hidden features -> 16 hidden features -> 2 logits
```

Composite models are constructed and called like individual layers. A custom
model declares its submodules as fields:

```python
class Classifier(nn.Module):
    hidden: nn.Linear
    output: nn.Linear

    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, *, key):
        key_hidden, key_output = jax.random.split(key)
        self.hidden = nn.Linear(in_dim, hidden_dim, key=key_hidden)
        self.output = nn.Linear(hidden_dim, num_classes, key=key_output)

    def __call__(self, x):
        x = jax.nn.relu(self.hidden(x))
        return self.output(x)


model = Classifier(3, 16, 2, key=jax.random.key(2))
logits = model(x)
```

The fields form a pytree containing the model's parameters and static
configuration. There is no separate parameter dictionary.

## Losses and predictions

For a classification task, each output row contains one logit per class:

```python
targets = jnp.array([0, 1, 1, 0])
predictions = jnp.argmax(logits, axis=-1)

predictions.shape  # (4,)
```

Optax provides the loss function. The loss takes the model as an explicit
argument so JAX can differentiate with respect to it:

```python
def loss_fn(model, x, targets):
    logits = model(x)
    losses = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    return losses.mean()


loss = loss_fn(model, x, targets)
```

Keeping the computation in a function also makes the data flow visible: the
model and batch go in, and a scalar loss comes out.

## Gradients and updates

Because the model is a JAX pytree, `jax.grad` returns a matching pytree of
gradients:

```python
grads = jax.grad(loss_fn)(model, x, targets)
```

`jax.value_and_grad` computes the loss and gradients together, which is the
usual form inside a training step:

```python
loss, grads = jax.value_and_grad(loss_fn)(model, x, targets)
```

There is no `ion.grad`. Native JAX transformations operate directly on the
model.

`ion.Optimizer` wraps an Optax transformation and initializes its state from the
model:

```python
optimizer = ion.Optimizer(optax.adam(3e-4), model)
model, optimizer = optimizer.update(model, grads)
```

Models and optimizers are immutable values. `update` returns the next model and
optimizer rather than modifying either input.

## Compiling the training step

The complete training step can be compiled with `jax.jit`:

```python
@jax.jit
def train_step(model, optimizer, x, targets):
    loss, grads = jax.value_and_grad(loss_fn)(model, x, targets)
    model, optimizer = optimizer.update(model, grads)
    return model, optimizer, loss


model, optimizer, loss = train_step(model, optimizer, x, targets)
```

The same function is called for each batch:

```python
for x, targets in load_data():
    model, optimizer, loss = train_step(model, optimizer, x, targets)
```

The loader and loss depend on the task. The model, gradient, update, and
compilation pattern remains ordinary JAX.

## Array conventions

Pointwise layers such as `Linear`, `LayerNorm`, and `MLP` transform the final
feature axis and preserve arbitrary leading dimensions:

```python
linear = nn.Linear(64, 128, key=jax.random.key(3))

linear(jnp.ones((64,))).shape         # (128,)
linear(jnp.ones((32, 64))).shape      # (32, 128)
linear(jnp.ones((32, 16, 64))).shape  # (32, 16, 128)
```

### Input layouts

Structural layers assign specific meanings to their leading axes. Ion uses
channels-last ordering:

| Domain | Format | Example |
|---|---|---|
| Vector data | `(batch, features)` | `(32, 256)` |
| Sequences | `(batch, length, channels)` | `(32, 128, 64)` |
| Images | `(batch, height, width, channels)` | `(32, 32, 32, 3)` |
| Attention | `(batch, sequence, dimension)` | `(32, 128, 512)` |
| Recurrent | `(batch, time, features)` | `(32, 50, 64)` |

Convolution and pooling interpret spatial axes, while attention, recurrent, and
state space layers interpret sequence axes. Each [layer reference](layers/index.md)
documents any stricter rank or batching requirements.

### Adding batch dimensions

Some structural layers accept exactly one batch dimension. `jax.vmap` can add
an additional leading batch dimension without changing the layer:

```python
# x.shape == (5, batch, time, features)
y = jax.vmap(rnn)(x)
```

Pointwise layers do not normally need this because they already preserve
arbitrary leading dimensions.

### Shape labels

Single-letter dimension labels appear in type annotations and einsum strings.
Their meaning is local to each layer:

| Label | Common meaning |
|---|---|
| `...` | Arbitrary leading dimensions |
| `b` | Batch |
| `d` | Model or feature dimension |
| `i`, `o` | Input and output features |
| `r` | LoRA rank |
| `v` | Vocabulary size |

The signature and example on each layer page define its local shapes.

### Dtypes

Layer constructors do not take a `dtype` argument. Parameters use JAX's default
floating dtype when a model is constructed:

```python
model = nn.MLP([64, 128, 10], key=jax.random.key(4))
model = model.astype(jnp.bfloat16)
```

Use `model.astype(dtype)` to change parameter precision after construction. See
[Mixed precision](../workflows.md#mixed-precision) for activation casting and
optimizer considerations.

## Further examples

- [CNN on MNIST](../examples/cnn-mnist.md) trains a convolutional image
  classifier.
- [RNN on MNIST](../examples/rnn-mnist.md) treats each image as a pixel
  sequence.
- [Ion Tour](../examples/ion-tour.ipynb) covers modules, parameters,
  optimization, and checkpointing in a notebook.
