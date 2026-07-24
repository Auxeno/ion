# Neural networks

This guide follows a batch of MNIST images through an Ion model, from arrays to trained predictions. For individual
layers and their APIs, see the [NN layer reference](layers/index.md).

Follow the guide from the beginning for a complete training walkthrough, or jump directly to a topic:

- [Image batches](#image-batches)
- [Building the model](#building-the-model)
- [The model pytree](#the-model-pytree)
- [Forward pass](#forward-pass)
- [Loss and gradients](#loss-and-gradients)
- [Optimizer updates](#optimizer-updates)
- [Compiling the training step](#compiling-the-training-step)
- [Training on MNIST](#training-on-mnist)
- [Evaluation and randomness](#evaluation-and-randomness)
- [Array conventions](#array-conventions)
- [Further examples](#further-examples)

## Image batches

MNIST contains 28 by 28 grayscale images and an integer class label for each image. The loader in the complete
[CNN on MNIST example](../examples/cnn-mnist.md) returns NumPy arrays:

```python
train_images, train_labels, test_images, test_labels = load_mnist()

train_images.shape  # (60000, 28, 28, 1)
train_labels.shape  # (60000,)
train_images.dtype  # uint8
```

Ion uses channels-last image arrays. The axes here mean:

```text
(batch, height, width, channels)
```

Before a batch enters the model, move it into JAX and scale the pixel values from integers in `[0, 255]` to
floating-point values in `[0, 1]`:

```python
import jax
import jax.numpy as jnp
import optax

import ion
from ion import nn

batch_indices = jnp.arange(128)
images = jnp.asarray(train_images[batch_indices], dtype=jnp.float32) / 255.0
labels = jnp.asarray(train_labels[batch_indices])

images.shape  # (128, 28, 28, 1)
labels.shape  # (128,)
```

Every operation below receives arrays explicitly. Ion does not provide a data loader or training-loop abstraction.

## Building the model

The classifier uses two convolution and pooling stages, followed by two fully connected layers:

```python
class CNN(nn.Module):
    conv_1: nn.Conv
    conv_2: nn.Conv
    pool: nn.MaxPool
    fc_1: nn.Linear
    fc_2: nn.Linear

    def __init__(self, *, key):
        keys = jax.random.split(key, 4)
        self.conv_1 = nn.Conv(1, 16, kernel_shape=(3, 3), padding=1, key=keys[0])
        self.conv_2 = nn.Conv(16, 32, kernel_shape=(3, 3), padding=1, key=keys[1])
        self.pool = nn.MaxPool(kernel_shape=(2, 2))
        self.fc_1 = nn.Linear(32 * 7 * 7, 128, key=keys[2])
        self.fc_2 = nn.Linear(128, 10, key=keys[3])

    def __call__(self, x):
        x = jax.nn.relu(self.conv_1(x))
        x = self.pool(x)

        x = jax.nn.relu(self.conv_2(x))
        x = self.pool(x)

        x = x.reshape(x.shape[0], -1)
        x = jax.nn.relu(self.fc_1(x))
        return self.fc_2(x)
```

Submodules are ordinary fields. Parameterized layers receive an initialization key; `MaxPool` has no parameters and
needs no key. The same pool instance can be called twice because modules are immutable.

The feature shapes follow the operations in `__call__`:

| Operation | Output shape |
|---|---|
| Input batch | `(batch, 28, 28, 1)` |
| `conv_1`, ReLU, pool | `(batch, 14, 14, 16)` |
| `conv_2`, ReLU, pool | `(batch, 7, 7, 32)` |
| Flatten | `(batch, 1568)` |
| `fc_1`, ReLU | `(batch, 128)` |
| `fc_2` | `(batch, 10)` |

The final layer has no activation. Its ten outputs are logits, one for each digit class.

## The model pytree

Constructing the model creates its parameters:

```python
model = CNN(key=jax.random.key(0))
print(model)
```

Ion's terminal pretty printer displays the complete tree without printing every array value:

```text
CNN(
  conv_1=Conv(
    w=Param(f32[3, 3, 1, 16], trainable=True),
    b=Param(f32[16], trainable=True),
    kernel_shape=(3, 3),
    stride=(1, 1),
    padding=((1, 1), (1, 1)),
    dilation=(1, 1),
    groups=1,
  ),
  conv_2=Conv(
    w=Param(f32[3, 3, 16, 32], trainable=True),
    b=Param(f32[32], trainable=True),
    kernel_shape=(3, 3),
    stride=(1, 1),
    padding=((1, 1), (1, 1)),
    dilation=(1, 1),
    groups=1,
  ),
  pool=MaxPool(
    kernel_shape=(2, 2),
    stride=(2, 2),
    padding=((0, 0), (0, 0)),
  ),
  fc_1=Linear(
    w=Param(f32[1568, 128], trainable=True),
    b=Param(f32[128], trainable=True),
  ),
  fc_2=Linear(
    w=Param(f32[128, 10], trainable=True),
    b=Param(f32[10], trainable=True),
  ),
)
```

The fields form one JAX pytree. There is no separate parameter dictionary: `model` is both the callable network and the
value differentiated and updated during training. In IPython and Jupyter, the same model renders as an interactive
[Treescope](https://github.com/google-deepmind/treescope) tree.

The main training operations correspond as follows:

| Stateful training pattern | Ion and JAX |
|---|---|
| Parameters are collected from a model | Parameters are leaves of the model pytree |
| Gradients accumulate on parameters | A gradient pytree is returned |
| An optimizer mutates parameters | `update` returns a new model and optimizer |
| Random state is implicit | RNG keys are passed explicitly |
| A framework transform compiles the model | `jax.jit` compiles an ordinary function |

## Forward pass

Calling the model produces one row of logits per image:

```python
logits = model(images)

logits.shape  # (128, 10)
```

The largest logit determines the predicted class:

```python
predictions = jnp.argmax(logits, axis=-1)

predictions.shape  # (128,)
```

Use `jax.nn.softmax` only when probabilities are needed for inspection. The cross-entropy function below accepts
logits directly:

```python
probabilities = jax.nn.softmax(logits, axis=-1)
probabilities[0].sum()  # 1.0
```

## Loss and gradients

The loss function takes the model and batch as explicit arguments and returns a scalar:

```python
def loss_fn(model, images, labels):
    logits = model(images)
    losses = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    return losses.mean()
```

JAX differentiates this function with respect to its first argument:

```python
loss, grads = jax.value_and_grad(loss_fn)(model, images, labels)
```

`grads` has the same pytree structure as `model`, with one gradient array for each trainable `Param`. Gradients are
returned from the transformation; they are not stored on the model. There is no `ion.grad`.

## Optimizer updates

`ion.Optimizer` wraps an Optax transformation and creates its state from the model:

```python
optimizer = ion.Optimizer(optax.adam(3e-4), model)
model, optimizer = optimizer.update(model, grads)
```

Models and optimizers are immutable values. `update` returns their next versions rather than modifying either input.
Frozen parameters are excluded from the optimizer state automatically; see [Freezing](../workflows.md#freezing).

## Compiling the training step

The forward pass, differentiation, and update form one ordinary function:

```python
@jax.jit
def train_step(model, optimizer, images, labels):
    loss, grads = jax.value_and_grad(loss_fn)(model, images, labels)
    model, optimizer = optimizer.update(model, grads)
    return model, optimizer, loss
```

The first call traces and compiles the function for the model structure, input shapes, and dtypes it receives. Later
batches with the same shapes and dtypes reuse the compiled program:

```python
model, optimizer, loss = train_step(model, optimizer, images, labels)
```

Keeping a fixed batch size avoids compiling a separate program for a shorter final batch.

## Training on MNIST

The outer data loop stays in Python. Each batch is converted and passed to the compiled step:

```python
batch_size = 128
num_batches = len(train_images) // batch_size

for epoch in range(5):
    key = jax.random.key(epoch)
    indices = jax.random.permutation(key, len(train_images))

    for i in range(num_batches):
        batch_indices = indices[i * batch_size : (i + 1) * batch_size]
        images = jnp.asarray(train_images[batch_indices], dtype=jnp.float32) / 255.0
        labels = jnp.asarray(train_labels[batch_indices])

        model, optimizer, loss = train_step(model, optimizer, images, labels)
```

The complete example trains for five epochs and reaches about 99% test accuracy. These are the values from one
recorded run:

<iframe
  class="nn-plot nn-plot--training"
  src="../../assets/nn-mnist-training.html"
  title="MNIST training loss and test accuracy over five epochs"
  loading="lazy"
></iframe>

The exact values vary with hardware and library versions. The model, loss, gradient, update, and compilation pattern
remains the same for other supervised tasks.

## Evaluation and randomness

Evaluation is another function over the model and arrays:

```python
@jax.jit
def accuracy(model, images, labels):
    logits = model(images)
    predictions = jnp.argmax(logits, axis=-1)
    return (predictions == labels).mean()


test_accuracy = accuracy(
    model,
    jnp.asarray(test_images, dtype=jnp.float32) / 255.0,
    jnp.asarray(test_labels),
)
```

This CNN is deterministic. Stochastic layers such as `Dropout` take a key at call time instead of reading global random
state or a global training mode:

```python
dropout = nn.Dropout(0.1)

training_x = dropout(x, key=jax.random.key(1))
evaluation_x = dropout(x, deterministic=True)
```

Models containing stochastic layers pass their key through `__call__` and the loss function. See the
[Dropout reference](layers/dropout.md) for the complete call contract.

## Array conventions

Pointwise layers such as `Linear`, `LayerNorm`, and `MLP` transform the final feature axis and preserve arbitrary
leading dimensions:

```python
linear = nn.Linear(64, 128, key=jax.random.key(2))

linear(jnp.ones((64,))).shape  # (128,)
linear(jnp.ones((32, 64))).shape  # (32, 128)
linear(jnp.ones((32, 16, 64))).shape  # (32, 16, 128)
```

Structural layers assign specific meanings to the leading axes:

| Domain | Input layout |
|---|---|
| Vector data | `(batch, features)` |
| Images | `(batch, height, width, channels)` |
| Sequences | `(batch, length, channels)` |
| Attention | `(batch, sequence, dimension)` |
| Recurrent | `(batch, time, features)` |

Each [layer reference](layers/index.md) documents any stricter rank or batching requirements. `jax.vmap` can add another
leading batch dimension where a structural layer accepts exactly one:

```python
y = jax.vmap(rnn)(x)  # x.shape == (groups, batch, time, features)
```

Layer constructors use JAX's default floating dtype and do not take a `dtype` argument. Cast parameters after
construction when another precision is needed:

```python
model = model.astype(jnp.bfloat16)
```

See [Mixed precision](../workflows.md#mixed-precision) for activation and optimizer considerations.

## Further examples

- [CNN on MNIST](../examples/cnn-mnist.md) contains the complete loader, training loop, evaluation, and recorded output
  used by this guide.
- [RNN on MNIST](../examples/rnn-mnist.md) treats each image as a length-784 pixel sequence.
- [VAE on MNIST](../examples/vae-mnist.ipynb) adds a stochastic latent space and reparameterized sampling.
- [Ion Tour](../examples/ion-tour.ipynb) covers module surgery, freezing, optimization, and checkpointing in a notebook.
