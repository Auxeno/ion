# Quickstart

This page builds and trains a small model using Ion and native JAX
transformations.

## Installation

Ion requires Python 3.11 or newer.

```bash
pip install ion-nn
```

This installs the standard JAX dependency, suitable for CPU use. For NVIDIA
GPU, AMD GPU, or TPU support, follow the official [JAX installation
guide](https://docs.jax.dev/en/latest/installation.html), then install Ion.

## Create a model

Import JAX, Optax, and Ion, then construct a multi-layer perceptron:

```python
import jax
import jax.numpy as jnp
import optax

import ion
from ion import nn

# Two input features, four hidden units, one output
model = nn.MLP([2, 4, 1], key=jax.random.key(0))
```

Printing the model shows its complete structure:

```pycon
>>> print(model)
MLP(
  layers=(
    Linear(
      w=Param(f32[2, 4], trainable=True),
      b=Param(f32[4], trainable=True),
    ),
    Linear(
      w=Param(f32[4, 1], trainable=True),
      b=Param(f32[1], trainable=True),
    ),
  ),
  activation=relu,
  final_activation=None,
)
```

The model has 17 scalar parameters:

```python
model.num_params  # 17
```

In IPython and Jupyter, evaluating `model` displays the same tree
interactively with Treescope.

## Run a forward pass

The following four examples have two input features and one target value:

```python
x = jnp.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
])

y = jnp.array([
    [0.0],
    [-2.0],
    [1.0],
    [-1.0],
])
```

Calling the model produces one prediction for each row:

```python
predictions = model(x)
predictions.shape  # (4, 1)
```

## Compute gradients

Define a mean squared error loss. The model is the first argument because JAX
differentiates with respect to the first argument by default:

```python
def loss_fn(model, x, y):
    predictions = model(x)
    return jnp.mean((predictions - y) ** 2)


loss, grads = jax.value_and_grad(loss_fn)(model, x, y)
float(loss)  # approximately 2.19
```

`grads` has the same tree structure as `model`, with one gradient array for
each trainable parameter:

```pycon
>>> print(grads)
MLP(
  layers=(
    Linear(
      w=Param(f32[2, 4], trainable=True),
      b=Param(f32[4], trainable=True),
    ),
    Linear(
      w=Param(f32[4, 1], trainable=True),
      b=Param(f32[1], trainable=True),
    ),
  ),
  activation=relu,
  final_activation=None,
)
```

Gradients are returned from `jax.value_and_grad`; they are not stored on the
model.

## Update the model

`ion.Optimizer` wraps an Optax transformation and initializes its state from
the model:

```python
optimizer = ion.Optimizer(optax.adam(1e-2), model)
model, optimizer = optimizer.update(model, grads)

float(loss_fn(model, x, y))  # approximately 2.14
```

Models and optimizers are immutable. `update` returns their next values instead
of modifying either input.

## Compile and train

The forward pass, differentiation, and optimizer update form one ordinary
function. `jax.jit` compiles the complete training step:

```python
@jax.jit
def train_step(model, optimizer, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(model, x, y)
    model, optimizer = optimizer.update(model, grads)
    return model, optimizer, loss


for _ in range(500):
    model, optimizer, loss = train_step(model, optimizer, x, y)
```

The trained predictions match the target values:

```python
print(float(loss))
print(jnp.round(model(x), 2))
```

```text
9.64e-13
[[ 0.]
 [-2.]
 [ 1.]
 [-1.]]
```

## Next steps

- [Overview](overview.md) explains Ion's core abstractions and how to build a
  custom `Module`.
- The [NN guide](nn/guide.md) trains a convolutional classifier on MNIST.
- The [layer reference](nn/layers/index.md) lists the available neural network
  layers.
- [Workflows](workflows.md) covers freezing, mixed precision, and
  checkpointing.
