<div align="center">

  <h1><img src="assets/logo.png" alt="Ion" width="72"><br>Ion</h1>

  <h3>Simple neural networks in JAX</h3>

  [![Python](https://img.shields.io/badge/Python-≥3.11-636EFA.svg)](https://www.python.org/)
  [![JAX](https://img.shields.io/badge/JAX-≥0.5-AB63FA.svg)](https://github.com/google/jax)
  [![License](https://img.shields.io/badge/License-Apache_2.0-FFA15A.svg)](LICENSE)

</div>

---

Ion is a neural network library for JAX. Models are represented as [pytrees](https://docs.jax.dev/en/latest/pytrees.html) with explicit parameters and native compatibility with JAX transformations (`jax.jit`, `jax.grad`, `jax.vmap`). It ships with most standard neural network layers and supports custom layer definition via `nn.Module` subclassing.

## Installation

```bash
pip install git+https://github.com/auxeno/ion
```

## Overview

Ion is designed to be minimal. The core has three concepts: `Module`, `Param`, and `apply_updates`. Everything else is just JAX. Models are pytrees, so native JAX transforms like `jax.grad`, `jax.vmap`, and `jax.jit` work directly.

Visit the [Ion Tour Notebook](https://nbviewer.org/github/auxeno/ion/blob/main/examples/ion_tour.ipynb) for a hands-on walkthrough.

### Module

Inherit from `nn.Module` to define a model. Subclasses are automatically registered as JAX pytrees and become immutable after initialization (`__init__`).

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
```

Because modules are immutable after construction, use the `replace` method to create a modified copy.

```python
model = model.replace(activation=jax.nn.gelu)  # swap activation function
```

Non-array fields (ints, floats, strings, callables) are automatically treated as static metadata. They are preserved during pytree flattening and unflattening but remain invisible to JAX tracing. This prevents `jax.jit` tracing errors, allowing config settings (e.g., `num_heads`, `use_bias`) and activation functions to be stored directly on the module with no issues.

### Param

`Param` wraps an array and marks it as a model parameter. It controls whether the array is trainable or frozen.

```python
self.w = nn.Param(w_init(key=key, shape=(3, 16)))  # trainable
self.b = nn.Param(jnp.zeros(16), trainable=False)  # frozen
```

`Param` implements `__jax_array__` and forwards arithmetic operations, allowing it to function as a drop-in replacement for standard arrays (e.g., `x @ self.w` evaluates without explicit unwrapping).

Setting `trainable=False` applies `jax.lax.stop_gradient` under the hood, making the parameter a constant for differentiation. `jax.grad` naturally produces zero gradients for frozen parameters, and XLA skips backward computation through them. All native JAX transforms work directly: `jax.grad`, `jax.jacobian`, `jax.hessian`, etc.

Access all parameters via the `params` property. This returns a pytree with identical structure, containing only `Param` leaves (non-parameter leaves are replaced with `None`):

```python
model.params      # pytree of Param leaves, passable to an Optax optimizer
model.num_params  # total number of parameters (trainable and frozen)
```

Freeze the parameters of entire models or specific sub-modules:

```python
frozen_model = model.freeze()                          # freeze everything
model = model.replace(encoder=model.encoder.freeze())  # freeze one layer
unfrozen_model = model.unfreeze()                      # unfreeze everything
```

### Gradients

Because Ion models are standard pytrees and `Param` implements `__jax_array__`, all native JAX transforms work directly:

```python
def loss_fn(model, x, y):
    logits = model(x)
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

loss, grads = jax.value_and_grad(loss_fn)(model, x, y)
```

Frozen parameters (`trainable=False`) automatically have `stop_gradient` applied, so their gradients are zero and the backward pass skips them entirely.

There are no custom wrappers like `ion.jit`, `ion.vmap`, or `ion.scan`. Because Ion models are standard pytrees, all native JAX transformations work directly out of the box.

See [Internals](docs/internals.md) for how the module system and pytree registration work under the hood.

### Apply Updates

`apply_updates` only modifies `Param` leaves. Non-parameter arrays (like batch statistics) pass through unchanged.

```python
updates, opt_state = optimizer.update(grads, opt_state)
model = ion.apply_updates(model, updates)
```

That's the entire core.

### Layers

Ion ships with standard neural network layers built on the core. Each is a `Module` with trainable `Param` leaves. Under the hood these are just a JAX pytrees.

| Category        | Layers                                                                    |
|-----------------|---------------------------------------------------------------------------|
| Linear          | `Linear`, `Identity`, `LoRALinear`                                        |
| Convolution     | `Conv`, `ConvTranspose`                                                   |
| Attention       | `SelfAttention`, `CrossAttention`                                         |
| Normalization   | `LayerNorm`, `RMSNorm`, `GroupNorm`, `BatchNorm`                          |
| Recurrent       | `LSTMCell`, `GRUCell`, `LSTM`, `GRU`                                     |
| Pooling         | `MaxPool`, `AvgPool`                                                      |
| Embedding       | `Embedding`, `LearnedPositionalEmbedding`                                 |
| Positional      | `sinusoidal`, `rope`, `apply_rope`, `alibi`                               |
| Regularization  | `Dropout`                                                                 |
| Blocks          | `Sequential`, `MLP`, `TransformerBlock`, `CrossTransformerBlock`          |

See [Layer Conventions](docs/layers.md) for data format, weight init, and spatial layer usage.

### Pretty Printing

In notebook environments, Ion utilizes [Treescope](https://github.com/google-deepmind/treescope) for interactive, color-coded visualization. Modules also have a built-in text formatter for standard terminal `repr` output.

```python
>>> model = MLP(key=jax.random.key(0))
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
```

Treescope integration is enabled by default and can be toggled via `ion.enable_treescope()` / `ion.disable_treescope()`. 

Interact with models using Treescope by running locally or in [Colab](https://colab.research.google.com/github/auxeno/ion/blob/main/examples/ion_tour.ipynb).

### Serialization

```python
ion.save("model.npz", model)
model = ion.load("model.npz", model)
```

## Example

```python
import jax
import optax

import ion
from ion import nn


class MLP(nn.Module):
    layer_1: nn.Linear
    layer_2: nn.Linear

    def __init__(self, *, key):
        keys = jax.random.split(key, 2)
        self.layer_1 = nn.Linear(784, 128, key=keys[0])
        self.layer_2 = nn.Linear(128, 10, key=keys[1])

    def __call__(self, x):
        x = jax.nn.relu(self.layer_1(x))
        return self.layer_2(x)


def loss_fn(model, x, y):
    logits = model(x)
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()


@jax.jit
def train_step(model, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(model, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = ion.apply_updates(model, updates)
    return model, opt_state, loss


model = MLP(key=jax.random.key(0))

optimizer = optax.adam(3e-4)
opt_state = optimizer.init(model)

for x, y in data:
    model, opt_state, loss = train_step(model, opt_state, x, y)
```

## Examples

- [Ion Tour](examples/ion_tour.ipynb): Hands-on walkthrough of the core API
- [CNN Demo](examples/cnn_mnist.py): Image classification with convolutional networks
- [RNN Demo](examples/rnn_mnist.py): Sequence classification with recurrent networks
- [VAE Demo Notebook](examples/vae_mnist.ipynb): Variational autoencoder for image generation
- [PPO Demo](examples/ppo_gymnax.py): Reinforcement learning with Gymnax

## License

Released under the Apache License 2.0.

## Citation

To cite this repository:

```bibtex
@software{ion,
  title = {Ion: Simple Neural Networks in JAX},
  author = {Alex Goddard},
  url = {https://github.com/auxeno/ion},
  year = {2026}
}
```