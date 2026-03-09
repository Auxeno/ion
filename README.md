<div align="center">

  <h1><img src="https://raw.githubusercontent.com/auxeno/ion/main/assets/logo.png" alt="Ion" width="72"><br>Ion</h1>

  <h3>Simple neural networks in JAX</h3>

  [![Python](https://img.shields.io/badge/Python-3.11+-636EFA.svg)](https://www.python.org/)
  [![JAX](https://img.shields.io/badge/JAX-0.5+-AB63FA.svg)](https://github.com/google/jax)
  [![License](https://img.shields.io/badge/License-Apache_2.0-FFA15A.svg)](LICENSE)
  [![CI](https://github.com/auxeno/ion/actions/workflows/ci.yml/badge.svg)](https://github.com/auxeno/ion/actions/workflows/ci.yml)

</div>

---

Ion is a minimal neural network library for JAX. The core is three files and ~250 lines of code, with three concepts to learn: `Module`, `Param`, and `apply_updates`. Everything else is just JAX. Models are [pytrees](https://docs.jax.dev/en/latest/pytrees.html), so all native JAX transforms (`jax.grad`, `jax.jit`, `jax.vmap`, etc.) work directly. There are no custom `ion.jit` or `ion.grad` wrappers. Ion ships with most standard neural network layers and supports custom layer definition via `nn.Module`.

## Installation

```bash
pip install ion-nn
```

## Core Concepts

Visit the [Ion Tour Notebook](https://nbviewer.org/github/Auxeno/ion/blob/main/examples/ion_tour.ipynb) for a hands-on walkthrough.

### Module

Inherit from `nn.Module` to define a model. Subclasses are automatically registered as JAX pytrees and become immutable after `__init__`.

```python
import ion.nn as nn

class MLP(nn.Module):
    layer_1: nn.Linear
    layer_2: nn.Linear
    activation: Callable

    def __init__(self, activation=jax.nn.relu, *, key):
        keys = jax.random.split(key, 2)
        self.layer_1 = nn.Linear(784, 128, key=keys[0])
        self.layer_2 = nn.Linear(128, 10, key=keys[1])
        self.activation = activation

    def __call__(self, x):
        x = self.activation(self.layer_1(x))
        return self.layer_2(x)
```

Non-array fields (ints, floats, strings, callables) are automatically treated as static metadata, invisible to JAX tracing. Config values like `num_heads` or `use_bias` and activation functions can be stored directly on the module with no issues.

Modules are immutable after construction. Use `replace` to create a modified copy:

```python
model = model.replace(activation=jax.nn.tanh)
```

### Param

`Param` wraps an array and marks it as a model parameter, either trainable or frozen.

```python
self.w = nn.Param(w_init(shape=(3, 16), key=key))  # trainable
self.b = nn.Param(jnp.zeros(16), trainable=False)  # frozen
```

`Param` acts as a drop-in for arrays via `__jax_array__` (e.g. `x @ self.w` works without unwrapping). Frozen params have `stop_gradient` applied, so `jax.grad` returns zero gradients at those positions and XLA skips their backward computation.

```python
model.freeze()                                         # freeze all params
model.unfreeze()                                       # unfreeze all params
model = model.replace(encoder=model.encoder.freeze())  # freeze a sub-module
```

```python
model.params      # pytree of Param leaves (non-Param leaves are None)
model.num_params  # total parameter count
```

### apply_updates

Adds optimizer deltas to trainable `Param` leaves only. Non-parameter arrays and frozen params pass through unchanged.

```python
updates, opt_state = optimizer.update(grads, opt_state)
model = ion.apply_updates(model, updates)
```

That's the entire core. See [Internals](docs/internals.md) for how it works under the hood.

## Example

Putting it together with the MLP defined above:

```python
import jax
import optax

import ion
import ion.nn as nn


@jax.grad
def loss_fn(model, x, y):
    logits = model(x)
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()


@jax.jit
def train_step(model, opt_state, x, y):
    grads = loss_fn(model, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = ion.apply_updates(model, updates)
    return model, opt_state


model = MLP(key=jax.random.key(0))

optimizer = optax.adam(3e-4)
opt_state = optimizer.init(model)

for x, y in data:
    model, opt_state = train_step(model, opt_state, x, y)
```

## Layers

Ion ships with standard neural network layers. Each is a `Module` with trainable `Param` leaves.

| Category        | Layers                                                                    |
|-----------------|---------------------------------------------------------------------------|
| Linear          | `Linear`, `Identity`, `LoRALinear`                                        |
| Convolution     | `Conv`, `ConvTranspose`                                                   |
| Attention       | `SelfAttention`, `CrossAttention`                                         |
| Normalization   | `LayerNorm`, `RMSNorm`, `GroupNorm`, `BatchNorm`                          |
| Recurrent       | `LSTMCell`, `GRUCell`, `LSTM`, `GRU`                                      |
| Pooling         | `MaxPool`, `AvgPool`                                                      |
| Embedding       | `Embedding`, `LearnedPositionalEmbedding`                                 |
| Positional      | `sinusoidal`, `rope`, `apply_rope`, `alibi`                               |
| Regularization  | `Dropout`                                                                 |
| Blocks          | `Sequential`, `MLP`, `TransformerBlock`, `CrossTransformerBlock`          |

See [Layer Conventions](docs/layers.md) for data format, weight init, and spatial layer usage.

### Pretty Printing

In notebooks, [Treescope](https://github.com/google-deepmind/treescope) provides interactive, color-coded visualization. Modules also have built-in text formatting for terminal output.

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
  activation=relu,
)
```

### Serialization

```python
ion.save("model.npz", model)
model = ion.load("model.npz", model)
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
