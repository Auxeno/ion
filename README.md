<div align="center">

  <h1><img src="https://raw.githubusercontent.com/auxeno/ion/main/assets/logo.png" alt="Ion" width="72"><br>Ion</h1>

  <h3>A minimal neural network library for JAX</h3>

[![Python](https://img.shields.io/badge/Python-3.11+-4E69FF.svg)](https://www.python.org/)
[![PyPI](https://img.shields.io/pypi/v/ion-nn?color=8E51FF)](https://pypi.org/project/ion-nn/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json&color=313131&labelColor=555555)](https://github.com/astral-sh/ruff)
[![CI](https://github.com/auxeno/ion/actions/workflows/ci.yml/badge.svg)](https://github.com/auxeno/ion/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/auxeno/ion/graph/badge.svg)](https://codecov.io/gh/auxeno/ion)

</div>

---

Ion is a library for building and training neural networks in JAX. The core is a minimal ~250 lines of code across three files. There are three concepts to learn (`Module`, `Param`, and `apply_updates`). Models are [pytrees](https://docs.jax.dev/en/latest/pytrees.html) with simple calling semantics `model(x)` that *always* work directly with `jax.grad`, `jax.jit`, `jax.vmap`, and every other JAX transform. Everything else is just JAX.

Ion also ships with standard neural network layers (linear, convolution, attention, normalization, recurrent, and more) built with the core.
<br><br>

> <details>
> <summary>Why do I need a neural network library in JAX?</summary>
> <br>
>
> Building *simple* NN models from scratch in JAX is straightforward. As they get more complex however, two things become painful: managing parameters (initializing them, tracking which are trainable, freezing some for fine-tuning) and composing modules (reusing layers, wiring them through JAX transforms, not reimplementing things like convolution padding from scratch). A neural network library takes care of this so you can focus on model building and training.
> </details>
<br>

> <details>
> <summary>How does Ion compare to Equinox and Flax?</summary>
> <br>
>
> **[Equinox](https://github.com/patrick-kidger/equinox)** is a pytree library for scientific computing where neural networks are one of several possible use-cases. It provides filtered transforms, partition/combine utilities, and general pytree tools that give users fine-grained control over how JAX interacts with their code. Ion is narrower in scope: three building blocks on top of JAX for defining and training neural networks, using native `jax.grad`/`jax.jit` directly with no additional API layer. Equinox is excellent, and particularly well-suited if your work extends beyond neural networks into broader scientific computing.
>
> **[Flax NNX](https://github.com/google/flax)** takes a different philosophical approach. NNX models are mutable graph objects with reference semantics, and custom transforms (`nnx.jit`, `nnx.grad`) that allow mutability within JAX's functional programming model. The result is flexible, PyTorch-like ergonomics at the cost of complexity behind the scenes (state extraction, reference threading, graph tracing) that can make it hard to reason about what your code is doing. Ion keeps things straightforward: immutable pytrees, native JAX transforms, and explicit state passing. NNX is a great choice if you value PyTorch-like ergonomics and are happy to trust the framework.
> </details>
<br>

## Installation

```bash
pip install ion-nn
```

## Core Concepts

There are three Ion-specific concepts you need to build and train models: `Param`, `Module` and `apply_updates`.

### Param

`Param` wraps an array and marks it as a model parameter, either trainable or frozen.

```python
w = nn.Param(jax.random.normal(shape=(3, 16), key=key))      # trainable
b = nn.Param(jax.numpy.zeros(shape=(16,)), trainable=False)  # frozen
```

`Param` acts as a drop-in for arrays via `__jax_array__` (e.g. `x @ w` works without unwrapping). Frozen params have `stop_gradient` applied, so `jax.grad` returns zero gradients at those positions and XLA skips their backward computation.

### Module

Inherit from `nn.Module` to define a layer. Subclasses are automatically registered as JAX pytrees and become immutable after `__init__`.

```python
import ion.nn as nn

class Linear(nn.Module):
    w: nn.Param
    b: nn.Param

    def __init__(self, in_dim, out_dim, *, key):
        self.w = nn.Param(jax.random.normal(shape=(in_dim, out_dim), key=key))
        self.b = nn.Param(jax.numpy.zeros(shape=(out_dim,)))

    def __call__(self, x):
        return x @ self.w + self.b
```

Non-array fields (ints, floats, strings, callables) are automatically treated as static metadata, invisible to JAX tracing. Config values like `num_heads` or `use_bias` and activation functions can be stored directly on the module.

### apply_updates

Adds optimizer deltas to trainable `Param` leaves only. Non-parameter arrays and frozen params pass through unchanged.

```python
updates, opt_state = optimizer.update(grads, opt_state)
model = ion.apply_updates(model, updates)
```

That's the entire core. See [Internals](https://github.com/auxeno/ion/blob/main/docs/internals.md) for design details and sharp edges.

## Example

Putting it all together with a model built from Ion's standard layers:

```python
import jax, optax, typing

import ion
import ion.nn as nn


class MLP(nn.Module):
    layer_1: nn.Linear
    layer_2: nn.Linear
    activation: typing.Callable

    def __init__(self, activation=jax.nn.relu, *, key):
        keys = jax.random.split(key, 2)
        self.layer_1 = nn.Linear(784, 128, key=keys[0])
        self.layer_2 = nn.Linear(128, 10, key=keys[1])
        self.activation = activation

    def __call__(self, x):
        return self.layer_2(self.activation(self.layer_1(x)))


def loss_fn(model, x, y):
    logits = model(x)
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()


@jax.jit
def train_step(model, opt_state, x, y):
    grads = jax.grad(loss_fn)(model, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = ion.apply_updates(model, updates)
    return model, opt_state


model = MLP(key=jax.random.key(0))

optimizer = optax.adam(3e-4)
opt_state = optimizer.init(model)

for x, y in data:
    model, opt_state = train_step(model, opt_state, x, y)
```

## Utilities

`nn.Module` provides convenience methods and properties for common operations. Methods return new instances, as modules are immutable.

```python
model.replace(activation=jax.nn.tanh)    # create a modified copy
model.freeze()                           # freeze all params
model.unfreeze()                         # unfreeze all params
model.replace(base=model.base.freeze())  # freeze a sub-module
model.astype(jax.numpy.bfloat16)         # cast params to a different dtype
model.params                             # pytree of Param leaves
model.num_params                         # total parameter count
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

See [Layer Conventions](https://github.com/auxeno/ion/blob/main/docs/layers.md) for data format, weight init, and spatial layer usage.

## Pretty Printing

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

## Serialization

Save and load model parameters as `.npz` files. `load` requires a model instance as a template to reconstruct the pytree structure.

```python
ion.save("model.npz", model)
model = ion.load("model.npz", model)
```

## Examples

- [Ion Tour](https://github.com/auxeno/ion/blob/main/examples/ion_tour.ipynb): Hands-on walkthrough of the core API
- [CNN Demo](https://github.com/auxeno/ion/blob/main/examples/cnn_mnist.py): Image classification with convolutional networks
- [RNN Demo](https://github.com/auxeno/ion/blob/main/examples/rnn_mnist.py): Sequence classification with recurrent networks
- [GPT Demo Notebook](https://github.com/auxeno/ion/blob/main/examples/gpt_tinystories.ipynb): Character-level GPT on TinyStories
- [VAE Demo Notebook](https://github.com/auxeno/ion/blob/main/examples/vae_mnist.ipynb): Variational autoencoder for image generation
- [PPO Demo](https://github.com/auxeno/ion/blob/main/examples/ppo_gymnax.py): Reinforcement learning with Gymnax

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
