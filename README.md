<div align="center">

  <h1><img src="https://raw.githubusercontent.com/auxeno/ion/main/assets/logo-transparent.png" alt="Ion" width="72"><br>Ion</h1>

  <h3>A simple library for neural and graph networks in JAX.</h3>

[![Python](https://img.shields.io/badge/Python-3.11+-7C3AED.svg)](https://www.python.org/)
[![PyPI](https://img.shields.io/pypi/v/ion-nn?color=478AF5)](https://pypi.org/project/ion-nn/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json&color=313131&labelColor=555555)](https://github.com/astral-sh/ruff)
[![CI](https://github.com/auxeno/ion/actions/workflows/ci.yml/badge.svg)](https://github.com/auxeno/ion/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/auxeno/ion/graph/badge.svg)](https://codecov.io/gh/auxeno/ion)

</div>

---

Ion is a simple neural network library for JAX. The core introduces three concepts (`Module`, `Param`, `Optimizer`) that make it simple to build and train neural networks. Models are [pytrees](https://docs.jax.dev/en/latest/pytrees.html) that *always* work directly with `jax.grad`, `jax.jit`, and `jax.vmap`. Ion also ships neural and graph network layers built on the core.

```bash
pip install ion-nn
```

The [documentation](https://auxeno.github.io/ion/) covers the core, layers, and common workflows in full.

## Example

A model built from Ion's standard layers, trained with native JAX transforms:

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
def train_step(model, optimizer, x, y):
    grads = jax.grad(loss_fn)(model, x, y)
    model, optimizer = optimizer.update(model, grads)
    return model, optimizer


model = MLP(key=jax.random.key(0))

optimizer = ion.Optimizer(optax.adam(3e-4), model)

for x, y in data:
    model, optimizer = train_step(model, optimizer, x, y)
```

## Documentation

- [Overview](https://auxeno.github.io/ion/overview/) -- the core abstractions and design
- [Core](https://auxeno.github.io/ion/core/module/) -- `Module`, `Param`, and `Optimizer`
- [NN guide](https://auxeno.github.io/ion/nn/guide/) and [GNN guide](https://auxeno.github.io/ion/gnn/guide/) -- array formats and shared conventions
- [Workflows](https://auxeno.github.io/ion/workflows/) -- freezing, mixed precision, serialization, inspecting models
- [Sharp edges](https://auxeno.github.io/ion/sharp-edges/) -- known constraints and gotchas
- [Examples](https://auxeno.github.io/ion/examples/) -- end-to-end training scripts and notebooks

## Layers

Ion ships with standard neural network layers. Each is a `Module` with trainable `Param` leaves.

| Category        | Layers                                                                    |
|-----------------|---------------------------------------------------------------------------|
| Linear          | `Linear`, `LoRALinear`                                                    |
| Convolution     | `Conv`, `ConvTranspose`                                                   |
| Attention       | `SelfAttention`, `CrossAttention`                                         |
| Normalization   | `LayerNorm`, `RMSNorm`, `GroupNorm`                                       |
| Recurrent       | `RNNCell`, `LSTMCell`, `GRUCell`, `RNN`, `LSTM`, `GRU`                    |
| SSM             | `LRUCell`, `S4DCell`, `S5Cell`, `LRU`, `S4D`, `S5`                        |
| Pooling         | `MaxPool`, `AvgPool`                                                      |
| Embedding       | `Embedding`, `LearnedPositionalEmbedding`                                 |
| Positional      | `RoPE`, `sinusoidal`, `alibi`                                             |
| Regularization  | `Dropout`                                                                 |
| Identity        | `Identity`                                                                |
| Blocks          | `Sequential`, `MLP`                                                       |
| GNN             | `GCNConv`, `GATConv`, `GATv2Conv`, `GINConv`                              |

See the [NN guide](https://auxeno.github.io/ion/nn/guide/) and [GNN guide](https://auxeno.github.io/ion/gnn/guide/) for array formats, spatial layers, and shared conventions.

## License

Released under the Apache License 2.0.

## Citation

To cite this repository:

```bibtex
@software{ion,
  title = {Ion: A simple library for neural and graph networks in JAX.},
  author = {Alex Goddard},
  url = {https://github.com/auxeno/ion},
  year = {2026}
}
```
</content>
</invoke>
