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

## Package Contents

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

Access all parameters via the `params` property. This returns a pytree with identical structure, containing only `Param` leaves (non-parameter leaves are replaced with `None`):

```python
model.params       # pytree of Param leaves — can be passed to an Optax optimizer
model.num_params   # total number of parameters (trainable and frozen)
```

Freeze the parameters of entire models or specific sub-modules:

```python
frozen_model = model.freeze()                          # freeze everything
model = model.replace(encoder=model.encoder.freeze())  # freeze one layer
unfrozen_model = model.unfreeze()                      # unfreeze everything
```

### Transforms

`ion.grad` and `ion.value_and_grad` replicate `jax.grad`, but differentiate strictly with respect to trainable `Param` leaves. All other fields are treated as constants.

```python
@ion.grad
def loss_fn(model, x, y):
    logits = model(x)
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

grads = loss_fn(model, x, y)  # grads has same structure as model
```

The gradient tree matches the model structure — trainable `Param` positions have gradients, while everything else is `None`. Standard `jax.grad` also works natively with Ion modules, but it will compute gradients with respect to all JAX arrays in the tree rather than isolating the trainable parameters.

There are no custom wrappers like `ion.jit`, `ion.vmap`, or `ion.scan`. Because Ion models are standard pytrees, all native JAX transformations work directly out of the box.

### Layers

| Category        | Layers                                                                    |
|-----------------|---------------------------------------------------------------------------|
| Linear          | `Linear`, `Identity`, `LoRALinear`                                        |
| Convolution     | `Conv1d`, `Conv2d`, `Conv`, `ConvTranspose1d`, `ConvTranspose2d`, `ConvTranspose` |
| Attention       | `SelfAttention`, `CrossAttention`                                         |
| Normalization   | `LayerNorm`, `RMSNorm`, `GroupNorm`, `BatchNorm`, `InstanceNorm`          |
| Recurrent       | `LSTMCell`, `GRUCell`, `LSTM`, `GRU`                                     |
| Pooling         | `MaxPool1d`, `MaxPool2d`, `AvgPool1d`, `AvgPool2d`                       |
| Upsampling      | `Upsample1d`, `Upsample2d`                                               |
| Embedding       | `Embedding`, `LearnedPositionalEmbedding`                                 |
| Positional      | `sinusoidal`, `rope`, `apply_rope`, `alibi`                               |
| Regularization  | `Dropout`                                                                 |
| Blocks          | `Sequential`, `MLP`, `TransformerBlock`, `CrossTransformerBlock`          |

### Pretty Printing

Modules have a built-in text formatter for standard terminal `repr` output. In notebook environments, Ion utilizes [Treescope](https://github.com/google-deepmind/treescope) for interactive, color-coded visualization.

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


@ion.value_and_grad
def loss_fn(model, x, y):
    logits = model(x)
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()


@jax.jit
def train_step(model, opt_state, x, y):
    loss, grads = loss_fn(model, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = ion.apply_updates(model, updates)
    return model, opt_state, loss


model = MLP(key=jax.random.key(0))

optimizer = optax.adam(3e-4)
opt_state = optimizer.init(model.params)

for x, y in data:
    model, opt_state, loss = train_step(model, opt_state, x, y)
```

## License

Released under the Apache License 2.0.

## Citation

```bibtex
@software{ion,
  title  = {Ion: Simple Neural Networks in JAX},
  author = {Alex Goddard},
  url    = {https://github.com/auxeno/ion},
  year   = {2025}
}
```