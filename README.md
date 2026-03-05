<div align="center">

  <h1><img src="assets/logo.png" alt="Ion" width="96"><br>Ion</h1>

  <h3>A minimal JAX neural network library</h3>

  [![Python](https://img.shields.io/badge/Python-≥3.11-636EFA.svg)](https://www.python.org/)
  [![JAX](https://img.shields.io/badge/JAX-≥0.5-AB63FA.svg)](https://github.com/google/jax)
  [![License](https://img.shields.io/badge/License-Apache_2.0-FFA15A.svg)](LICENSE)

</div>

---

Ion is a neural network library for JAX designed to be minimal and easy to use. Models are pytrees, parameters are explicit, and layers follow familiar conventions.

It ships with layers for convolution, attention, recurrence, normalization, and pooling, along with filtered transforms for working with models directly. Users can write their own if they so wish.

## Installation

```bash
pip install git+https://github.com/auxeno/ion
```

## Example

```python
import jax, optax

import ion
from ion import nn


class MLP(nn.Module):
    layer_1: nn.Linear
    layer_2: nn.Linear
    activation: Callable

    def __init__(self, *, key):
        keys = jax.random.split(key, 2)
        self.layer_1 = nn.Linear(784, 128, key=keys[0])
        self.layer_2 = nn.Linear(128, 10, key=keys[1])
        self.activation = jax.nn.relu

    def __call__(self, x):
        x = self.activation(self.layer_1(x))
        return self.layer_2(x)

@ion.grad
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
opt_state = optimizer.init(model.params)

for x, y in data:
    model, opt_state = train_step(model, opt_state, x, y)
```

> Ion models are JAX [pytrees](https://docs.jax.dev/en/latest/pytrees.html), meaning native compatibility with `jax.jit`, `jax.grad`, and `jax.vmap`.
>
> The `ion.grad` transformation extends JAX's autograd with a filtering layer. It automatically identifies and differentiates only the `ion.nn.Param` leaves where `trainable=True`.

## License

Released under the Apache License 2.0.
