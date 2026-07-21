# Ion

A simple neural network library for JAX. The core is three concepts (`Module`, `Param`, `Optimizer`) in under 1000 lines of code. Models are [pytrees](https://docs.jax.dev/en/latest/pytrees.html) that always work directly with `jax.grad`, `jax.jit`, and `jax.vmap`. Ion also ships with standard neural network layers (linear, convolution, attention, normalization, recurrent, and more) built on the core.

```bash
pip install ion-nn
```

## Quick example

```python
import jax
import ion.nn as nn

key = jax.random.key(0)
model = nn.MLP([4, 64, 64, 2], key=key)

x = jax.numpy.ones((32, 4))
y = model(x)  # (32, 4) -> (32, 2)
```

## Where to go next

- [Core](core/module.md): how `Module`, `Param`, and `Optimizer` work.
- [NN](nn/index.md): the layer library and its [reference](nn/reference.md).
- [GNN](gnn/index.md): graph neural network layers and ops.
- [Examples](examples.md): end-to-end training scripts and notebooks.
