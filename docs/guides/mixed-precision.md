# Mixed Precision

Layer constructors take no `dtype` argument. Parameters are created in JAX's default float dtype (`float32`, or `float64` when `jax_enable_x64` is set), and precision is controlled entirely through `astype`. There are three patterns.

## Default (float32)

Build and train with no dtype handling at all. This is the right choice unless memory or throughput demands otherwise.

## Mixed precision (bfloat16)

Keep float32 master params and cast the model to `bfloat16` *inside* the loss:

```python
def loss_fn(model, x, y):
    model = ion.astype(model, jnp.bfloat16)
    logits = model(x)
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
```

The cast is differentiable, so gradients return in float32 to match the master params and the optimizer state; only the forward and backward math runs in bfloat16. This mirrors Keras `mixed_bfloat16`, PyTorch AMP, and Flax's `param_dtype`/`dtype` split. See [`examples/gpt_tinystories.ipynb`](https://github.com/auxeno/ion/blob/main/examples/gpt_tinystories.ipynb) for a worked example.

## Full bfloat16 inference

Cast once after construction and keep the low-precision model:

```python
model = model.astype(jnp.bfloat16)
```

A `bfloat16` model applied to `float32` inputs silently upcasts every result back to `float32` under JAX type promotion, so cast both the model and its inputs. See [Sharp Edges](sharp-edges.md) for the details.

## Reference

`astype` also exists as a method (`model.astype(dtype)`) and targets a subtree through [`at`](../core/module.md) (`model.at.encoder.astype(dtype)`). It only touches float leaves, so integer buffers and indices are left alone.

::: ion.astype
