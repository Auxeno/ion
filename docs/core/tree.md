# Tree Utilities

Functions for transforming a model's pytree of `Param` leaves: toggling trainability and changing precision. All are pure and return a new model, leaving the original untouched. `freeze` and `unfreeze` also exist as methods (`model.freeze()`), and `at` targets a subtree (`model.at.encoder.freeze()`).

::: ion.freeze

::: ion.unfreeze

::: ion.astype

::: ion.is_param

::: ion.is_trainable_param

## Precision

Layer constructors take no `dtype` argument. Parameters are created in JAX's default float dtype (`float32`, or `float64` when `jax_enable_x64` is set), and precision is controlled entirely through `astype`. There are three patterns:

- **Default (float32).** Build and train with no dtype handling.
- **Mixed precision.** Keep float32 master params and cast the model to `bfloat16` *inside* the loss with `ion.astype(model, jnp.bfloat16)`. The cast is differentiable, so gradients return in float32 to match the master params and the optimizer state; only the forward/backward math runs in bfloat16. This mirrors Keras `mixed_bfloat16`, PyTorch AMP, and Flax's `param_dtype`/`dtype` split. See [`examples/gpt_tinystories.ipynb`](https://github.com/auxeno/ion/blob/main/examples/gpt_tinystories.ipynb) for a worked example.
- **Full bfloat16 inference.** Cast once after construction with `model.astype(jnp.bfloat16)`.

A `bfloat16` model applied to `float32` inputs silently upcasts every result back to `float32` under JAX type promotion, so cast both the model and its inputs. See [Sharp Edges](../guides/sharp-edges.md) for the details.
