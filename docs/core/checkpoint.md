# Checkpointing

`save` and `load` persist any pytree (models, optimizer states, tuples of both) to `.ion` files.

::: ion.save

::: ion.load

## Format

The format is [safetensors](https://huggingface.co/docs/safetensors)-style: an 8-byte little-endian header length, a JSON header mapping tensor names to `{dtype, shape, data_offsets}`, then raw little-endian tensor bytes. Safetensors tools can read Ion checkpoints directly.

Tensor names are tree paths (`blocks[2].attn.w_q`), the same paths used by `Module.at`. Ion-specific state lives in the header's `__metadata__` entry: a format version and a JSON map of `Param` trainable flags. All JAX dtypes are stored natively, including `bfloat16`, `float8` and `complex64` (via `ml_dtypes`); `complex128` is not representable in the format and raises on save.

`load` takes a reference pytree that supplies the tree structure and all non-array leaves; arrays and trainable flags come from the file. Loading validates the container (header size, tensor offsets must exactly tile the data section) and the structure (missing or extra keys and shape mismatches raise `ValueError`, dtype mismatches warn and keep the saved dtype).

Non-array fields (ints, strings, activation functions) come from the reference tree, not the file. If you change an activation in code and load an old checkpoint, you get the new activation with the old weights, with no warning. See [Sharp Edges](../guides/sharp-edges.md).
