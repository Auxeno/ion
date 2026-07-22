# CNN on MNIST

A convolutional classifier for handwritten digits. Two `Conv` layers with `MaxPool` downsampling feed two `Linear` layers, trained with Adam for five epochs to around 99% test accuracy.

Points of interest:

- The model is a plain `Module` subclass: annotate the submodules as fields, build them in `__init__`, and use them in `__call__`.
- `jax.random.split` gives one key per layer, passed as the keyword-only `key` argument.
- `train_step` is a normal `jax.jit`ed function taking the model and optimizer as arguments and returning their updated copies. No custom transform, no state object.
- `jax.value_and_grad(loss_fn)` differentiates through the model pytree directly, and `optimizer.update` returns both the new model and the new optimizer.

## Source

[examples/cnn_mnist.py](https://github.com/auxeno/ion/blob/main/examples/cnn_mnist.py) on GitHub.

```python title="examples/cnn_mnist.py" linenums="1"
--8<-- "examples/cnn_mnist.py"
```

## Output

```bash
uv run --group examples examples/cnn_mnist.py
```

```
Epoch 1/5: 100%|██████████| 468/468 [00:17<00:00, 26.07it/s]
  loss: 0.3003  test accuracy: 97.44%
Epoch 2/5: 100%|██████████| 468/468 [00:13<00:00, 33.83it/s]
  loss: 0.0801  test accuracy: 98.32%
Epoch 3/5: 100%|██████████| 468/468 [00:15<00:00, 30.56it/s]
  loss: 0.0580  test accuracy: 98.43%
Epoch 4/5: 100%|██████████| 468/468 [00:14<00:00, 32.32it/s]
  loss: 0.0450  test accuracy: 98.46%
Epoch 5/5: 100%|██████████| 468/468 [00:11<00:00, 41.33it/s]
  loss: 0.0372  test accuracy: 98.82%
```
