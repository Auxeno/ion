# RNN on MNIST

Sequential MNIST: each 28x28 image is flattened into a length-784 sequence of single pixels and read one pixel at a time. Classifying it correctly demands carrying information across hundreds of steps, so it is a standard long-range dependency benchmark. A `GRU` reads the sequence and its final hidden state feeds a `Linear` classifier.

Points of interest:

- The image is reshaped to `(b, 784, 1)`, one scalar per timestep, before it reaches the model.
- `nn.GRU` returns `(outputs, final_state)`; only the final hidden state `h` is used for classification.
- The optimizer is `optax.chain(clip_by_global_norm(1.0), adam(lr))`, gradient clipping being the usual stabilizer for recurrent training.

## Source

[examples/rnn_mnist.py](https://github.com/auxeno/ion/blob/main/examples/rnn_mnist.py) on GitHub.

```python title="examples/rnn_mnist.py" linenums="1"
--8<-- "examples/rnn_mnist.py"
```

## Output

```bash
uv run --group examples examples/rnn_mnist.py
```
