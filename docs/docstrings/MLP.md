Multi-layer perceptron.

Builds a stack of `Linear` layers from a list of dimensions, with an activation between hidden layers and an optional activation on the output.

Parameters
----------
dims : Sequence[int]
    Layer sizes from input to output. `[in, h1, ..., out]` creates
    `len(dims) - 1` linear layers.
activation : Callable, default=jax.nn.relu
    Activation applied after each hidden layer.
final_activation : Callable | None, default=None
    Activation applied to the output. `None` leaves the output linear.
bias : bool, default=True
    Whether the linear layers include bias terms.
w_init : Initializer
    Weight initializer for every linear layer. He uniform by default.
b_init : Initializer
    Bias initializer. Zeros by default.
key : jax.Array
    RNG key, split across the layers. Keyword-only.

Attributes
----------
layers : list[Linear]
    The linear layers, in order.
activation : Callable
    Hidden-layer activation.
final_activation : Callable | None
    Output activation, if any.

Example
-------
```python
batch, in_dim, hidden_dim, out_dim = 32, 3, 64, 1

mlp = nn.MLP([in_dim, hidden_dim, hidden_dim, out_dim], key=key)
x = jnp.ones((batch, in_dim))
y = mlp(x)  # (32, 3) -> (32, 1)
```
