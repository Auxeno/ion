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
    Weight initializer for every linear layer. He normal by default.
b_init : Initializer
    Bias initializer. Zeros by default.
key : PRNGKeyArray
    RNG key, split across the layers. Keyword-only.

Attributes
----------
layers : list[Linear]
    The linear layers, in order.
activation : Callable
    Hidden-layer activation.
final_activation : Callable | None
    Output activation, if any.

Examples
--------
>>> mlp = nn.MLP([3, 64, 64, 1], key=key)
>>> y = mlp(x)  # (*, 3) -> (*, 1)
