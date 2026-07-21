Fully connected linear layer computing `x @ w + b`.

Parameters
----------
in_dim : int
    Input feature dimension.
out_dim : int
    Output feature dimension.
bias : bool, default=True
    Whether to include a learnable bias term.
w_init : Initializer
    Weight initializer. He normal by default, suited to ReLU networks.
b_init : Initializer
    Bias initializer. Zeros by default.
key : jax.Array
    RNG key for parameter initialization. Keyword-only.

Attributes
----------
w : Param
    Weight matrix of shape `(in_dim, out_dim)`.
b : Param | None
    Bias vector of shape `(out_dim,)`. `None` when `bias=False`.

Examples
--------
>>> linear = nn.Linear(3, 16, key=key)
>>> y = linear(x)  # (*, 3) -> (*, 16)
