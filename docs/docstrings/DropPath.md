Stochastic depth ([Huang et al., 2016](https://arxiv.org/abs/1603.09382)).

Randomly drops a residual branch for whole samples with probability `p` and
scales the survivors by `1 / (1 - p)`, so activation magnitudes match between
training and inference. One decision covers each sample, whatever the input
rank: the mask is shared across every dimension after the batch dimension.

Parameters
----------
p : float
    Drop probability in `[0, 1]`.

Example
-------
```python
drop_path = nn.DropPath(0.1)
branch = mlp(x)
x = x + drop_path(branch, training=True, key=key)  # (8, 64)
```

The branch is still evaluated before its output is masked. To drop individual
elements instead, use `Dropout`.
