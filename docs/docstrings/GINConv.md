Graph Isomorphism Network layer ([Xu et al., 2019](https://arxiv.org/abs/1810.00826)).

Sum-aggregates neighbor features and applies a caller-supplied MLP to `(1 + eps) * x + aggregated`. Sum aggregation preserves neighbor multiplicity, making GIN as discriminative as the Weisfeiler-Lehman graph isomorphism test.

Parameters
----------
mlp : Module
    Update network applied after aggregation. Supplies all of the layer's
    weights, so `GINConv` takes no `key` and creates none of its own.
eps : float, default=0.0
    Weights a node's own features against its aggregated neighbors. Fixed unless
    `train_eps=True`.
train_eps : bool, default=False
    If `True`, `eps` becomes a learnable scalar `Param`.

Attributes
----------
mlp : Module
    The update network passed at construction.
eps : Param | float
    Learnable scalar when `train_eps=True`, otherwise the fixed float.

Warning
-------
Do not add self-loops: a node's own features already enter through the `(1 + eps)` term, so adding them double-counts.

Example
-------
```python
gin = gnn.GINConv(nn.MLP([16, 32, 32], key=key))
y = gin(x, senders, receivers)                     # (n, 16) -> (n, 32)

gin = gnn.GINConv(nn.MLP([16, 32, 32], key=key), train_eps=True)
```
