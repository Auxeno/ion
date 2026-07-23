Chains single-argument layers, applying them in order.

Each layer's output feeds the next. Layers that accept a `key` (like `Dropout`) receive a freshly split one when a `key` is passed at call time; the rest are called with just their input.

Parameters
----------
*layers : Callable
    The layers or callables to apply in order. Any callable taking one array
    and returning one array works, not only `Module`s.

Attributes
----------
layers : tuple[Callable, ...]
    The chained layers, in order.

Example
-------
```python
model = nn.Sequential(
    nn.Linear(3, 16, key=keys[0]),
    nn.Dropout(0.1),
    nn.Linear(16, 1, key=keys[1]),
)
y = model(x, key=key)  # (*, 3) -> (*, 1)
```
