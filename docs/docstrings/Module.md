Base class for all neural network modules.

Subclass `Module` and annotate fields with their types. Ion converts the subclass to a frozen dataclass, registers it as a JAX pytree, and freezes each instance after `__init__`. The result works directly with `jax.jit`, `jax.grad`, and `jax.vmap`, with no wrappers.

Notes
-----
Fields are classified once at construction. `Param`, `Module`, and array leaves (and containers of them) become dynamic pytree children; everything else (ints, strings, callables) becomes static metadata baked into the treedef. Instances are immutable after `__init__`, so build modified copies with `at` rather than assigning to fields.

Examples
--------
>>> class MLP(nn.Module):
...     up: nn.Linear
...     down: nn.Linear
...
...     def __init__(self, dim, hidden, *, key):
...         key_up, key_down = jax.random.split(key)
...         self.up = nn.Linear(dim, hidden, key=key_up)
...         self.down = nn.Linear(hidden, dim, key=key_down)
...
...     def __call__(self, x):
...         return self.down(jax.nn.relu(self.up(x)))
>>> model = MLP(16, 64, key=key)
>>> model = model.at.up.b.set(None)  # modified copy: drop the bias on `up`
