Base class for immutable, pytree-native models.

Subclass `Module`, annotate its stored fields, and assign them during
`__init__`. Ion converts the subclass to a dataclass, registers it as a JAX
pytree, and freezes each instance after construction.

Example
-------
```python
batch, in_dim, hidden_dim, out_dim = 32, 3, 16, 1

class MLP(nn.Module):
    up: nn.Linear
    down: nn.Linear

    def __init__(self, in_dim, hidden_dim, out_dim, *, key):
        key_up, key_down = jax.random.split(key)
        self.up = nn.Linear(in_dim, hidden_dim, key=key_up)
        self.down = nn.Linear(hidden_dim, out_dim, key=key_down)

    def __call__(self, x):
        return self.down(jax.nn.relu(self.up(x)))

model = MLP(in_dim, hidden_dim, out_dim, key=key)
x = jnp.ones((batch, in_dim))
y = model(x)  # (32, 3) -> (32, 1)
```
