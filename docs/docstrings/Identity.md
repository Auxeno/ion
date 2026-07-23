Pass-through layer that returns its first argument unchanged and ignores the rest.

Useful as a placeholder for optional layers (e.g. a normalization slot that can be switched off) or as a residual stub, so surrounding code needs no conditional.

Example
-------
```python
batch, seq, dim = 4, 16, 64

identity = nn.Identity()
x = jnp.ones((batch, seq, dim))
y = identity(x)  # (4, 16, 64) -> (4, 16, 64)
```
