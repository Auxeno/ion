# State Space

Deep state space model layers. `S4D`, `S5`, and `LRU` process a sequence with a linear complex-valued recurrence, returning outputs plus the final state; the matching `*Cell` classes apply a single timestep. The output dimension is always `in_dim`.

::: ion.nn.S4D

::: ion.nn.S5

::: ion.nn.LRU

::: ion.nn.S4DCell

::: ion.nn.S5Cell

::: ion.nn.LRUCell

---

## State Representation

SSM matrix parameters use the literature's uppercase names (`A`, `B`, `C`,
`D`).

The recurrent state is complex-valued. S4D and S5 use conjugate-pair structure:
`state_dim=N` stores `N//2` complex eigenvalues and the readout uses
`2*Re(...)` to recover each pair's full contribution. LRU instead stores
`hidden_dim` independent complex eigenvalues without conjugate symmetry.

## Recurrent State

Sequence layers default to a zero complex state. Pass `hx` to start from a
custom state or continue the recurrence across chunks.

```python
s4d = nn.S4D(3, 8, key=key)
outputs, h = s4d(x)
outputs, h = s4d(x_next, hx=h)

s5 = nn.S5(3, 8, key=key)
outputs, h = s5(x)
outputs, h = s5(x_next, hx=h)
```

Cells expose an unbatched `initial_state` property. S4D and S5 states have
length `state_dim//2`; LRU state has length `hidden_dim`.
