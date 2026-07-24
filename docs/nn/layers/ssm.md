# State Space

State space models process sequences with a linear complex-valued recurrence.
`S4D`, `S5`, and `LRU` use an associative scan with \(O(\log T)\) parallel
depth, compared with \(O(T)\) for RNNs, while total work remains \(O(T)\)
rather than the \(O(T^2)\) of standard attention.

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
`state_dim=N` stores \(N/2\) complex eigenvalues and the readout uses
\(2\operatorname{Re}(\cdot)\) to recover each pair's full contribution. LRU
instead stores `hidden_dim` independent complex eigenvalues without conjugate
symmetry.

Ion stores the state and complex parameters as `complex64`, with float32 real
and imaginary components. As of 2026, JAX does not expose a `complex32` dtype,
so casting the surrounding model to float16 or bfloat16 leaves this complex path
at `complex64`. These layers therefore see less benefit from reduced-precision
computation than ordinary real-valued layers.

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
