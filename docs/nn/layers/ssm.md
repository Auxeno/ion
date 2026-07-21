# State Space

Deep state space model layers. `S4D`, `S5`, and `LRU` process a sequence with a linear complex-valued recurrence, returning outputs plus the final state; the matching `*Cell` classes apply a single timestep. The output dimension is always `in_dim`. See [Reference](../reference.md#ssm) for the complex-state conventions.

::: ion.nn.S4D

::: ion.nn.S5

::: ion.nn.LRU

::: ion.nn.S4DCell

::: ion.nn.S5Cell

::: ion.nn.LRUCell
