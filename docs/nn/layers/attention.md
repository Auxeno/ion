# Attention

Multi-head attention. One layer covers both modes: called with a single input it attends within that sequence, and called with a second `x_kv` input it attends from the query sequence into a separate context.

::: ion.nn.MultiHeadAttention
