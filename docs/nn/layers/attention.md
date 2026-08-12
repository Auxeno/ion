# Attention

Multi-head attention. One layer covers both modes: called with a single input it attends within that sequence, and called with a second `x_kv` input it attends from the query sequence into a separate context.

The stored projections are two-dimensional, with attention heads folded into
their output axes. In the shape annotations, `q` is the folded query projection
width (`num_heads * head_dim`) and `k` is the folded key/value projection width
(`num_kv_heads * head_dim`). The projections are reshaped into heads only during
the forward pass.

::: ion.nn.MultiHeadAttention
