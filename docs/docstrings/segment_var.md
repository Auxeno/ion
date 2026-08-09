Population variance within each segment.

Divides by the number of values in the segment, matching `jnp.var` at its
default `ddof=0`. Empty segments and segments holding a single value return
zero rather than `NaN`.

Parameters
----------
data : jax.Array["e ...", float]
    Values to reduce. The leading dimension assigns one value to each edge or
    item.
segment_ids : jax.Array["e", int]
    Segment index for each value.
num_segments : int
    Total number of segments.

Returns
-------
jax.Array["s ...", float]
    Variance of each segment.

Example
-------
```python
messages = jnp.array([[1.0, 2.0], [3.0, 4.0], [8.0, 10.0]])
receivers = jnp.array([0, 0, 2])
variances = gnn.segment_var(messages, receivers, num_segments=3)  # [[1., 1.], [0., 0.], [0., 0.]]
```
