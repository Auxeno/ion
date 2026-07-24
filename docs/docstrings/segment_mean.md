Mean reduction within each segment.

Empty segments return zeros rather than `NaN`.

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
    Mean value for each segment.

Example
-------
```python
messages = jnp.array([[1.0, 2.0], [3.0, 4.0], [8.0, 10.0]])
receivers = jnp.array([0, 0, 2])
means = gnn.segment_mean(messages, receivers, num_segments=3)  # [[2., 3.], [0., 0.], [8., 10.]]
```
