Population standard deviation within each segment.

The square root of `segment_var`, so it divides by the number of values in the
segment. Empty segments and segments holding a single value return zero rather
than `NaN`, but the gradient is undefined there; add a small constant before
the square root if those segments must stay differentiable.

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
    Standard deviation of each segment.

Example
-------
```python
messages = jnp.array([[1.0, 2.0], [3.0, 4.0], [8.0, 10.0]])
receivers = jnp.array([0, 0, 2])
stds = gnn.segment_std(messages, receivers, num_segments=3)  # [[1., 1.], [0., 0.], [0., 0.]]
```
