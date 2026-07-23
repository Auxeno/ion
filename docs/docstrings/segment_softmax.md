Softmax normalized independently within each segment.

The output has the same shape as `data`. Values that share a segment ID sum to
one, which is useful for normalizing incoming attention scores by receiver
node.

Parameters
----------
data : jax.Array["e ...", float]
    Values to normalize. The leading dimension assigns one value to each edge
    or item.
segment_ids : jax.Array["e", int]
    Segment index for each value.
num_segments : int
    Total number of segments.

Returns
-------
jax.Array["e ...", float]
    Values normalized within each segment.

Example
-------
```python
scores = jnp.array([1.0, 2.0, 3.0, 1.0])
receivers = jnp.array([0, 0, 1, 1])
weights = gnn.segment_softmax(scores, receivers, num_segments=2)
# [0.26894143, 0.7310586, 0.880797, 0.11920292]
```
