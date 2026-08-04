Sum values within each segment.

Parameters
----------
data : jax.Array["e ..."]
    Values to reduce. The leading dimension assigns one value to each segment
    ID.
segment_ids : jax.Array["e", int]
    Segment index for each value.
num_segments : int | None, default=None
    Total number of segments. Inferred from `segment_ids` when omitted.
indices_are_sorted : bool, default=False
    Whether `segment_ids` are known to be sorted.
unique_indices : bool, default=False
    Whether `segment_ids` are known to be unique.
bucket_size : int | None, default=None
    Optional bucket size used to improve reduction stability.
mode : Any, default=None
    Out-of-bounds scatter behavior passed to JAX.

Returns
-------
jax.Array["s ..."]
    Sum for each segment, with the same dtype as `data`.

Example
-------
```python
messages = jnp.array([[1.0, 2.0], [3.0, 4.0], [8.0, 10.0]])
receivers = jnp.array([0, 0, 2])
sums = gnn.segment_sum(messages, receivers, num_segments=3)  # [[4., 6.], [0., 0.], [8., 10.]]
```
