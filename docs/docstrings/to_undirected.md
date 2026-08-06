Add the reverse of every edge so the graph is stored symmetrically.

Message passing flows from senders to receivers, so an undirected graph needs
both `(i, j)` and `(j, i)` present for information to travel in both
directions. This function guarantees that, and is the precondition for
anything that assumes a symmetric edge list.

The reversed edges are appended and the result is coalesced, so calling this
on a graph that is already symmetric returns it unchanged. Applying it
defensively is safe, unlike `add_self_loops`, which appends unconditionally.
Direction information is destroyed: a graph built from `(1, 0)` and one built
from `(0, 1)` produce the same output, and nothing can tell them apart
afterwards.

Edge features are handled through the returned indices. They address a
conceptual array of `2 * e` rows holding the original edges followed by the
reversed ones, so building that array and indexing it gives features aligned
to the result. Indices below `e` come from an edge in its original
orientation and indices at or above `e` come from a flipped one, which is what
lets direction-dependent features be negated rather than copied.

The output size depends on how much symmetry the input already had, so this
function cannot be called inside `jax.jit`. Use it when preparing edge arrays,
not inside a training step.

Parameters
----------
senders : jax.Array["e", int]
    Source node index for each edge.
receivers : jax.Array["e", int]
    Destination node index for each edge.

Returns
-------
tuple[jax.Array["e2", int], jax.Array["e2", int], jax.Array["e2", int]]
    Symmetric sender and receiver arrays in coalesced order, followed by the
    index of the row kept for each edge, addressing the original edges
    concatenated with the reversed ones.

Example
-------
```python
senders = jnp.array([0, 1, 1])
receivers = jnp.array([1, 0, 2])
senders, receivers, kept = gnn.to_undirected(senders, receivers)
# senders: [0, 1, 1, 2]
# receivers: [1, 0, 2, 1]
# kept: [0, 1, 2, 5]
```

The `(0, 1)` and `(1, 0)` pair was already present and is left alone, while
`(1, 2)` gains its reverse. Features that describe the edge itself, such as a
bond type or a weight, are copied onto the reverse:

```python
x_edge = jnp.concatenate([x_edge, x_edge])[kept]
```

Features that depend on the direction of travel, such as a displacement
between endpoints, are negated on the reverse instead:

```python
x_edge = jnp.concatenate([x_edge, -x_edge])[kept]
```

Self-loops are their own reverse, so they appear once in the output rather
than twice.
