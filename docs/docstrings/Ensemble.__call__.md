Evaluate every ensemble member on the same input.

Parameters
----------
x : Any
    Input shared by every member.
*args : Any
    Additional positional arguments forwarded to each member.
training : bool | None, default=None
    Training mode forwarded when members accept it.
key : jax.Array | None, default=None
    RNG key split across members, then forwarded when they accept it.
**kwargs : Any
    Additional keyword arguments forwarded to each member.

Returns
-------
Any
    Member outputs with a leading ensemble axis, or their configured reduction.
