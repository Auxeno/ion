Module-shaped pytree containing model parameters.

Returns
-------
PyTree
    `Param` leaves are preserved, while bare array and `Buffer` fields are
    replaced by `None`. Static configuration remains unchanged.
