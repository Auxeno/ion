"""Array and pytree annotations for signatures."""

from typing import (
    Annotated as BFloat16,
    Annotated as Bool,
    Annotated as Complex,
    Annotated as Complex64,
    Annotated as Complex128,
    Annotated as Float,
    Annotated as Float8E4M3,
    Annotated as Float8E5M2,
    Annotated as Float16,
    Annotated as Float32,
    Annotated as Float64,
    Annotated as Int,
    Annotated as Int8,
    Annotated as Int16,
    Annotated as Int32,
    Annotated as Int64,
    Annotated as Shaped,
    Annotated as UInt,
    Annotated as UInt8,
    Annotated as UInt16,
    Annotated as UInt32,
    Annotated as UInt64,
    Any,
    TypeAlias,
)

from jax import Array

__all__ = [
    "Array",
    "BFloat16",
    "Bool",
    "Complex",
    "Complex64",
    "Complex128",
    "Float",
    "Float8E4M3",
    "Float8E5M2",
    "Float16",
    "Float32",
    "Float64",
    "Int",
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "PRNGKey",
    "PyTree",
    "Shaped",
    "UInt",
    "UInt8",
    "UInt16",
    "UInt32",
    "UInt64",
]

PRNGKey: TypeAlias = Array
PyTree: TypeAlias = Any
