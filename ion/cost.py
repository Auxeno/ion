"""Static compute and memory analysis of a model call.

Functions:
    cost   Describe one call's arithmetic, memory and outputs, layer by layer.

Classes:
    Cost        Whole-call totals and its per-layer breakdown.
    LayerCost   One layer's share of the call.
"""

import dataclasses
import re
from collections.abc import Iterator
from typing import Any

import jax
import numpy as np

from . import tree
from .nn.module import Module, _cost_context


@dataclasses.dataclass(frozen=True)
class LayerCost:
    """One layer's share of a call."""

    name: str
    label: str
    depth: int
    flops: int
    share: float
    ops: int
    output: Any
    loop: int


@dataclasses.dataclass(frozen=True)
class Cost:
    """Static compute and memory analysis of one call."""

    name: str
    inputs: Any
    backend: str
    flops: int
    params: int
    param_bytes: int
    ops: int
    fused: int
    input_bytes: int
    intermediate_bytes: int
    output_bytes: int
    reused_bytes: int
    layers: dict[str, LayerCost]

    @property
    def total_memory(self) -> int:
        """Total bytes in the compiler's memory plan for the call."""
        return self.input_bytes + self.intermediate_bytes + self.output_bytes - self.reused_bytes

    def __repr__(self) -> str:
        from . import _rendering

        return _rendering.cost_repr(self)


class _Context:
    """Names modules and records their abstract outputs during one trace."""

    __slots__ = ("scopes", "outputs")

    def __init__(self) -> None:
        self.scopes: dict[int, tuple[str, str]] = {}
        self.outputs: dict[str, Any] = {}

    def record(self, path: str, value: Any) -> None:
        """Store one module's output, keeping shapes and dtypes rather than tracers."""
        self.outputs[path] = _abstract(value)


def _abstract(value: Any) -> Any:
    """Replace array leaves with shape/dtype placeholders without allocating data."""

    def leaf(x: Any) -> Any:
        if isinstance(x, jax.ShapeDtypeStruct):
            return x
        if hasattr(x, "shape") and hasattr(x, "dtype"):
            return jax.ShapeDtypeStruct(x.shape, x.dtype)
        return x

    return jax.tree.map(leaf, value)


def _has_array(value: Any) -> bool:
    """Whether a pytree contains a traceable array or shape placeholder."""
    return any(
        isinstance(leaf, jax.ShapeDtypeStruct)
        or (hasattr(leaf, "shape") and hasattr(leaf, "dtype"))
        for leaf in jax.tree.leaves(value)
    )


def _walk(
    module: Module, path: str = "", label: str = "", depth: int = 0
) -> Iterator[tuple[Module, str, str, int]]:
    """Yield every module as (module, path, label, depth), in the order the repr prints them."""
    yield module, path, label, depth
    for field in dataclasses.fields(module):  # type: ignore[reportArgumentType]
        if not field.repr:
            continue

        value = getattr(module, field.name)
        if isinstance(value, Module):
            children = [(field.name, field.name, value)]
        else:
            # Sequences splice in as (0), (1), ... exactly as the repr indents them
            items = enumerate(value) if isinstance(value, (list, tuple)) else ()
            children = [
                (f"{field.name}[{i}]", f"({i})", x) for i, x in items if isinstance(x, Module)
            ]

        for name, shown, child in children:
            yield from _walk(child, f"{path}.{name}" if path else name, shown, depth + 1)


def _measure(jaxpr: Any, scale: int, totals: dict[str, dict[str, int]]) -> None:
    """Accumulate execution-weighted FLOPs and static graph operations per scope."""
    for eqn in jaxpr.eqns:
        name = eqn.primitive.name
        if name in ("cond", "while"):
            raise NotImplementedError(f"ion.cost does not support dynamic {name} control flow")

        inner = scale * (eqn.params.get("length", 1) if name == "scan" else 1)
        for value in eqn.params.values():
            for item in value if isinstance(value, (tuple, list)) else (value,):
                closed = item if type(item).__name__ == "Jaxpr" else None
                body = getattr(item, "jaxpr", None) or closed
                if body is not None:
                    _measure(body, inner, totals)

        path = str(eqn.source_info.name_stack).replace("/", ".")
        entry = totals.setdefault(path, {"flops": 0, "ops": 0, "loop": 1})
        entry["ops"] += 1
        entry["loop"] = max(entry["loop"], scale)

        shaped = [var for var in eqn.outvars if hasattr(var, "aval") and hasattr(var.aval, "shape")]
        if not shaped:
            continue

        out = shaped[0].aval
        if name == "dot_general":
            (contract, _), _ = eqn.params["dimension_numbers"]
            lhs = eqn.invars[0].aval
            flops = 2 * int(np.prod(out.shape)) * int(np.prod([lhs.shape[i] for i in contract]))
        elif name == "conv_general_dilated":
            rhs = eqn.invars[1].aval
            flops = 2 * int(np.prod(out.shape)) * int(np.prod(rhs.shape[:-1]))
        else:
            flops = sum(int(np.prod(var.aval.shape)) for var in shaped)
        entry["flops"] += scale * flops


def _fused(text: str) -> int:
    """Count executable operations remaining in the optimized entry computation."""
    if "ENTRY " not in text:
        raise RuntimeError("The current backend did not expose an optimized XLA entry computation")

    # Buffer bookkeeping is not arithmetic, so only instructions that compute something count
    bookkeeping = ("parameter", "constant", "bitcast", "tuple", "get-tuple-element")
    count = 0
    for line in text.split("ENTRY ", 1)[1].splitlines():
        instruction = re.search(r"=\s*\S+\s+([a-z-]+)\(", line)
        if instruction is not None and instruction.group(1) not in bookkeeping:
            count += 1

    return count


def cost(target: Any, *args: Any, **kwargs: Any) -> Cost:
    """Describe a call's arithmetic, memory and outputs, broken down by layer.

    >>> ion.cost(model, x)  # doctest: +SKIP
    """
    wrapped = isinstance(target, Module)
    forward = (lambda model, *rest, **options: model(*rest, **options)) if wrapped else target
    values = (target, *args) if wrapped else args

    root = next((value for value in (*values, *kwargs.values()) if isinstance(value, Module)), None)
    if root is None:
        raise TypeError("cost needs a Module, either as its target or among its arguments")

    abstract_values = tuple(_abstract(value) for value in values)
    abstract_kwargs = {name: _abstract(value) for name, value in kwargs.items()}

    # The heading shows the call's data, so modules and static options are left out of it
    shown = [x for x in (*args, *kwargs.values()) if not isinstance(x, Module) and _has_array(x)]
    inputs = _abstract(shown[0] if len(shown) == 1 else tuple(shown))
    context = _Context()

    def traced(*positional: Any, **options: Any) -> Any:
        # A module names its own scope after the path it sits at, so the two match afterwards
        context.scopes = {
            id(module): (path.rsplit(".", 1)[-1], path)
            for value in (*positional, *options.values())
            if isinstance(value, Module)
            for module, path, _, _ in _walk(value)
        }
        token = _cost_context.set(context)
        try:
            return forward(*positional, **options)
        finally:
            _cost_context.reset(token)

    static_argnums = tuple(i for i, value in enumerate(abstract_values) if not _has_array(value))
    static_argnames = tuple(
        name for name, value in abstract_kwargs.items() if not _has_array(value)
    )
    lowered = jax.jit(traced, static_argnums=static_argnums, static_argnames=static_argnames).trace(
        *abstract_values, **abstract_kwargs
    )
    compiled = lowered.lower().compile()

    totals: dict[str, dict[str, int]] = {}
    _measure(lowered.jaxpr.jaxpr, 1, totals)
    flops = sum(entry["flops"] for entry in totals.values())

    layout = list(_walk(root))
    scopes = {path for _, path, _, _ in layout} | set(totals)

    layers = {}
    for module, path, label, depth in layout:
        # A layer owns its own scope and every scope nested inside it
        within = [key for key in scopes if not path or key == path or key.startswith(path + ".")]
        entries = [totals.get(key, {}) for key in within]
        layer_flops = sum(entry.get("flops", 0) for entry in entries)
        layers[path] = LayerCost(
            name=type(module).__name__,
            label=label,
            depth=depth,
            flops=layer_flops,
            share=layer_flops / flops if flops else 1.0,
            ops=sum(entry.get("ops", 0) for entry in entries),
            output=context.outputs.get(path),
            loop=max((entry.get("loop", 1) for entry in entries), default=1),
        )

    memory = compiled.memory_analysis()
    if memory is None:
        raise RuntimeError("The current backend did not provide a compiler memory analysis")

    leaves = jax.tree.leaves(root, is_leaf=tree.is_param)

    return Cost(
        name=type(root).__name__,
        inputs=inputs,
        backend=f"{jax.default_backend().upper()}/XLA",
        flops=flops,
        params=root.num_params,
        param_bytes=sum(getattr(x._value, "nbytes", 0) for x in leaves if tree.is_param(x)),
        ops=sum(entry["ops"] for entry in totals.values()),
        fused=_fused(compiled.as_text()),  # pyright: ignore[reportArgumentType]
        input_bytes=memory.argument_size_in_bytes,
        intermediate_bytes=memory.temp_size_in_bytes,
        output_bytes=memory.output_size_in_bytes,
        reused_bytes=memory.alias_size_in_bytes,
        layers=layers,
    )
