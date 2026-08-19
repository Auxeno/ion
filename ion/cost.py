"""Static compute and memory analysis of a model call.

Functions:
    cost   Describe one call's arithmetic, memory and outputs, layer by layer.

Classes:
    Cost        Whole-call totals and its per-layer breakdown.
    LayerCost   One layer's share of the call.
"""

import dataclasses
import re
from typing import Any

import jax
import numpy as np

from . import tree
from .nn.module import Module, _cost_context

SKIP = ("parameter", "constant", "bitcast", "tuple", "get-tuple-element")
INSTRUCTION = re.compile(r"=\s*\S+\s+([a-z-]+)\(")


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
    input_components: tuple[int, ...]
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


@dataclasses.dataclass
class _Context:
    """Names modules and records their abstract outputs during one trace."""

    scopes: dict[int, tuple[str, str]]
    outputs: dict[str, Any] = dataclasses.field(default_factory=dict)

    def record(self, path: str, value: Any) -> None:
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


def _structure(
    module: Module, path: str = "", depth: int = 0, label: str = "", out: Any = None
) -> list:
    """List every module as (path, label, class, depth) in repr order."""
    from ._rendering import fields

    out = [] if out is None else out
    out.append((path, label, type(module).__name__, depth))
    for shown, child, name, _ in fields(module)[3]:
        _structure(child, f"{path}.{name}" if path else name, depth + 1, shown.rstrip("=: "), out)
    return out


def _scopes(
    module: Module,
    path: str = "",
    label: str = "",
    out: dict[int, tuple[str, str]] | None = None,
) -> dict[int, tuple[str, str]]:
    """Map module identities to their local scope label and full tree path."""
    from ._rendering import fields

    out = {} if out is None else out
    out[id(module)] = (label, path)
    for _, child, name, _ in fields(module)[3]:
        child_path = f"{path}.{name}" if path else name
        _scopes(child, child_path, name, out)
    return out


def _measure(jaxpr: Any, scale: int, totals: dict) -> None:
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

    count = 0
    for line in text.split("ENTRY ", 1)[1].splitlines():
        instruction = INSTRUCTION.search(line)
        if instruction is not None and instruction.group(1) not in SKIP:
            count += 1
    return count


def _param_bytes(module: Module) -> int:
    """Stored bytes belonging to model parameters, excluding buffers."""
    leaves = jax.tree.leaves(module, is_leaf=tree.is_param)
    return sum(getattr(leaf._value, "nbytes", 0) for leaf in leaves if tree.is_param(leaf))


def _shown_inputs(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """Return the abstract non-module array arguments shown in the report heading."""
    values = [value for value in (*args, *kwargs.values()) if not isinstance(value, Module)]
    values = [value for value in values if _has_array(value)]
    if not values:
        return ()
    return values[0] if len(values) == 1 else tuple(values)


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
    shown_inputs = _abstract(_shown_inputs(args, kwargs))
    context = _Context({})

    def traced(*positional: Any, **options: Any) -> Any:
        modules = [value for value in (*positional, *options.values()) if isinstance(value, Module)]
        context.scopes = {}
        for module in modules:
            _scopes(module, out=context.scopes)
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

    structure = _structure(root)
    paths = {path for path, *_ in structure}
    totals: dict[str, dict] = {}
    _measure(lowered.jaxpr.jaxpr, 1, totals)
    within = lambda path: (
        key for key in paths | set(totals) if not path or key == path or key.startswith(path + ".")
    )

    flops = sum(entry["flops"] for entry in totals.values())
    ops = sum(entry["ops"] for entry in totals.values())
    layers = {}
    for path, label, name, depth in structure:
        entries = [totals.get(key, {}) for key in within(path)]
        layer_flops = sum(entry.get("flops", 0) for entry in entries)
        layers[path] = LayerCost(
            name=name,
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

    param_bytes = _param_bytes(root)
    input_bytes = memory.argument_size_in_bytes
    if param_bytes <= input_bytes:
        other_inputs = input_bytes - param_bytes
        components = tuple(value for value in (other_inputs, param_bytes) if value)
    else:
        components = (input_bytes,)

    return Cost(
        name=type(root).__name__,
        inputs=shown_inputs,
        backend=f"{jax.default_backend().upper()}/XLA",
        flops=flops,
        params=root.num_params,
        param_bytes=param_bytes,
        ops=ops,
        fused=_fused(compiled.as_text()),  # pyright: ignore[reportArgumentType]
        input_bytes=input_bytes,
        input_components=components,
        intermediate_bytes=memory.temp_size_in_bytes,
        output_bytes=memory.output_size_in_bytes,
        reused_bytes=memory.alias_size_in_bytes,
        layers=layers,
    )
