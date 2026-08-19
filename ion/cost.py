"""Per-layer compute analysis of a model or training step.

Functions:
    cost   Measure one call's arithmetic, traffic and memory, layer by layer.

Classes:
    Cost        Whole-model totals and the per-layer breakdown of a single call.
    LayerCost   One layer's measured share of that call.

Both passes key on the scopes `Module.__call__` installs while tracing, so arithmetic comes
from the jaxpr and traffic from post-fusion HLO without either guessing which operation
belongs to which layer. Scan bodies are scaled by their length, which `cost_analysis` omits.
"""

import dataclasses
import re
from typing import Any

import jax
import numpy as np

from .nn.module import Module, _scopes

BALANCE = 250.0  # peak flops per peak byte, the ridge dividing compute from bandwidth limits
SKIP = ("parameter", "constant", "bitcast", "tuple", "get-tuple-element")
SHAPE = re.compile(r"\b(pred|f8[a-z0-9]+|[a-z]+\d+)\[([0-9,]*)\]")
OP_NAME = re.compile(r'op_name="([^"]+)"')
INSTRUCTION = re.compile(r"=\s*\S+\s+([a-z-]+)\(")


@dataclasses.dataclass(frozen=True)
class LayerCost:
    """One layer's measured share of a call."""

    name: str
    label: str
    depth: int
    flops: int
    transferred: int
    memory: int
    share: float
    ceiling: float
    dtype: str
    loop: int

    @property
    def intensity(self) -> float:
        """Arithmetic performed per byte moved."""
        return self.flops / self.transferred if self.transferred else 0.0


@dataclasses.dataclass(frozen=True)
class Cost:
    """Whole-model totals and the per-layer breakdown of one call."""

    flops: int
    transferred: int
    memory: int
    peak_memory: int
    params: int
    ops: int
    kernels: int
    balance: float
    layers: dict[str, LayerCost]

    def __repr__(self) -> str:
        from . import _rendering

        return _rendering.cost_repr(self)


def _structure(
    module: Module, path: str = "", depth: int = 0, label: str = "", out: Any = None
) -> list:
    """List every module as (path, label, class, depth) in the order the repr prints them."""
    from ._rendering import fields

    out = [] if out is None else out
    out.append((path, label, type(module).__name__, depth))
    for shown, child, name, _ in fields(module)[3]:
        # The label is the one the repr prints, so `(0)` rather than the path segment
        _structure(child, f"{path}.{name}" if path else name, depth + 1, shown.rstrip("=: "), out)

    return out


def _labels(module: Module, out: Any = None, name: str | None = None) -> dict[int, str]:
    """Map each child module's id to its field label, so nested scopes rebuild the full path."""
    from ._rendering import fields

    out = {} if out is None else out
    if name is not None:
        out[id(module)] = name
    for _, child, field, _ in fields(module)[3]:
        _labels(child, out, field)

    return out


def _measure(jaxpr: Any, scale: int, totals: dict) -> int:
    """Accumulate FLOPs and op counts per scope, and return the peak bytes held live."""
    nbytes = lambda var: int(np.prod(var.aval.shape)) * var.aval.dtype.itemsize

    # A value dies once the last equation reading it has run, freeing its buffer for reuse
    last = {}
    for index, eqn in enumerate(jaxpr.eqns):
        for var in eqn.invars:
            if type(var).__name__ != "Literal":
                last[var] = index

    live, peak = {}, 0
    for index, eqn in enumerate(jaxpr.eqns):
        name = eqn.primitive.name
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

        # A scan body appears once but runs `length` times, so its arithmetic scales with it
        inner = scale * (eqn.params.get("length", 1) if name == "scan" else 1)
        nested = 0
        for value in eqn.params.values():
            for item in value if isinstance(value, (tuple, list)) else (value,):
                closed = item if type(item).__name__ == "Jaxpr" else None
                body = getattr(item, "jaxpr", None) or closed
                if body is not None:
                    nested = max(nested, _measure(body, inner, totals))

        path = str(eqn.source_info.name_stack).replace("/", ".")
        entry = totals.setdefault(
            path, {"flops": 0, "memory": 0, "ops": 0, "dtype": ("", 0), "loop": 1}
        )
        entry["loop"] = max(entry["loop"], scale)
        entry["flops"] += scale * flops
        entry["ops"] += 1

        # Memory is a high-water mark, so buffers already dead do not count towards it
        live.update({var: nbytes(var) for var in shaped})

        # A loop or branch holds its own frame; other sub-jaxprs are the equation itself inlined
        frame = name in ("scan", "while", "cond")
        held = max(sum(live.values()) + (nested if frame else 0), nested)
        entry["memory"] = max(entry["memory"], held)
        peak = max(peak, held)
        for var in [var for var, when in ((v, last.get(v, -1)) for v in live) if when <= index]:
            del live[var]

        # The dtype shown is whichever one the layer's heaviest operation ran in
        if scale * flops > entry["dtype"][1]:
            entry["dtype"] = (out.dtype.name, scale * flops)

    return peak


def _traffic(text: str, paths: set[str]) -> tuple[dict[str, int], int]:
    """Sum post-fusion bytes per layer, skipping instructions that reinterpret rather than move."""
    moved, kernels, sizes = {}, 0, {}
    for line in text.split("ENTRY ", 1)[1].splitlines():
        named = re.match(r"\s*(?:ROOT\s+)?%(\S+) = ", line)
        if named is None:
            continue

        # Operands appear as bare references, so every instruction's own shape is recorded first
        shape = SHAPE.search(line[named.end() :])
        if shape is None:
            continue
        token, dims = shape.groups()
        narrow = token.startswith("f8") or token == "pred"
        width = 1 if narrow else int(re.sub(r"\D", "", token)) // 8
        sizes[named.group(1)] = int(np.prod([int(d) for d in dims.split(",") if d])) * width

        instruction = INSTRUCTION.search(line)
        if instruction is None or instruction.group(1) in SKIP:
            continue

        # An instruction reads its operands and writes its output, so both count as traffic
        operands = re.findall(r"%([\w.\-]+)", line[instruction.end() :].split(")", 1)[0])
        nbytes = sizes[named.group(1)] + sum(sizes.get(operand, 0) for operand in operands)

        op_name = OP_NAME.search(line)
        path, best = "", ""
        for segment in op_name.group(1).split("/")[1:] if op_name else []:
            path = f"{path}.{segment}" if path else segment
            best = path if path in paths else best

        moved[best] = moved.get(best, 0) + nbytes
        kernels += 1

    return moved, kernels


def cost(target: Any, *args: Any, balance: float = BALANCE, **kwargs: Any) -> Cost:
    """Measure a call's arithmetic, memory traffic and footprint, broken down by layer.

    >>> ion.cost(model, x)  # doctest: +SKIP
    """
    wrapped = isinstance(target, Module)
    forward = (lambda model, *rest, **options: model(*rest, **options)) if wrapped else target
    values = (target, *args) if wrapped else args

    root = next((value for value in values if isinstance(value, Module)), None)
    if root is None:
        raise TypeError("cost needs a Module, either as its target or among its arguments")

    def traced(*positional, **options):
        for value in positional:
            if isinstance(value, Module):
                _scopes.update(_labels(value))
        try:
            return forward(*positional, **options)
        finally:
            _scopes.clear()

    # Non-array keywords cannot be traced, so they compile in as static configuration
    static = tuple(name for name, value in kwargs.items() if not hasattr(value, "shape"))
    lowered = jax.jit(traced, static_argnames=static).trace(*values, **kwargs)
    compiled = lowered.lower().compile()

    structure = _structure(root)
    paths = {path for path, *_ in structure}
    totals: dict[str, dict] = {}
    _measure(lowered.jaxpr.jaxpr, 1, totals)
    moved, kernels = _traffic(compiled.as_text(), paths)  # pyright: ignore[reportArgumentType]

    flops = sum(entry["flops"] for entry in totals.values())
    transferred = sum(moved.values())
    within = lambda path: (key for key in paths | set(totals) | set(moved)
                           if not path or key == path or key.startswith(path + "."))

    rolled = {
        path: (
            sum(totals.get(key, {}).get("flops", 0) for key in within(path)),
            sum(moved.get(key, 0) for key in within(path)),
        )
        for path, *_ in structure
    }

    # Kernels run one after another, so each is weighed alone and the tree sums those weights
    weighed = {
        scope: max(totals.get(scope, {}).get("flops", 0), moved.get(scope, 0) * balance)
        for scope in set(totals) | set(moved)
    }
    weights = {path: sum(weighed.get(key, 0.0) for key in within(path)) for path, *_ in structure}

    layers = {}
    for path, label, name, depth in structure:
        entries = [totals.get(key, {}) for key in within(path)]
        layer_flops, layer_bytes = rolled[path]
        heaviest = max((entry.get("dtype", ("", 0)) for entry in entries), key=lambda pair: pair[1])
        layers[path] = LayerCost(
            name=name,
            label=label,
            depth=depth,
            flops=layer_flops,
            transferred=layer_bytes,
            memory=max((entry.get("memory", 0) for entry in entries), default=0),
            share=min(1.0, weights[path] / weights[""]) if weights[""] else 1.0,
            ceiling=min(1.0, (layer_flops / layer_bytes if layer_bytes else 0.0) / balance),
            dtype=heaviest[0],
            loop=max((entry.get("loop", 1) for entry in entries), default=1),
        )

    return Cost(
        flops=flops,
        transferred=transferred,
        memory=max((entry["memory"] for entry in totals.values()), default=0),
        peak_memory=compiled.memory_analysis().temp_size_in_bytes,  # pyright: ignore[reportOptionalMemberAccess]
        params=root.num_params,
        ops=sum(entry["ops"] for entry in totals.values()),
        kernels=kernels,
        balance=balance,
        layers=layers,
    )
