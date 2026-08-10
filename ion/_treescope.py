"""Private Treescope renderers for Ion's core types.

Called lazily by each type's `__treescope_repr__` hook.
"""

import dataclasses
import inspect
import zlib
from typing import TYPE_CHECKING, Any

import jax
import numpy as np
from treescope import rendering_parts as parts

from . import tree
from .nn.buffer import Buffer
from .nn.module import Module
from .nn.param import Param

if TYPE_CHECKING:
    from .optimizer import Optimizer


def param(self: Param, path: str | None, subtree_renderer: Any) -> Any:
    """Render a `Param` as `Param(float32(64, 10))`, marking it frozen if it is."""
    value = self._value
    summary = f"{value.dtype.name}{value.shape}" if hasattr(value, "dtype") else repr(value)

    # Nesting hides the array statistics, leaving float32(64, 10)
    full = subtree_renderer(value, path=None).renderable
    array = parts.abbreviatable(full, parts.text(summary))
    children = [array] if self.trainable else [array, parts.text("frozen")]
    node = parts.build_foldable_tree_node_from_children(
        prefix="Param(",
        children=children,
        suffix=")",
        comma_separated=True,
        path=path,
        expand_state=parts.ExpandState.COLLAPSED,
    )

    # Deeper still the wrapper drops away too
    summary += "" if self.trainable else ", frozen"
    return parts.abbreviatable_with_annotations(node, parts.text(summary))


def buffer(self: Buffer, path: str | None, subtree_renderer: Any) -> Any:
    """Render a `Buffer` as `Buffer(float32(64,))`."""
    value = self.value
    summary = f"{value.dtype.name}{value.shape}"

    # Nesting hides the array statistics, leaving float32(64,)
    full = subtree_renderer(value, path=None).renderable
    array = parts.abbreviatable(full, parts.text(summary))
    node = parts.build_foldable_tree_node_from_children(
        prefix="Buffer(",
        children=[array],
        suffix=")",
        path=path,
        expand_state=parts.ExpandState.COLLAPSED,
    )

    # Buffers keep their wrapper when abbreviated, marking them out from parameters
    return parts.abbreviatable_with_annotations(node, parts.text(f"Buffer({summary})"))


def module(self: Module, path: str | None, subtree_renderer: Any) -> Any:
    """Render a `Module`, grouping its fields and coloring it by class."""
    # Config values matching their constructor default carry no information
    signature = inspect.signature(type(self).__init__).parameters
    defaults = {name: p.default for name, p in signature.items()}

    # Fields are collected before rendering so the last visible entry can drop its separator
    config, params, buffers, children = [], [], [], []
    for field in dataclasses.fields(self):  # type: ignore[reportArgumentType]
        if not field.repr:
            continue
        name, value = field.name, getattr(self, field.name)
        if isinstance(value, (list, tuple)) and any(isinstance(x, Module) for x in value):
            # Sequences splice in as (0), (1), ... rather than nesting one level deeper
            children += [(f"({i}): ", x, f"{name}[{i}]", False) for i, x in enumerate(value)]
        elif isinstance(value, Module):
            children.append((f"{name}=", value, name, False))
        elif isinstance(value, (Param, Buffer)):
            group = params if isinstance(value, Param) else buffers
            group.append((f"{name}=", value, name, False))
        else:
            config.append((f"{name}=", value, name, repr(value) == repr(defaults.get(name))))

    ordered = config + params + buffers + children
    last = max((i for i, item in enumerate(ordered) if not item[3]), default=-1)
    entry = lambda i, label, value, name: parts.siblings_with_annotations(
        label,
        # Plain arrays are described by shape rather than dumped in full
        parts.text(f"{value.dtype.name}{value.shape}")
        if isinstance(value, (jax.Array, np.ndarray))
        else subtree_renderer(value, path=None if path is None else f"{path}.{name}"),
        parts.fold_condition(
            expanded=parts.text(", " if i < len(config) - 1 else ","),
            collapsed=parts.text("" if i == last else ", "),
        ),
    )
    rendered = [entry(i, *item[:3]) for i, item in enumerate(ordered)]
    grouped = iter(rendered[len(config) :])

    # Config fields share one line, dropping their copy buttons, arrays and children follow
    shown = [
        parts.fold_condition(expanded=line.renderable) if hidden else line.renderable
        for line, (*_, hidden) in zip(rendered, config)
    ]
    lines = [parts.siblings(*shown)] if config else []
    for header, group in (("Parameters", params), ("Buffers", buffers), ("Modules", children)):
        if group:
            comment = parts.comment_color(parts.text(f"# {header}:"))
            lines.append(parts.fold_condition(expanded=comment))
            lines += [parts.build_full_line_with_annotations(next(grouped)) for _ in group]

    # Totals annotate the first line, e.g. Linear(  # 1,088 params, 4.25 KB
    total = self.num_params
    leaves = [p for p in jax.tree.leaves(self, is_leaf=tree.is_param) if tree.is_param(p)]
    frozen = sum(getattr(p._value, "size", 0) for p in leaves if not p.trainable)
    summary = f"  # {total:,} params, {self.disk_usage}"
    summary += f", {frozen:,} frozen" if frozen else ""

    # Hue derived from a salted hash of the class name; the salt tunes the palette
    h = zlib.crc32(f"j4h9be:{type(self).__qualname__}".encode())

    node = parts.build_foldable_tree_node_from_children(
        prefix=parts.siblings(parts.maybe_qualified_type_name(type(self)), "("),
        children=lines,
        suffix=")",
        path=path,
        background_color=f"oklch(0.8 0.1 {h % 10_000 / 10_000 * 360:.1f})",
        first_line_annotation=parts.comment_color(parts.text(summary)) if leaves else None,
        expand_state=(
            parts.ExpandState.WEAKLY_EXPANDED
            if children or path == ""
            else parts.ExpandState.COLLAPSED
        ),
    )

    # Deeper in the tree the fields drop away, leaving Linear(<16,640 params>)
    short = parts.abbreviation_color(parts.text(f"{type(self).__name__}(<{total:,} params>)"))
    return parts.abbreviatable_with_annotations(node, short)


def optimizer(self: "Optimizer", path: str | None, subtree_renderer: Any) -> Any:
    """Render an `Optimizer`, folding its state away behind a size summary."""
    # State mirrors the model once per moment, so collapsed it shows only its size
    leaves = jax.tree.leaves(self.state)
    nbytes = sum(getattr(leaf, "nbytes", 0) for leaf in leaves)
    state = parts.fold_condition(
        collapsed=parts.abbreviation_color(
            parts.text(f"<{len(leaves):,} leaves, {nbytes / 2**20:.2f} MB>")
        ),
        expanded=subtree_renderer(self.state, path=None).renderable,
    )
    fields = [parts.text(f"fields={list(self._fields)}")] if self._fields is not None else []
    children = [parts.text(f"step={self.step}"), parts.siblings("state=", state), *fields]

    return parts.build_foldable_tree_node_from_children(
        prefix=parts.siblings(parts.maybe_qualified_type_name(type(self)), "("),
        children=children,
        suffix=")",
        comma_separated=True,
        path=path,
        background_color="oklch(0.88 0.10 95)",
        expand_state=parts.ExpandState.COLLAPSED,
    )
