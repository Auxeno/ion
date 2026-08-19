"""Private renderers for Ion's core types, one set for Treescope and one for the terminal.

Functions:
    hue         Map a class name to its palette hue.
    fields      Split a module's fields into config, params, buffers and children.
    summary     Describe a module's parameter count and size.
    statistics  Describe every parameter's distribution in one device sync.

Both layouts share `hue`, `fields` and `summary`, so the two cannot drift apart. Called
lazily by each type's `__treescope_repr__` and `__repr__` hooks.
"""

import dataclasses
import inspect
import math
import os
import sys
import zlib
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
from treescope import rendering_parts as parts

from . import gnn, nn, tree
from .nn.buffer import Buffer
from .nn.module import Module
from .nn.param import Param

if TYPE_CHECKING:
    from .optimizer import Optimizer

BINS = 11
BLOCKS = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
COMMENT = "\x1b[38;2;204;204;204m"


def hue(name: str) -> float:
    """Map a class name to its palette hue, hashing names from outside the palette."""

    # Hues follow export order, so layers of the same family are colored alike
    exports = [n for m in (nn, gnn.layers) for n, v in vars(m).items() if isinstance(v, type)]
    hues = {name: (222 + 11 * index) % 360 for index, name in enumerate(exports)}
    return hues.get(name, zlib.crc32(name.encode()) % 3_600 / 10)


def fields(self: Module) -> tuple[list, list, list, list]:
    """Split a module's fields into config, parameters, buffers and child modules."""
    signature = inspect.signature(type(self).__init__).parameters
    defaults = {name: parameter.default for name, parameter in signature.items()}

    # Entries are (label, value, name, default), the flag marking an uninformative default
    config, params, buffers, children = [], [], [], []
    for field in dataclasses.fields(self):  # type: ignore[reportArgumentType]
        if not field.repr:
            continue
        name, value = field.name, getattr(self, field.name)
        if isinstance(value, (list, tuple)) and any(isinstance(x, Module) for x in value):
            # Sequences splice in as (0), (1), ... rather than nesting one level deeper
            children += [(f"({i}): ", item, f"{name}[{i}]", False) for i, item in enumerate(value)]
        elif isinstance(value, Module):
            children.append((f"{name}=", value, name, False))
        elif isinstance(value, (Param, Buffer)):
            group = params if isinstance(value, Param) else buffers
            group.append((f"{name}=", value, name, False))
        else:
            config.append((f"{name}=", value, name, repr(value) == repr(defaults.get(name))))

    return config, params, buffers, children


def summary(self: Module) -> str:
    """Describe a module's size as `1,088 params, 4.25 KB`, or nothing if it holds none."""
    leaves = [x for x in jax.tree.leaves(self, is_leaf=tree.is_param) if tree.is_param(x)]
    if not leaves:
        return ""

    frozen = sum(getattr(leaf._value, "size", 0) for leaf in leaves if not leaf.trainable)
    text = f"{self.num_params:,} params, {self.disk_usage}"
    return text + (f", {frozen:,} frozen" if frozen else "")


def param_treescope(self: Param, path: str | None, subtree_renderer: Any) -> Any:
    """Render a `Param` as `Param(float32(64, 10))`, marking it frozen if it is."""
    value = self._value
    described = f"{value.dtype.name}{value.shape}" if hasattr(value, "dtype") else repr(value)

    # Nesting hides the array statistics, leaving float32(64, 10)
    full = subtree_renderer(value, path=None).renderable
    array = parts.abbreviatable(full, parts.text(described))
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
    described += "" if self.trainable else ", frozen"
    return parts.abbreviatable_with_annotations(node, parts.text(described))


def buffer_treescope(self: Buffer, path: str | None, subtree_renderer: Any) -> Any:
    """Render a `Buffer` as `Buffer(float32(64,))`."""
    value = self.value
    described = f"{value.dtype.name}{value.shape}"

    # Nesting hides the array statistics, leaving float32(64,)
    full = subtree_renderer(value, path=None).renderable
    array = parts.abbreviatable(full, parts.text(described))
    node = parts.build_foldable_tree_node_from_children(
        prefix="Buffer(",
        children=[array],
        suffix=")",
        path=path,
        expand_state=parts.ExpandState.COLLAPSED,
    )

    # Buffers keep their wrapper when abbreviated, marking them out from parameters
    return parts.abbreviatable_with_annotations(node, parts.text(f"Buffer({described})"))


def module_treescope(self: Module, path: str | None, subtree_renderer: Any) -> Any:
    """Render a `Module`, grouping its fields and coloring it by class."""
    # Fields are collected before rendering so the last visible entry can drop its separator
    config, params, buffers, children = fields(self)

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
    total = summary(self)

    node = parts.build_foldable_tree_node_from_children(
        prefix=parts.siblings(parts.maybe_qualified_type_name(type(self)), "("),
        children=lines,
        suffix=")",
        path=path,
        background_color=f"oklch(0.8 0.12 {hue(type(self).__qualname__):.1f})",
        first_line_annotation=parts.comment_color(parts.text(f"  # {total}")) if total else None,
        expand_state=(
            parts.ExpandState.WEAKLY_EXPANDED
            if children or path == ""
            else parts.ExpandState.COLLAPSED
        ),
    )

    # Deeper in the tree the fields drop away, leaving Linear(<16,640 params>)
    short = parts.abbreviation_color(
        parts.text(f"{type(self).__name__}(<{self.num_params:,} params>)")
    )
    return parts.abbreviatable_with_annotations(node, short)


def optimizer_treescope(self: "Optimizer", path: str | None, subtree_renderer: Any) -> Any:
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
    selected = [parts.text(f"fields={list(self._fields)}")] if self._fields is not None else []
    children = [parts.text(f"step={self.step}"), parts.siblings("state=", state), *selected]

    return parts.build_foldable_tree_node_from_children(
        prefix=parts.siblings(parts.maybe_qualified_type_name(type(self)), "("),
        children=children,
        suffix=")",
        comma_separated=True,
        path=path,
        background_color="oklch(0.88 0.12 95)",
        expand_state=parts.ExpandState.WEAKLY_EXPANDED,
    )


def color(text: str, code: str) -> str:
    """Wrap text in an ANSI code, leaving it bare where escape sequences would not render."""
    # Notebooks render ANSI but report no terminal, so IPython is detected separately
    ipython = sys.modules.get("IPython")
    notebook = ipython is not None and ipython.get_ipython() is not None
    if os.environ.get("NO_COLOR") or not (sys.stdout.isatty() or notebook):
        return text

    return f"{code}{text}\x1b[0m"


def oklch(tone: float, lightness: float = 0.8) -> str:
    """Convert an `oklch(l 0.12 h)` palette color into 24-bit ANSI `r;g;b` channels."""
    a, b = 0.12 * math.cos(math.radians(tone)), 0.12 * math.sin(math.radians(tone))
    oklab = (
        (0.3963377774, 0.2158037573),
        (-0.1055613458, -0.0638541728),
        (-0.0894841775, -1.2914855),
    )
    srgb = (
        (4.0767416621, -3.3077115913, 0.2309699292),
        (-1.2684380046, 2.6097574011, -0.3413193965),
        (-0.0041960863, -0.7034186147, 1.7076147010),
    )

    lms = [(lightness + ca * a + cb * b) ** 3 for ca, cb in oklab]
    channels = []
    for row in srgb:
        linear = sum(weight * cone for weight, cone in zip(row, lms))
        gamma = 1.055 * max(linear, 0.0) ** (1 / 2.4) - 0.055
        encoded = gamma if linear > 0.0031308 else linear * 12.92
        channels.append(min(255, max(0, round(encoded * 255))))

    return ";".join(str(channel) for channel in channels)


def statistics(self: Module) -> dict[int, str]:
    """Describe every parameter's distribution as a histogram and moments, keyed by `id`."""
    leaves = [x for x in jax.tree.leaves(self, is_leaf=tree.is_param) if tree.is_param(x)]

    reductions = []
    for leaf in leaves:
        # Low precision accumulates poorly and cannot be formatted, so reductions use float32
        values = jnp.asarray(leaf._value).ravel()
        values = jnp.abs(values) if jnp.iscomplexobj(values) else values.astype(jnp.float32)

        # Moments stay exact, but the window comes from a subsample to bound the sort
        sample = values[:: max(1, values.size // 8192)]
        low, high = jnp.percentile(sample, jnp.array([1.0, 99.0]))

        # A constant parameter has no width, so its mass falls in the middle bucket
        constant = high <= low
        spread = jnp.where(constant, 1.0, high - low)
        origin = jnp.where(constant, low - 0.5, low)
        index = jnp.clip((sample - origin) / spread * BINS, 0, BINS - 0.001).astype(jnp.int32)

        reductions.append((jnp.mean(values), jnp.std(values), jnp.bincount(index, length=BINS)))

    # Every reduction is issued before the first read, so the whole tree costs one sync
    described = {}
    for leaf, (mean, std, counts) in zip(leaves, jax.device_get(reductions)):
        if not math.isfinite(mean):
            described[id(leaf)] = f"{' ' * BINS}  not finite"
        else:
            bars = "".join(BLOCKS[round(count / counts.max() * 7)] for count in counts)
            described[id(leaf)] = f"{bars}  \u03bc={mean:.2g} \u03c3={std:.2g}"

    return described


def module_repr(self: Module, stats: dict[int, str] | None = None) -> str:
    """Render a `Module` as fully expanded text, mirroring the Treescope layout."""
    config, params, buffers, children = fields(self)

    # Config fields share one line, arrays and children follow under their own headings
    lines = []
    if config:
        shown = []
        for label, value, *_ in config:
            if isinstance(value, (jax.Array, np.ndarray)):
                shown.append(f"{label}{value.dtype.name}{value.shape}")
            elif callable(value) and hasattr(value, "__name__"):
                shown.append(f"{label}{value.__name__}")
            else:
                shown.append(f"{label}{value!r}")
        lines.append(", ".join(shown) + ",")

    for header, group in (("Parameters", params), ("Buffers", buffers), ("Modules", children)):
        if not group:
            continue

        lines.append(color(f"# {header}:", COMMENT))
        entries = []
        for label, value, *_ in group:
            rendered = module_repr(value, stats) if isinstance(value, Module) else repr(value)
            entries.append((f"{label}{rendered},", value))

        # Descriptions share one column, so distributions line up down the group
        width = max(len(entry) for entry, _ in entries) if stats and group is params else 0
        for entry, value in entries:
            described = stats.get(id(value)) if stats else None
            if described:
                entry += " " * (width - len(entry) + 2) + described
            lines += entry.split("\n")

    # Both brackets share one highlight, marking where the module's fields begin and end
    chip = f"\x1b[48;2;{oklch(hue(type(self).__qualname__))}m\x1b[38;2;30;30;30m"
    head, close = color(f"{type(self).__name__}(", chip), color(")", chip)
    if not lines:
        return f"{head}{close}"

    # Totals annotate the first line, e.g. Linear(  # 1,088 params, 4.25 KB
    total = summary(self)
    body = "\n".join(f"  {line}" for line in lines)
    return head + (color(f"  # {total}", COMMENT) if total else "") + f"\n{body}\n{close}"


def optimizer_repr(self: "Optimizer") -> str:
    """Render an `Optimizer`, folding its state away behind a size summary."""
    # State mirrors the model once per moment, so it shows only its size
    leaves = jax.tree.leaves(self.state)
    nbytes = sum(getattr(leaf, "nbytes", 0) for leaf in leaves)
    state = color(f"<{len(leaves):,} leaves, {nbytes / 2**20:.2f} MB>", COMMENT)

    step = self.step.item() if hasattr(self.step, "item") else self.step
    selected = f", fields={list(self._fields)}" if self._fields is not None else ""
    head = color("Optimizer", f"\x1b[48;2;{oklch(95, 0.88)}m\x1b[38;2;30;30;30m")

    return f"{head}(step={step}, state={state}{selected})"
