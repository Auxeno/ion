"""Private renderers for Ion's core types, one set for Treescope and one for the terminal.

Functions:
    palette     Map a class name to its lightness, chroma and hue.
    fields      Split a module's fields into config, params, buffers and children.
    summary     Describe a module's parameter count and size.
    statistics  Describe every parameter's distribution in one device sync.
    scaled      Format a quantity against a unit ladder.
    cost_repr   Lay out a cost analysis as a per-layer table.

Both layouts share `palette`, `fields` and `summary`, so the two cannot drift apart. Called
lazily by each type's `__treescope_repr__` and `__repr__` hooks. Only the terminal colors
literals, following the docs palette rather than Treescope's own value styling.
"""

import dataclasses
import inspect
import math
import os
import re
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
    from .cost import Cost
    from .optimizer import Optimizer

BINS = 11
WIDTH = 10  # cells in a share bar, so one cell is ten percent of the step
BYTES = (" B", " KB", " MB", " GB", " TB")
FLOPS = ("", "K", "M", "G", "T")
BLOCKS = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
BARS = "\u258f\u258e\u258d\u258c\u258b\u258a\u2589\u2588"  # left eighths, continuing a bar
COMMENT = "\x1b[38;2;166;178;197m"
NUMBER = "\x1b[38;2;34;211;238m"
STRING = "\x1b[38;2;31;206;156m"
CONSTANT = "\x1b[38;2;142;81;255m"
SYMBOL = "\x1b[38;2;71;138;245m"
FROZEN = "\x1b[38;2;71;138;245m"
WARNING = "\x1b[38;2;239;83;80m"
ANSI = re.compile(r"\x1b\[[0-9;]*m")
LITERALS = re.compile(
    rf"{ANSI.pattern}|'[^']*'|\b(?:None|True|False)\b|(?<![\w.])-?\d+\.?\d*(?:[eE][-+]?\d+)?"
)
ARC = (150, 265)  # first hue and span of the layer arc, in degrees
LAYER = (0.80, 0.80)  # lightness and saturation for an ordinary family
ACCENT = (0.82, 0.75)  # lightness and saturation for the families worth spotting first
USER = (0.78, 0.95)  # band for classes defined outside Ion
SPREAD = (16.0, 0.09, 0.20)  # hue, lightness and saturation a family fans out over

# Families take slots along the arc in this order, and the three most reached for are accented
ORDER = [
    "pool",
    "blocks",
    "norm",
    "linear",
    "ssm",
    "embedding",
    "conv",
    "composite",
    "recurrent",
    "attention",
]
ACCENTED = {"linear", "conv", "attention"}

# Mechanisms fold into their nearest neighbour, so no family is left with one or two members
MERGE = {
    "isomorphism": "conv",
    "gated": "conv",
    "relational": "attention",
    "positional": "embedding",
    "stochastic": "blocks",
}
LEAD = ("Sequential",)  # classes that take the leading position within their family

FAMILIES, MEMBERS = {}, {}
for _module in (nn, gnn.layers):
    for _name, _class in vars(_module).items():
        if isinstance(_class, type) and _name not in ("Module", "Param", "Buffer"):
            _path = _class.__module__
            _mechanism = "blocks" if ".blocks." in _path else _path.split(".")[-1]
            FAMILIES[_name] = MERGE.get(_mechanism, _mechanism)

# Members fan out symmetrically about their family, so a family reads as one band of color
for _family in ORDER:
    _names = sorted((n for n, k in FAMILIES.items() if k == _family), key=lambda n: n not in LEAD)
    for _index, _name in enumerate(_names):
        MEMBERS[_name] = _index / (len(_names) - 1) - 0.5 if len(_names) > 1 else 0.0


def palette(name: str) -> tuple[float, float, float]:
    """Map a class name to its lightness, chroma and hue, hashing names from outside Ion."""
    family = FAMILIES.get(name)
    if family is None:
        slot, (lightness, saturation) = zlib.crc32(name.encode()) % len(ORDER), USER
    else:
        slot = ORDER.index(family)
        lightness, saturation = ACCENT if family in ACCENTED else LAYER

    offset = MEMBERS.get(name, 0.0)
    tone = ARC[0] + ARC[1] * slot / len(ORDER) + SPREAD[0] * offset
    lightness = lightness + SPREAD[1] * offset
    saturation = min(max(saturation - SPREAD[2] * offset, 0.0), 1.0)

    # Saturation is a fraction of the gamut, so no hue is dragged down by the narrowest one
    return lightness, saturation * limit(lightness, tone), tone


def limit(lightness: float, tone: float) -> float:
    """Return the most chroma that stays inside sRGB at this lightness and hue."""
    low, high = 0.0, 0.4
    for _ in range(24):
        middle = (low + high) / 2
        inside = all(0.0 <= channel <= 1.0 for channel in oklch(lightness, middle, tone))
        low, high = (middle, high) if inside else (low, middle)

    return low


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
        background_color="oklch({:.3f} {:.3f} {:.1f})".format(*palette(type(self).__qualname__)),
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


def highlight(text: str) -> str:
    """Color the numbers, strings and constants inside a value's repr."""

    def paint(match: re.Match) -> str:
        # A nested repr arrives already colored, so its escape sequences pass through untouched
        token = match.group()
        if token.startswith("\x1b"):
            return token
        if token.startswith("'"):
            return color(token, STRING)
        return color(token, CONSTANT if token[0].isalpha() else NUMBER)

    return LITERALS.sub(paint, text)


def oklch(lightness: float, chroma: float, tone: float) -> list[float]:
    """Convert an oklch color into linear sRGB, leaving the unit range where it exits the gamut."""
    a, b = chroma * math.cos(math.radians(tone)), chroma * math.sin(math.radians(tone))
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
    return [sum(weight * cone for weight, cone in zip(row, lms)) for row in srgb]


def ansi(lightness: float, chroma: float, tone: float) -> str:
    """Encode an oklch color as 24-bit ANSI `r;g;b` channels."""
    channels = []
    for linear in oklch(lightness, chroma, tone):
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
            described[id(leaf)] = f"{' ' * BINS}  " + color("not finite", WARNING)
        else:
            bars = "".join(BLOCKS[round(count / counts.max() * 7)] for count in counts)
            average, deviation = color(f"{mean:.2g}", NUMBER), color(f"{std:.2g}", NUMBER)
            described[id(leaf)] = f"{bars}  \u03bc={average} \u03c3={deviation}"

    return described


def param_repr(self: Param) -> str:
    """Render a `Param` as `Param(float32(64, 10))`, marking it frozen if it is."""
    value = self._value
    frozen = "" if self.trainable else color(", frozen", FROZEN)
    if not hasattr(value, "dtype"):
        return f"Param({highlight(repr(value))}{frozen})"

    return f"Param({color(value.dtype.name, SYMBOL)}{highlight(str(value.shape))}{frozen})"


def buffer_repr(self: Buffer) -> str:
    """Render a `Buffer` as `Buffer(float32(64,))`."""
    value = self.value

    return f"Buffer({color(value.dtype.name, SYMBOL)}{highlight(str(value.shape))})"


def module_repr(self: Module, stats: dict[int, str] | None = None) -> str:
    """Render a `Module` as fully expanded text, mirroring the Treescope layout."""
    config, params, buffers, children = fields(self)

    # Config fields share one line, arrays and children follow under their own headings
    lines = []
    if config:
        shown = []
        for label, value, *_ in config:
            if isinstance(value, (jax.Array, np.ndarray)):
                dtype = color(value.dtype.name, SYMBOL)
                shown.append(f"{label}{dtype}{highlight(str(value.shape))}")
            elif callable(value) and hasattr(value, "__name__"):
                shown.append(f"{label}{color(value.__name__, SYMBOL)}")
            else:
                shown.append(f"{label}{highlight(repr(value))}")
        lines.append(", ".join(shown) + ",")

    for header, group in (("Parameters", params), ("Buffers", buffers), ("Modules", children)):
        if not group:
            continue

        lines.append(color(f"# {header}:", COMMENT))
        entries = []
        for label, value, *_ in group:
            rendered = module_repr(value, stats) if isinstance(value, Module) else repr(value)
            entries.append((f"{label}{rendered},", value))

        # Descriptions share one column, measured on visible text since escapes take no width
        widths = [len(ANSI.sub("", entry)) for entry, _ in entries]
        width = max(widths) if stats and group is params else 0
        for (entry, value), visible in zip(entries, widths):
            described = stats.get(id(value)) if stats else None
            if described:
                entry += " " * (width - visible + 2) + described
            lines += entry.split("\n")

    # Both brackets share one highlight, marking where the module's fields begin and end
    chip = f"\x1b[48;2;{ansi(*palette(type(self).__qualname__))}m\x1b[38;2;30;30;30m"
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
    head = color("Optimizer", f"\x1b[48;2;{ansi(0.88, 0.12, 95)}m\x1b[38;2;30;30;30m")

    return f"{head}(step={step}, state={state}{selected})"


def scaled(value: float, base: float, units: tuple[str, ...]) -> str:
    """Format a quantity against its unit ladder, as `4.25 KB` or `82.4G`."""
    exponent = 0
    while value >= base and exponent < len(units) - 1:
        value, exponent = value / base, exponent + 1

    return f"{value:.3g}{units[exponent]}"


def cost_repr(self: "Cost") -> str:
    """Render a `Cost` as a per-layer table, following the tree the module repr prints."""

    def shape(value: Any) -> str:
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            dtype = getattr(value.dtype, "name", str(value.dtype))
            return f"{color(dtype, SYMBOL)}{highlight(str(value.shape))}"
        if isinstance(value, tuple):
            items = ", ".join(shape(item) for item in value)
            return f"({items}{',' if len(value) == 1 else ''})"
        if isinstance(value, list):
            return f"[{', '.join(shape(item) for item in value)}]"
        if isinstance(value, dict):
            return "{" + ", ".join(f"{key}: {shape(item)}" for key, item in value.items()) + "}"
        return "---" if value is None else repr(value)

    inputs = " + ".join(scaled(value, 1024, BYTES) for value in self.input_components)
    inputs = f"({inputs}) input" if len(self.input_components) > 1 else f"{inputs} input"
    memory = (
        f"{scaled(self.total_memory, 1024, BYTES)} total memory = {inputs}"
        f" + {scaled(self.intermediate_bytes, 1024, BYTES)} intermediate"
        f" + {scaled(self.output_bytes, 1024, BYTES)} output"
    )
    if self.reused_bytes:
        memory += f" - {scaled(self.reused_bytes, 1024, BYTES)} reused"

    title = f"{self.name} \u00b7 input {shape(self.inputs)} \u00b7 {self.backend}"
    totals = (
        f"{scaled(self.flops, 1e3, FLOPS)}FLOP \u00b7 {self.params:,} params"
        f" ({scaled(self.param_bytes, 1024, BYTES)})"
        f" \u00b7 {self.ops:,} ops \u2192 {self.fused:,} fused"
    )
    # A scan runs its body once in the jaxpr, so the count it was scaled by is spelled out
    rows = [
        (
            "  " * layer.depth + (f"{layer.label} " if layer.label else ""),
            layer,
            f" loop x{layer.loop}" if layer.loop > 1 else "",
        )
        for layer in self.layers.values()
    ]

    # The name column is sized to its longest entry, so a shallow tree leaves no gutter
    width = max(len("layer"), *(len(a) + len(x.name) + len(z) for a, x, z in rows)) + 2
    op_width = max(len("ops"), len(f"{self.ops:,}"))
    columns = (
        f"{'layer':<{width}}{'FLOPs':>7}  {'':<{WIDTH}}{'share':>7}  {'ops':>{op_width}}  output"
    )

    # Siblings tile their parent's bar, so each one starts where the previous ended
    bars, filled = {}, {}
    for path, layer in self.layers.items():
        parent = path.rsplit(".", 1)[0] if "." in path else ""
        whole = self.layers[parent].share if path else 1.0
        within = layer.share / whole if whole else 0.0
        bars[path] = (filled.get(parent, 0.0) if path else 0.0, within)
        if path:
            filled[parent] = bars[path][0] + within
    lines = [
        color(title, COMMENT),
        "",
        color(totals, COMMENT),
        color(memory, COMMENT),
        "",
        color(columns, COMMENT),
    ]

    for (name, layer, suffix), (start, within) in zip(rows, bars.values()):
        # The tree is drawn exactly as the module repr indents it, class names and all
        chip = f"\x1b[48;2;{ansi(*palette(layer.name))}m\x1b[38;2;30;30;30m"
        pad = " " * (width - len(name) - len(layer.name) - len(suffix))
        title = f"{name}{color(layer.name, chip)}{color(suffix, COMMENT)}{pad}"

        # Bars fade with depth, so a nested level never reads as louder than the one above it
        # A partial block occupies its whole terminal cell, so the next sibling starts after it
        offset = math.ceil(start * WIDTH) if start else 0
        cells = min(max((start + within) * WIDTH - offset, 0.0), WIDTH - offset)
        bar = " " * offset + "\u2588" * int(cells)
        if cells % 1:
            bar += BARS[min(7, int(cells % 1 * 8))]
        grey = f"\x1b[38;2;{ansi(max(0.92 - 0.13 * layer.depth, 0.40), 0.0, 0.0)}m"

        flops = scaled(layer.flops, 1e3, FLOPS)
        lines.append(
            f"{title}{flops:>7}"
            f"  {color(bar.ljust(WIDTH), grey)}{layer.share * 100:6.1f}%"
            f"  {color(f'{layer.ops:>{op_width},}', NUMBER)}  {shape(layer.output)}"
        )

    return "\n".join(lines)
