"""Renderers for Ion's core types, one set for terminal and one for Treescope.

Functions:
    palette     Map a class name to its lightness, chroma and hue.
    ansi        Encode an oklch color as 24-bit ANSI channels.
    chip        Return the codes that print a class name on its own background.
    color       Wrap text in an ANSI code where escape sequences will render.
    highlight   Color the literals inside a value's repr.
    scaled      Format a quantity against a unit ladder.
    fields      Split a module's fields into config, params, buffers and children.
    summary     Describe a module's parameter count and size.
    statistics  Describe every parameter's distribution in one device sync.

Every `__repr__` and `__treescope_repr__` hook delegates here, to the matching `*_repr` or
`*_treescope` function. Both layouts share `palette`, `fields` and `summary`, so the two
cannot drift apart. Only the terminal colors literals, following the docs palette rather
than Treescope's own value styling.
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
from jax.core import Tracer

from . import tree
from .nn.buffer import Buffer
from .nn.module import Module
from .nn.param import Param

if TYPE_CHECKING:
    from .cost import Cost
    from .optimizer import Optimizer

_COMMENT = "\x1b[38;2;166;178;197m"
_NUMBER = "\x1b[38;2;34;211;238m"
_STRING = "\x1b[38;2;31;206;156m"
_CONSTANT = "\x1b[38;2;142;81;255m"
_SYMBOL = "\x1b[38;2;71;138;245m"
_WARNING = "\x1b[38;2;239;83;80m"
_BYTES = (" B", " KB", " MB", " GB", " TB")
_FLOPS = ("", "K", "M", "G", "T")
_COUNTS = ("", "K", "M", "B", "T")
_SAMPLE = 16_384
_BINS = 11
_BLOCKS = "▁▂▃▄▅▆▇█"
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_LITERALS = re.compile(
    rf"{_ANSI.pattern}|'[^']*'|\b(?:None|True|False)\b|(?<![\w.])-?\d+\.?\d*(?:[eE][-+]?\d+)?"
)
_OKLAB = ((0.3963377774, 0.2158037573), (-0.1055613458, -0.0638541728), (-0.0894841775, -1.2914855))
_SRGB = (
    (4.0767416621, -3.3077115913, 0.2309699292),
    (-1.2684380046, 2.6097574011, -0.3413193965),
    (-0.0041960863, -0.7034186147, 1.7076147010),
)
_PALETTE = {
    "AvgPool": (0.8, 0.12, 146),
    "MaxPool": (0.8, 0.12, 151),
    "GlobalAttentionPool": (0.8, 0.12, 157),
    "MultiHeadAttentionPool": (0.8, 0.12, 162),
    "Sequential": (0.8, 0.12, 172),
    "MLP": (0.8, 0.12, 175),
    "Dropout": (0.8, 0.12, 178),
    "DropPath": (0.8, 0.12, 180),
    "Bidirectional": (0.8, 0.12, 183),
    "Residual": (0.8, 0.12, 186),
    "Ensemble": (0.8, 0.12, 188),
    "LayerNorm": (0.8, 0.12, 199),
    "RMSNorm": (0.8, 0.12, 202),
    "BatchNorm": (0.8, 0.12, 205),
    "GroupNorm": (0.8, 0.12, 209),
    "SpectralNorm": (0.8, 0.12, 212),
    "GraphNorm": (0.8, 0.12, 215),
    "Linear": (0.8, 0.12, 244),
    "Identity": (0.8, 0.12, 242),
    "S4D": (0.8, 0.12, 252),
    "S4DCell": (0.8, 0.12, 257),
    "S5": (0.8, 0.12, 263),
    "S5Cell": (0.8, 0.12, 268),
    "Embedding": (0.8, 0.12, 278),
    "LearnedPositionalEmbedding": (0.8, 0.12, 284),
    "RoPE": (0.8, 0.12, 289),
    "SinusoidalPositionalEmbedding": (0.8, 0.12, 294),
    "Conv": (0.8, 0.12, 305),
    "ConvTranspose": (0.8, 0.12, 307),
    "GCNConv": (0.8, 0.12, 310),
    "GraphConv": (0.8, 0.12, 312),
    "SAGEConv": (0.8, 0.12, 314),
    "GatedGCNConv": (0.8, 0.12, 316),
    "GINConv": (0.8, 0.12, 319),
    "GINEConv": (0.8, 0.12, 321),
    "EdgeUpdate": (0.8, 0.12, 332),
    "GraphNetwork": (0.8, 0.12, 340),
    "NodeUpdate": (0.8, 0.12, 348),
    "GRU": (0.8, 0.12, 358),
    "GRUCell": (0.8, 0.12, 1),
    "LSTM": (0.8, 0.12, 4),
    "LSTMCell": (0.8, 0.12, 8),
    "RNN": (0.8, 0.12, 11),
    "RNNCell": (0.8, 0.12, 14),
    "MultiHeadAttention": (0.8, 0.12, 36),
    "GATConv": (0.8, 0.12, 28),
    "GATv2Conv": (0.8, 0.12, 31),
    "TransformerConv": (0.8, 0.12, 34),
    "HGTConv": (0.8, 0.12, 36),
    "RGCNConv": (0.8, 0.12, 36),
    "Optimizer": (0.8, 0.12, 80),
}


def palette(name: str) -> tuple[float, float, float]:
    """Map a class name to its lightness, chroma and hue, hashing names from outside Ion."""
    return _PALETTE.get(name, (0.8, 0.12, zlib.crc32(name.encode()) % 360))


def ansi(lightness: float, chroma: float, tone: float) -> str:
    """Encode an oklch color as 24-bit ANSI `r;g;b` channels."""
    a, b = chroma * math.cos(math.radians(tone)), chroma * math.sin(math.radians(tone))
    lms = [(lightness + ca * a + cb * b) ** 3 for ca, cb in _OKLAB]

    # Gamma encoding lands each channel on the sRGB curve, clipping where the color leaves it
    channels = []
    for linear in (sum(weight * cone for weight, cone in zip(row, lms)) for row in _SRGB):
        gamma = 1.055 * max(linear, 0.0) ** (1 / 2.4) - 0.055
        encoded = gamma if linear > 0.0031308 else linear * 12.92
        channels.append(min(255, max(0, round(encoded * 255))))

    return ";".join(str(channel) for channel in channels)


def chip(name: str) -> str:
    """Return the ANSI codes that print a class name on its own background color."""
    return f"\x1b[48;2;{ansi(*palette(name))}m\x1b[38;2;30;30;30m"


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
            return color(token, _STRING)
        return color(token, _CONSTANT if token[0].isalpha() else _NUMBER)

    return _LITERALS.sub(paint, text)


def scaled(value: float, base: float = 1024, units: tuple[str, ...] = _BYTES) -> str:
    """Format a quantity against its unit ladder, as `4.25 KB` or `82.4G`."""
    exponent = 0
    while value >= base and exponent < len(units) - 1:
        value, exponent = value / base, exponent + 1

    return f"{value:.3g}{units[exponent]}"


def fields(self: Module) -> tuple[list, list, list, list]:
    """Split a module's fields into config, parameters, buffers and child modules."""
    signature = inspect.signature(type(self).__init__).parameters
    defaults = {name: parameter.default for name, parameter in signature.items()}

    # Entries are (label, value, name, hidden), the flag marking config left at its default
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


def statistics(self: Module) -> dict[int, str]:
    """Describe every parameter as `▁▂▃█  μ=0.01 σ=0.1`, in one device synchronization."""
    import ion

    leaves = [x for x in jax.tree.leaves(self, is_leaf=tree.is_param) if tree.is_param(x)]
    if not ion.statistics or any(isinstance(leaf._value, Tracer) for leaf in leaves):
        return {}

    samples = []
    for leaf in leaves:
        # Low precision accumulates poorly and cannot be formatted, so summaries use float32
        values = jnp.asarray(leaf._value).ravel()
        values = jnp.abs(values) if jnp.iscomplexobj(values) else values.astype(jnp.float32)

        # Summaries come from a bounded sample, so a large parameter costs no more than a small one
        stride = max(1, math.ceil(values.size / _SAMPLE))
        samples.append((values[::stride], values.size <= _SAMPLE))

    # Every sample is issued before the first read, so the whole tree costs one sync
    described = {}
    for leaf, (sample, exact) in zip(leaves, jax.device_get(samples)):
        sample = np.asarray(sample, dtype=np.float32)
        if sample.size == 0 or not np.all(np.isfinite(sample)):
            described[id(leaf)] = f"{' ' * _BINS}  " + color("not finite", _WARNING)
            continue

        # The window comes from the percentiles, so a few outliers cannot flatten the bars
        low, high = np.percentile(sample, [1.0, 99.0])

        # A constant parameter has no width, so its mass falls in the middle bucket
        origin, spread = (low, high - low) if high > low else (low - 0.5, 1.0)
        index = np.clip((sample - origin) / spread * _BINS, 0, _BINS - 0.001).astype(np.int32)
        counts = np.bincount(index, minlength=_BINS)
        bars = "".join(_BLOCKS[round(count / counts.max() * 7)] for count in counts)

        # A sampled parameter marks both moments approximate, so no figure reads as measured
        sign = "=" if exact else "≈"
        mean = color(f"{np.mean(sample):.2g}", _NUMBER)
        std = color(f"{np.std(sample):.2g}", _NUMBER)
        described[id(leaf)] = f"{bars}  μ{sign}{mean} σ{sign}{std}"

    return described


def param_treescope(self: Param, path: str | None, subtree_renderer: Any) -> Any:
    """Render a `Param` as `Param(float32(64, 10))`, marking it frozen if it is."""
    from treescope import rendering_parts as parts

    # Collapsed the array is described by shape, expanded it renders in full
    value = self._value
    described = f"{value.dtype.name}{value.shape}" if hasattr(value, "dtype") else repr(value)
    array = parts.fold_condition(
        collapsed=parts.text(described),
        expanded=subtree_renderer(value, path=None).renderable,
    )
    children = [array] if self.trainable else [array, parts.text("frozen")]
    return parts.build_foldable_tree_node_from_children(
        prefix="Param(",
        children=children,
        suffix=")",
        comma_separated=True,
        path=path,
        expand_state=parts.ExpandState.COLLAPSED,
    )


def buffer_treescope(self: Buffer, path: str | None, subtree_renderer: Any) -> Any:
    """Render a `Buffer` as `Buffer(float32(64,))`."""
    from treescope import rendering_parts as parts

    # Collapsed the array is described by shape, expanded it renders in full
    value = self.value
    array = parts.fold_condition(
        collapsed=parts.text(f"{value.dtype.name}{value.shape}"),
        expanded=subtree_renderer(value, path=None).renderable,
    )
    return parts.build_foldable_tree_node_from_children(
        prefix="Buffer(",
        children=[array],
        suffix=")",
        path=path,
        expand_state=parts.ExpandState.COLLAPSED,
    )


def module_treescope(self: Module, path: str | None, subtree_renderer: Any) -> Any:
    """Render a `Module`, grouping its fields and coloring it by class."""
    from treescope import rendering_parts as parts

    # Fields are rendered together so the last visible entry can drop its separator
    config, params, buffers, children = fields(self)
    ordered = config + params + buffers + children
    last = max((i for i, (*_, hidden) in enumerate(ordered) if not hidden), default=-1)

    entries = []
    for index, (label, value, name, _) in enumerate(ordered):
        # Plain arrays are described by shape rather than dumped in full
        rendered = (
            parts.text(f"{value.dtype.name}{value.shape}")
            if isinstance(value, (jax.Array, np.ndarray))
            else subtree_renderer(value, path=None if path is None else f"{path}.{name}")
        )
        separator = parts.fold_condition(
            expanded=parts.text(", " if index < len(config) - 1 else ","),
            collapsed=parts.text("" if index == last else ", "),
        )
        entries.append(parts.siblings_with_annotations(label, rendered, separator))

    # Config fields share one line, dropping the copy buttons of anything left at its default
    shown = [
        parts.fold_condition(expanded=entry.renderable) if hidden else entry.renderable
        for entry, (*_, hidden) in zip(entries, config)
    ]
    lines = [parts.siblings(*shown)] if config else []

    # Parameters, buffers and children follow under their own headings, one to a line
    grouped = iter(entries[len(config) :])
    for header, group in (("Parameters", params), ("Buffers", buffers), ("Modules", children)):
        if group:
            comment = parts.comment_color(parts.text(f"# {header}:"))
            lines.append(parts.fold_condition(expanded=comment))
            lines += [parts.build_full_line_with_annotations(next(grouped)) for _ in group]

    # Totals annotate the first line, e.g. Linear(  # 1,088 params, 4.25 KB
    total = summary(self)

    return parts.build_foldable_tree_node_from_children(
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


def optimizer_treescope(self: "Optimizer", path: str | None, subtree_renderer: Any) -> Any:
    """Render an `Optimizer`, folding its state away behind a size summary."""
    from treescope import rendering_parts as parts

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
        background_color="oklch({:.3f} {:.3f} {:.1f})".format(*palette("Optimizer")),
        expand_state=parts.ExpandState.WEAKLY_EXPANDED,
    )


def param_repr(self: Param) -> str:
    """Render a `Param` as `Param(float32(64, 10))`, marking it frozen if it is."""
    value = self._value
    frozen = "" if self.trainable else color(", frozen", _SYMBOL)
    if not hasattr(value, "dtype"):
        return f"Param({highlight(repr(value))}{frozen})"

    return f"Param({color(value.dtype.name, _SYMBOL)}{highlight(str(value.shape))}{frozen})"


def buffer_repr(self: Buffer) -> str:
    """Render a `Buffer` as `Buffer(float32(64,))`."""
    value = self.value

    return f"Buffer({color(value.dtype.name, _SYMBOL)}{highlight(str(value.shape))})"


def module_repr(self: Module, stats: dict[int, str] | None = None) -> str:
    """Render a `Module` as fully expanded text, mirroring the Treescope layout."""
    config, params, buffers, children = fields(self)

    # Config fields share one line, arrays described by their shape and callables by name
    lines = []
    if config:
        shown = []
        for label, value, *_ in config:
            if isinstance(value, (jax.Array, np.ndarray)):
                dtype = color(value.dtype.name, _SYMBOL)
                shown.append(f"{label}{dtype}{highlight(str(value.shape))}")
            elif callable(value) and hasattr(value, "__name__"):
                shown.append(f"{label}{color(value.__name__, _SYMBOL)}")
            else:
                shown.append(f"{label}{highlight(repr(value))}")
        lines.append(", ".join(shown) + ",")

    # Parameters, buffers and children follow under their own headings, one to a line
    for header, group in (("Parameters", params), ("Buffers", buffers), ("Modules", children)):
        if not group:
            continue

        lines.append(color(f"# {header}:", _COMMENT))
        entries = []
        for label, value, *_ in group:
            rendered = module_repr(value, stats) if isinstance(value, Module) else repr(value)
            entries.append((f"{label}{rendered},", value))

        # Descriptions share one column, measured on visible text since escapes take no width
        widths = [len(_ANSI.sub("", entry)) for entry, _ in entries]
        for (entry, value), visible in zip(entries, widths):
            described = stats.get(id(value)) if stats else None
            if described is not None:
                entry += " " * (max(widths) - visible + 2) + described
            lines += entry.split("\n")

    # Both brackets share one highlight, marking where the module's fields begin and end
    name = type(self).__name__
    head, close = color(f"{name}(", chip(name)), color(")", chip(name))
    if not lines:
        return f"{head}{close}"

    # Totals annotate the first line, e.g. Linear(  # 1,088 params, 4.25 KB
    total = summary(self)
    body = "\n".join(f"  {line}" for line in lines)
    return head + (color(f"  # {total}", _COMMENT) if total else "") + f"\n{body}\n{close}"


def optimizer_repr(self: "Optimizer") -> str:
    """Render an `Optimizer`, folding its state away behind a size summary."""
    # State mirrors the model once per moment, so it shows only its size
    leaves = jax.tree.leaves(self.state)
    nbytes = sum(getattr(leaf, "nbytes", 0) for leaf in leaves)
    state = color(f"<{len(leaves):,} leaves, {nbytes / 2**20:.2f} MB>", _COMMENT)

    step = self.step.item() if hasattr(self.step, "item") else self.step
    selected = f", fields={list(self._fields)}" if self._fields is not None else ""
    head = color("Optimizer", chip("Optimizer"))

    return f"{head}(step={step}, state={state}{selected})"


def cost_repr(self: "Cost") -> str:
    """Render a `Cost` as a per-layer table, following the tree the module repr prints."""
    eighths = "▏▎▍▌▋▊▉█"  # left eighths, continuing a bar
    bar_width = 10  # cells in a layer share bar, so one cell is ten percent of the step
    memory_width = 20
    guide = f"\x1b[38;2;{ansi(0.46, 0.02, 260)}m"

    def bar(start: float, share: float, width: int, code: str, fill: str = "█") -> str:
        """Draw one cumulative segment over a fine guide spanning the whole scale."""
        offset = min(math.ceil(start * width), width)
        cells = min(max((start + share) * width - offset, 0.0), width - offset)
        segment = fill * int(cells)
        if cells % 1:
            segment += eighths[min(7, int(cells % 1 * 8))] if fill == "█" else fill
        remaining = width - offset - len(segment)
        return color("·" * offset, guide) + color(segment, code) + color("·" * remaining, guide)

    def shape(value: Any) -> str:
        """Describe a pytree by the dtype and shape of every array it holds."""
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            return f"{color(value.dtype.name, _SYMBOL)}{value.shape}"
        if isinstance(value, tuple):
            items = ", ".join(shape(item) for item in value)
            return f"({items},)" if len(value) == 1 else f"({items})"
        if isinstance(value, list):
            return f"[{', '.join(shape(item) for item in value)}]"
        if isinstance(value, dict):
            return "{" + ", ".join(f"{key}: {shape(item)}" for key, item in value.items()) + "}"
        return "---" if value is None else repr(value)

    # The tree is drawn exactly as the module repr indents it, class names and all
    labels = {}
    for path, layer in self.layers.items():
        # A scan runs its body once in the jaxpr, so the count it was scaled by is spelled out
        loop = f" loop x{layer.loop}" if layer.loop > 1 else ""
        indent = "  " * layer.depth + (f"{layer.label} " if layer.label else "")
        label = indent + color(layer.name, chip(layer.name)) + color(loop, _COMMENT)
        labels[path] = (label, len(indent) + len(layer.name) + len(loop))

    # The name column is sized to its longest entry, so a shallow tree leaves no gutter
    width = max(len("layer"), *(visible for _, visible in labels.values())) + 2
    op_width = max(len("ops"), len(f"{self.ops:,}"))

    name = color(self.name, chip(self.name))
    summary = color(
        f" · {scaled(self.params, 1e3, _COUNTS)} params"
        f" · {scaled(self.flops, 1e3, _FLOPS)}FLOP"
        f" · {self.ops:,} ops → {self.fused:,} fused · {self.backend}",
        _COMMENT,
    )
    title = name + summary
    inputs = color("input ", _COMMENT) + shape(self.inputs)

    # Memory components tile one shared scale; aliased output is shown as subtraction at its end
    data_bytes = max(self.input_bytes - self.param_bytes, 0)
    gross_memory = self.input_bytes + self.intermediate_bytes + self.output_bytes
    memory_grey = f"\x1b[38;2;{ansi(0.92, 0.03, 260)}m"
    memory_label_width = len("intermediate") + 2

    def memory_row(label: str, start: int, size: int, value: str, fill: str = "█") -> str:
        whole = gross_memory or 1
        code = memory_grey if label == "total" else _COMMENT
        meter = bar(start / whole, size / whole, memory_width, code, fill)
        return (
            color(f"{label:<{memory_label_width}}", _COMMENT)
            + meter
            + color(f"  {value}", _COMMENT)
        )

    memory = [
        color("memory", _COMMENT),
        memory_row("total", 0, self.total_memory, scaled(self.total_memory)),
        memory_row("params", 0, self.param_bytes, scaled(self.param_bytes)),
        memory_row("input", self.param_bytes, data_bytes, scaled(data_bytes)),
        memory_row(
            "intermediate",
            self.input_bytes,
            self.intermediate_bytes,
            scaled(self.intermediate_bytes),
        ),
        memory_row(
            "output",
            self.input_bytes + self.intermediate_bytes,
            self.output_bytes,
            scaled(self.output_bytes),
        ),
    ]
    if self.reused_bytes:
        memory.append(
            memory_row(
                "reused",
                gross_memory - self.reused_bytes,
                self.reused_bytes,
                f"-{scaled(self.reused_bytes)}",
                "░",
            )
        )
    columns = (
        f"{'layer':<{width}}{'FLOPs':>7}  {'':<{bar_width}}"
        f"{'share':>7}  {'ops':>{op_width}}  output"
    )

    lines = [
        title,
        inputs,
        "",
        *memory,
        "",
        color(columns, _COMMENT),
    ]

    # Siblings tile their parent's bar, so each one starts where the previous ended
    filled = {}
    for path, layer in self.layers.items():
        parent = path.rsplit(".", 1)[0] if "." in path else ""
        whole = self.layers[parent].share if path else 1.0
        within = layer.share / whole if whole else 0.0
        start = filled.get(parent, 0.0) if path else 0.0
        if path:
            filled[parent] = start + within

        # Bars fade with depth, so a nested level never reads as louder than the one above it
        grey = f"\x1b[38;2;{ansi(max(0.92 - 0.13 * layer.depth, 0.40), 0.03, 260)}m"
        meter = bar(start, within, bar_width, grey)

        # Only the dtypes carry color, so a row of figures reads as one measurement
        label, visible = labels[path]
        lines.append(
            f"{label}{' ' * (width - visible)}{scaled(layer.flops, 1e3, _FLOPS):>7}"
            f"  {meter}{layer.share * 100:6.1f}%"
            f"  {layer.ops:>{op_width},}  {shape(layer.output)}"
        )

    return "\n".join(lines)
