"""Render benchmark JSON results as Plotly HTML files."""

import argparse
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

MODELS = ("mlp", "resnet", "gpt")
SIZES = ("tiny", "small", "medium")
SERIES = (
    ("ion", "compiled"),
    ("equinox", "compiled"),
    ("nnx", "compiled"),
    ("pytorch", "compiled"),
    ("pytorch", "eager"),
)
LABELS = {
    ("ion", "compiled"): "Ion",
    ("equinox", "compiled"): "Equinox",
    ("nnx", "compiled"): "Flax NNX",
    ("pytorch", "compiled"): "PyTorch compiled",
    ("pytorch", "eager"): "PyTorch eager",
}
COLORS = {
    ("ion", "compiled"): "#7c3aed",
    ("equinox", "compiled"): "#22d3ee",
    ("nnx", "compiled"): "#1fce9c",
    ("pytorch", "compiled"): "#ef5350",
    ("pytorch", "eager"): "#f57e2c",
}

ValueKind = Literal["time", "throughput", "memory"]
GroupKey = tuple[str, str, str, str, str]


@dataclass(frozen=True)
class Point:
    """Median and interquartile range for one bar."""

    value: float
    lower: float
    upper: float
    count: int


def _template() -> go.layout.Template:
    """Return the Plotly theme used by Ion's example notebooks."""
    template = go.layout.Template(pio.templates["plotly_white"])
    layout = go.Layout(template.layout)
    layout.update(
        colorway=list(COLORS.values()),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#6b7688",
        xaxis_gridcolor="rgba(107,118,136,0.2)",
        xaxis_zerolinecolor="rgba(107,118,136,0.4)",
        yaxis_gridcolor="rgba(107,118,136,0.2)",
        yaxis_zerolinecolor="rgba(107,118,136,0.4)",
    )
    template.layout = layout
    return template


def load_results(path: Path) -> list[dict[str, Any]]:
    """Load every benchmark JSON record below ``path``."""
    records = [json.loads(result.read_text()) for result in sorted(path.rglob("*.json"))]
    if not records:
        raise ValueError(f"No benchmark JSON files found below {path}")
    return records


def _group(records: list[dict[str, Any]]) -> dict[GroupKey, list[dict[str, Any]]]:
    grouped = defaultdict(list)
    for record in records:
        key = (
            record["model"],
            record["size"],
            record["framework"],
            record["mode"],
            record["metric"],
        )
        grouped[key].append(record)
    return grouped


def _point(records: list[dict[str, Any]], kind: ValueKind) -> Point | None:
    if kind == "time":
        values = [sample for record in records for sample in record["samples_ms"]]
    elif kind == "throughput":
        values = [record["throughput"] for record in records if record["throughput"] is not None]
    else:
        values = [
            record["peak_memory_bytes"] / 2**30
            for record in records
            if record["peak_memory_bytes"] is not None
        ]

    if not values:
        return None
    median = statistics.median(values)
    if len(values) == 1:
        lower = upper = median
    else:
        lower, _, upper = statistics.quantiles(values, n=4, method="inclusive")
    return Point(median, lower, upper, len(values))


def _ordered(values: set[str], order: tuple[str, ...]) -> list[str]:
    return [value for value in order if value in values]


def _style(figure: go.Figure, title: str, *, height: int) -> None:
    figure.update_layout(
        title=title,
        height=height,
        template=_template(),
        barmode="group",
        bargap=0.18,
        bargroupgap=0.06,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.04,
            xanchor="left",
            x=0,
        ),
        margin=dict(l=70, r=30, t=100, b=60),
        hoverlabel=dict(align="left"),
    )
    figure.update_xaxes(showgrid=False)
    figure.update_yaxes(rangemode="tozero")


def _bar(
    series: tuple[str, str],
    x: list[str],
    points: list[Point | None],
    *,
    showlegend: bool,
    unit: str,
) -> go.Bar:
    values = [point.value if point is not None else None for point in points]
    upper = [point.upper - point.value if point is not None else None for point in points]
    lower = [point.value - point.lower if point is not None else None for point in points]
    counts = [point.count if point is not None else 0 for point in points]
    return go.Bar(
        x=x,
        y=values,
        name=LABELS[series],
        legendgroup=LABELS[series],
        showlegend=showlegend,
        marker_color=COLORS[series],
        opacity=0.85,
        error_y=dict(type="data", array=upper, arrayminus=lower, symmetric=False),
        customdata=counts,
        hovertemplate=(
            f"%{{x}}<br>{LABELS[series]}<br>%{{y:,.3f}} {unit}<br>n=%{{customdata}}<extra></extra>"
        ),
    )


def latency_figure(records: list[dict[str, Any]]) -> go.Figure:
    """Create the steady-state latency figure."""
    grouped = _group(records)
    models = _ordered({record["model"] for record in records}, MODELS)
    sizes = _ordered({record["size"] for record in records}, SIZES)
    metrics = (
        ("forward", "Forward"),
        ("forward_backward", "Forward + backward"),
        ("full_step", "Full step"),
    )
    titles = [f"{model.upper()} · {size.title()}" for model in models for size in sizes]
    figure = make_subplots(
        rows=len(models),
        cols=len(sizes),
        subplot_titles=titles,
        vertical_spacing=0.1,
        horizontal_spacing=0.06,
    )
    shown = set()
    for row, model in enumerate(models, start=1):
        for col, size in enumerate(sizes, start=1):
            for series in SERIES:
                points = [
                    _point(grouped[(model, size, *series, metric)], "time") for metric, _ in metrics
                ]
                if not any(point is not None for point in points):
                    continue
                figure.add_trace(
                    _bar(
                        series,
                        [label for _, label in metrics],
                        points,
                        showlegend=series not in shown,
                        unit="ms",
                    ),
                    row=row,
                    col=col,
                )
                shown.add(series)
    _style(
        figure,
        "Steady-state latency",
        height=max(420, 300 * len(models)),
    )
    for row in range(1, len(models) + 1):
        figure.update_yaxes(title_text="Median latency (ms)", row=row, col=1)
    return figure


def metric_figure(
    records: list[dict[str, Any]],
    metric: str,
    kind: ValueKind,
    title: str,
    y_title: str,
    unit: str,
) -> go.Figure:
    """Create one figure comparing a metric across model sizes."""
    grouped = _group(records)
    models = _ordered({record["model"] for record in records}, MODELS)
    sizes = _ordered({record["size"] for record in records}, SIZES)
    figure = make_subplots(
        rows=1,
        cols=len(models),
        subplot_titles=[model.upper() for model in models],
        horizontal_spacing=0.07,
    )
    shown = set()
    for col, model in enumerate(models, start=1):
        for series in SERIES:
            points = [_point(grouped[(model, size, *series, metric)], kind) for size in sizes]
            if not any(point is not None for point in points):
                continue
            figure.add_trace(
                _bar(
                    series,
                    [size.title() for size in sizes],
                    points,
                    showlegend=series not in shown,
                    unit=unit,
                ),
                row=1,
                col=col,
            )
            shown.add(series)
    _style(figure, title, height=440)
    figure.update_yaxes(title_text=y_title, row=1, col=1)
    return figure


def generate(
    results: Path,
    output: Path,
    *,
    include_plotlyjs: Literal["cdn", "inline"] = "cdn",
) -> list[Path]:
    """Render all available benchmark figures and return their paths."""
    records = load_results(results)
    figures = {
        "latency.html": latency_figure(records),
        "throughput.html": metric_figure(
            records,
            "full_step",
            "throughput",
            "Training throughput",
            "Samples or tokens per second",
            "/s",
        ),
        "compile-time.html": metric_figure(
            records,
            "compile",
            "time",
            "Compile time",
            "Estimated compile time (ms)",
            "ms",
        ),
        "first-step.html": metric_figure(
            records,
            "first_step",
            "time",
            "First training step",
            "First-step latency (ms)",
            "ms",
        ),
        "peak-memory.html": metric_figure(
            records,
            "peak_memory",
            "memory",
            "Peak device memory",
            "Peak memory (GiB)",
            "GiB",
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, figure in figures.items():
        path = output / name
        figure.write_html(
            path,
            include_plotlyjs=include_plotlyjs,
            full_html=True,
            config={"displaylogo": False, "responsive": True},
        )
        paths.append(path)
    return paths


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--inline",
        action="store_true",
        help="Embed Plotly in each HTML file instead of loading it from the CDN.",
    )
    args = parser.parse_args()
    output = args.output or args.results / "plots"
    paths = generate(
        args.results,
        output,
        include_plotlyjs="inline" if args.inline else "cdn",
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
