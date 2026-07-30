"""Render benchmark results as Plotly HTML files."""

import argparse
import statistics
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from .configs import MODELS, SIZES
from .protocol import Framework, Metric, Mode, Result

SERIES: dict[tuple[Framework, Mode], tuple[str, str]] = {
    ("ion", "compiled"): ("Ion", "#7c3aed"),
    ("equinox", "compiled"): ("Equinox", "#22d3ee"),
    ("nnx", "compiled"): ("Flax NNX", "#1fce9c"),
    ("pytorch", "compiled"): ("PyTorch compiled", "#ef5350"),
    ("pytorch", "eager"): ("PyTorch eager", "#f57e2c"),
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


def load_results(path: Path) -> list[Result]:
    """Load every benchmark result below a path."""
    results = [Result.read(result) for result in sorted(path.rglob("*.json"))]
    if not results:
        raise ValueError(f"No benchmark JSON files found below {path}")
    return results


def balance_results(results: list[Result]) -> list[Result]:
    """Match repetitions and sample counts across compared series."""
    comparisons = defaultdict(lambda: defaultdict(list))
    for result in results:
        comparison = result.model, result.size, result.metric
        series = result.framework, result.mode
        comparisons[comparison][series].append(result)

    balanced = []
    for series_results in comparisons.values():
        repetitions = min(map(len, series_results.values()))
        selected = [result for values in series_results.values() for result in values[:repetitions]]
        sample_counts = [len(result.samples_ms) for result in selected if result.samples_ms]
        samples_per_repetition = min(sample_counts, default=0)

        for result in selected:
            if not samples_per_repetition:
                balanced.append(result)
                continue
            samples = result.samples_ms[:samples_per_repetition]
            throughput = result.throughput
            if throughput is not None:
                throughput = result.units_per_step / (statistics.median(samples) / 1000)
            balanced.append(replace(result, samples_ms=samples, throughput=throughput))
    return balanced


def _group(results: list[Result]) -> dict[GroupKey, list[Result]]:
    grouped = defaultdict(list)
    for result in results:
        key = result.model, result.size, result.framework, result.mode, result.metric
        grouped[key].append(result)
    return grouped


def _point(results: list[Result], kind: ValueKind) -> Point | None:
    if kind == "time":
        values = [sample for result in results for sample in result.samples_ms]
    elif kind == "throughput":
        values = [result.throughput for result in results if result.throughput is not None]
    else:
        values = [
            result.peak_memory_bytes / 2**30
            for result in results
            if result.peak_memory_bytes is not None
        ]

    if not values:
        return None
    median = statistics.median(values)
    if len(values) == 1:
        lower = upper = median
    else:
        lower, _, upper = statistics.quantiles(values, n=4, method="inclusive")
    return Point(median, lower, upper, len(values))


def _bar(
    series: tuple[Framework, Mode],
    x: list[str],
    points: list[Point | None],
    *,
    showlegend: bool,
    unit: str,
) -> go.Bar:
    label, color = SERIES[series]
    values = [point.value if point else None for point in points]
    upper = [point.upper - point.value if point else None for point in points]
    lower = [point.value - point.lower if point else None for point in points]
    counts = [point.count if point else 0 for point in points]
    return go.Bar(
        x=x,
        y=values,
        name=label,
        legendgroup=label,
        showlegend=showlegend,
        marker_color=color,
        opacity=0.85,
        error_y=dict(type="data", array=upper, arrayminus=lower, symmetric=False),
        customdata=counts,
        hovertemplate=(
            f"%{{x}}<br>{label}<br>%{{y:,.3f}} {unit}<br>n=%{{customdata}}<extra></extra>"
        ),
    )


def _style(figure: go.Figure, title: str, *, height: int) -> None:
    template = go.layout.Template(pio.templates["plotly_white"])
    layout = go.Layout(template.layout)
    layout.update(
        colorway=[color for _, color in SERIES.values()],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#6b7688",
        xaxis_gridcolor="rgba(107,118,136,0.2)",
        xaxis_zerolinecolor="rgba(107,118,136,0.4)",
        yaxis_gridcolor="rgba(107,118,136,0.2)",
        yaxis_zerolinecolor="rgba(107,118,136,0.4)",
    )
    template.layout = layout

    figure.update_layout(
        title=title,
        height=height,
        template=template,
        barmode="group",
        bargap=0.18,
        bargroupgap=0.06,
        legend=dict(orientation="h", yanchor="top", y=-0.08, xanchor="right", x=1),
        margin=dict(l=70, r=30, t=70, b=100),
        hoverlabel=dict(align="left"),
    )
    figure.update_xaxes(showgrid=False)
    figure.update_yaxes(rangemode="tozero")


def latency_figure(results: list[Result]) -> go.Figure:
    """Plot forward, backward, and full-step execution time."""
    grouped = _group(results)
    models = [model for model in MODELS if any(result.model == model for result in results)]
    sizes = [size for size in SIZES if any(result.size == size for result in results)]
    metrics = (
        ("forward", "Forward"),
        ("forward_backward", "Forward + backward"),
        ("full_step", "Full step"),
    )
    figure = make_subplots(
        rows=len(models),
        cols=len(sizes),
        subplot_titles=[f"{model.upper()} · {size.title()}" for model in models for size in sizes],
        vertical_spacing=0.1,
        horizontal_spacing=0.06,
    )

    # Add one trace per available framework and mode
    shown = set()
    for row, model in enumerate(models, start=1):
        for col, size in enumerate(sizes, start=1):
            for series in SERIES:
                points = [
                    _point(grouped[(model, size, *series, metric)], "time") for metric, _ in metrics
                ]
                if not any(points):
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

    _style(figure, "Execution time", height=max(420, 300 * len(models)))
    for row in range(1, len(models) + 1):
        figure.update_yaxes(title_text="Median execution time (ms)", row=row, col=1)
    return figure


def metric_figure(
    results: list[Result],
    metric: Metric,
    kind: ValueKind,
    title: str,
    y_title: str,
    unit: str,
) -> go.Figure:
    """Plot one metric across model sizes."""
    grouped = _group(results)
    models = [model for model in MODELS if any(result.model == model for result in results)]
    sizes = [size for size in SIZES if any(result.size == size for result in results)]
    figure = make_subplots(
        rows=1,
        cols=len(models),
        subplot_titles=[model.upper() for model in models],
        horizontal_spacing=0.07,
    )

    # Add one trace per available framework and mode
    shown = set()
    for col, model in enumerate(models, start=1):
        for series in SERIES:
            points = [_point(grouped[(model, size, *series, metric)], kind) for size in sizes]
            if not any(points):
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
    """Render all benchmark figures."""
    records = balance_results(load_results(results))

    # Keep output filenames stable for docs and publication
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
    """Render plots from the command line."""
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
    for path in generate(
        args.results,
        output,
        include_plotlyjs="inline" if args.inline else "cdn",
    ):
        print(path)


if __name__ == "__main__":
    main()
