"""Summarize benchmark results as Markdown."""

import argparse
import statistics
from collections import defaultdict
from pathlib import Path

from .analysis import balance_results
from .configs import MODEL_LABELS
from .protocol import Result

TIME_METRICS = ("forward", "forward_backward", "full_step", "compile")


def _format_time(value: float | None) -> str:
    if value is None:
        return "—"
    if value < 1:
        return f"{value * 1000:.0f} µs"
    if value < 1000:
        return f"{value:.2f} ms"
    return f"{value / 1000:.2f} s"


def summarize(path: Path) -> str:
    """Return a Markdown summary of all results below a path."""
    results = [Result.read(result) for result in sorted(path.rglob("*.json"))]
    if not results:
        raise ValueError(f"No benchmark JSON files found below {path}")
    results = balance_results(results)

    # Group repetitions by displayed row and metric
    grouped = defaultdict(list)
    for result in results:
        key = result.model, result.size, result.framework, result.mode, result.metric
        grouped[key].append(result)

    # Render one table per model family
    lines = ["# Benchmark results", ""]
    for model in sorted({result.model for result in results}):
        lines.extend(
            [
                f"## {MODEL_LABELS[model]}",  # pyright: ignore[reportArgumentType]
                "",
                "| Size | Framework | Mode | Forward | Forward + backward | "
                "Full step | Compile | Throughput | Peak memory |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        combinations = sorted(
            {
                (result.size, result.framework, result.mode)
                for result in results
                if result.model == model
            }
        )
        for size, framework, mode in combinations:
            times = {}
            for metric in TIME_METRICS:
                samples = [
                    sample
                    for result in grouped[(model, size, framework, mode, metric)]
                    for sample in result.samples_ms
                ]
                times[metric] = statistics.median(samples) if samples else None

            throughput = [
                result.throughput
                for result in grouped[(model, size, framework, mode, "full_step")]
                if result.throughput is not None
            ]
            memory = [
                result.peak_memory_bytes
                for result in grouped[(model, size, framework, mode, "peak_memory")]
                if result.peak_memory_bytes is not None
            ]
            throughput_text = f"{statistics.median(throughput):,.0f} /s" if throughput else "—"
            memory_text = f"{max(memory) / 2**30:.2f} GiB" if memory else "—"
            lines.append(
                f"| {size} | {framework} | {mode} | "
                f"{_format_time(times['forward'])} | "
                f"{_format_time(times['forward_backward'])} | "
                f"{_format_time(times['full_step'])} | "
                f"{_format_time(times['compile'])} | "
                f"{throughput_text} | {memory_text} |"
            )
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    """Write a report from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = summarize(args.results)
    if args.output is None:
        print(report)
    else:
        args.output.write_text(report + "\n")


if __name__ == "__main__":
    main()
