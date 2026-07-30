"""Summarize benchmark JSON results as Markdown."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def _format_time(value: float | None) -> str:
    if value is None:
        return "—"
    if value < 1:
        return f"{value * 1000:.0f} µs"
    if value < 1000:
        return f"{value:.2f} ms"
    return f"{value / 1000:.2f} s"


def _format_memory(value: int | None) -> str:
    return "—" if value is None else f"{value / 2**30:.2f} GiB"


def summarize(results: Path) -> str:
    """Return a Markdown summary of all results below ``results``."""
    records = [json.loads(path.read_text()) for path in results.rglob("*.json")]
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

    lines = ["# Benchmark results", ""]
    models = sorted({record["model"] for record in records})
    for model in models:
        lines.extend(
            [
                f"## {model.upper()}",
                "",
                "| Size | Framework | Mode | Forward | Forward + backward | "
                "Full step | Compile | First step | Throughput | Peak memory |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        combinations = sorted(
            {
                (record["size"], record["framework"], record["mode"])
                for record in records
                if record["model"] == model
            }
        )
        for size, framework, mode in combinations:

            def records_for(metric):
                return grouped[(model, size, framework, mode, metric)]

            def median_time(metric):
                values = [
                    sample for record in records_for(metric) for sample in record["samples_ms"]
                ]
                return statistics.median(values) if values else None

            full_step = records_for("full_step")
            throughput = (
                statistics.median(
                    record["throughput"] for record in full_step if record["throughput"] is not None
                )
                if any(record["throughput"] is not None for record in full_step)
                else None
            )
            memory = records_for("peak_memory")
            peak_memory = max(
                (
                    record["peak_memory_bytes"]
                    for record in memory
                    if record["peak_memory_bytes"] is not None
                ),
                default=None,
            )
            throughput_text = f"{throughput:,.0f} /s" if throughput is not None else "—"
            lines.append(
                f"| {size} | {framework} | {mode} | "
                f"{_format_time(median_time('forward'))} | "
                f"{_format_time(median_time('forward_backward'))} | "
                f"{_format_time(median_time('full_step'))} | "
                f"{_format_time(median_time('compile'))} | "
                f"{_format_time(median_time('first_step'))} | "
                f"{throughput_text} | "
                f"{_format_memory(peak_memory)} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    """Command-line entry point."""
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
