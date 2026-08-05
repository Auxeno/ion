"""Run one benchmark case in an isolated process."""

import argparse
import importlib
import json
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from .configs import MODELS, SIZES, ModelName, ModelSize, get_config
from .protocol import (
    FRAMEWORKS,
    METRICS,
    MODES,
    Framework,
    Metric,
    Mode,
    Operation,
    Result,
    Workload,
    system_metadata,
)

WARMUP_STEPS = 5
MEASURED_STEPS = 50


def _time(operation: Callable[[], Any], workload: Workload) -> float:
    started = time.perf_counter()
    value = operation()
    workload.synchronize(value)
    return (time.perf_counter() - started) * 1000


def run_metrics(
    framework: Framework,
    mode: Mode,
    model: ModelName,
    size: ModelSize,
    *,
    warmup_steps: int = WARMUP_STEPS,
    measured_steps: int = MEASURED_STEPS,
    seed: int = 0,
    metrics: tuple[Metric, ...] = METRICS,
) -> dict[Metric, Result]:
    """Run selected metrics from one workload."""
    # Validate the requested benchmark combination
    if mode == "eager" and framework != "pytorch":
        raise ValueError("Eager mode is only benchmarked for PyTorch")
    if warmup_steps < 0 or measured_steps < 1:
        raise ValueError("warmup_steps must be non-negative and measured_steps must be positive")

    compiled = mode == "compiled"
    metrics = tuple(metric for metric in metrics if compiled or metric != "compile")
    if not metrics:
        raise ValueError("Compile time requires compiled mode")

    # Load one workload for all requested metrics
    implementation = importlib.import_module(f"benchmarks.implementations.{framework}")
    workload: Workload = implementation.Workload(get_config(model, size), seed=seed)
    metadata = system_metadata()
    results: dict[Metric, Result] = {}

    # Build results with metadata shared by the case
    def result(
        metric: Metric,
        samples: list[float],
        warmup: int,
        measured: int,
        peak_memory: int | None = None,
    ) -> Result:
        median = statistics.median(samples) if samples else None
        throughput = (
            workload.config.units_per_step / (median / 1000)
            if metric == "full_step" and median
            else None
        )
        return Result(
            framework=framework,
            framework_version=workload.framework_version,
            mode=mode,
            model=model,
            size=size,
            metric=metric,
            dtype="bfloat16",
            batch_size=workload.config.batch_size,
            parameter_count=workload.parameter_count,
            units_per_step=workload.config.units_per_step,
            samples_ms=samples,
            throughput=throughput,
            peak_memory_bytes=peak_memory,
            warmup_steps=warmup,
            measured_steps=measured,
            seed=seed,
            device=workload.device_name,
            software=workload.software,
            **metadata,
        )

    # Measure the first full step before compiling related operations can prime caches
    full_metrics = {"full_step", "compile", "peak_memory"} & set(metrics)
    if full_metrics:
        operation = workload.prepare("full_step", compiled=compiled)
        first_step = _time(operation, workload)

        for _ in range(warmup_steps):
            _time(operation, workload)
        if "peak_memory" in full_metrics:
            workload.reset_peak_memory()
        samples = [_time(operation, workload) for _ in range(measured_steps)]

        if "full_step" in full_metrics:
            results["full_step"] = result("full_step", samples, warmup_steps, measured_steps)
        if "compile" in full_metrics:
            results["compile"] = result(
                "compile",
                [max(0.0, first_step - statistics.median(samples))],
                0,
                1,
            )
        if "peak_memory" in full_metrics:
            results["peak_memory"] = result(
                "peak_memory",
                [],
                warmup_steps,
                measured_steps,
                workload.peak_memory(),
            )

    # Measure forward operations after the full training step
    for metric in ("forward", "forward_backward"):
        if metric not in metrics:
            continue
        operation = workload.prepare(cast(Operation, metric), compiled=compiled)
        for _ in range(warmup_steps):
            _time(operation, workload)
        samples = [_time(operation, workload) for _ in range(measured_steps)]
        results[metric] = result(metric, samples, warmup_steps, measured_steps)

    return results


def main() -> None:
    """Run a benchmark from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("framework", choices=FRAMEWORKS)
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("size", choices=SIZES)
    parser.add_argument("metric", choices=(*METRICS, "all"))
    parser.add_argument("--mode", choices=MODES, default="compiled")
    parser.add_argument("--warmup", type=int, default=WARMUP_STEPS)
    parser.add_argument("--steps", type=int, default=MEASURED_STEPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repetition", type=int, default=0)
    parser.add_argument("--metrics", nargs="+", choices=METRICS, default=METRICS)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    metrics = tuple(args.metrics) if args.metric == "all" else (cast(Metric, args.metric),)
    results = run_metrics(
        args.framework,
        args.mode,
        args.model,
        args.size,
        warmup_steps=args.warmup,
        measured_steps=args.steps,
        seed=args.seed,
        metrics=metrics,
    )

    # Print one JSON object or write results to their output paths
    if args.output is None:
        values = {metric: asdict(result) for metric, result in results.items()}
        print(json.dumps(values if args.metric == "all" else next(iter(values.values())), indent=2))
        return

    if args.metric != "all":
        next(iter(results.values())).write(args.output)
        return

    for metric, result in results.items():
        output = (
            args.output
            / args.framework
            / args.mode
            / args.model
            / args.size
            / f"{metric}-{args.repetition}.json"
        )
        if args.overwrite or not output.exists():
            result.write(output)


if __name__ == "__main__":
    main()
