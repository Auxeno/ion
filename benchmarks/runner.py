"""Run one benchmark case in an isolated process."""

import argparse
import importlib
import statistics
import time
from pathlib import Path
from typing import Any

from .configs import get_config
from .protocol import Framework, Metric, Mode, Result, Workload, system_metadata

WARMUP_STEPS = 10
MEASURED_STEPS = 100


def _time(operation, workload: Workload) -> tuple[float, Any]:
    started = time.perf_counter()
    value = operation()
    workload.synchronize(value)
    return (time.perf_counter() - started) * 1000, value


def run(
    framework: Framework,
    mode: Mode,
    model: str,
    size: str,
    metric: Metric,
    *,
    warmup_steps: int = WARMUP_STEPS,
    measured_steps: int = MEASURED_STEPS,
    seed: int = 0,
) -> Result:
    """Run and return one benchmark case."""
    if framework != "pytorch" and mode == "eager":
        raise ValueError("Eager mode is only benchmarked for PyTorch")
    if metric == "compile" and mode == "eager":
        raise ValueError("Compile time requires compiled mode")

    config = get_config(model, size)  # type: ignore[arg-type]
    implementation = importlib.import_module(f"benchmarks.implementations.{framework}")
    workload: Workload = implementation.create_workload(config, seed=seed)
    compiled = mode == "compiled"
    target: Metric = "full_step" if metric in {"compile", "first_step", "peak_memory"} else metric
    operation = workload.prepare(target, compiled=compiled)

    if metric in {"compile", "first_step"}:
        first_ms, _ = _time(operation, workload)
        if metric == "first_step":
            samples = [first_ms]
        else:
            steady = [_time(operation, workload)[0] for _ in range(5)]
            samples = [max(0.0, first_ms - statistics.median(steady))]
    elif metric == "peak_memory":
        for _ in range(warmup_steps):
            _time(operation, workload)
        workload.reset_peak_memory()
        _time(operation, workload)
        samples = []
    else:
        for _ in range(warmup_steps):
            _time(operation, workload)
        samples = [_time(operation, workload)[0] for _ in range(measured_steps)]

    median_ms = statistics.median(samples) if samples else None
    throughput = (
        config.units_per_step / (median_ms / 1000) if median_ms and metric == "full_step" else None
    )
    metadata = system_metadata()

    return Result(
        framework=framework,
        framework_version=workload.framework_version,
        mode=mode,
        model=model,
        size=size,
        metric=metric,
        dtype="bfloat16",
        batch_size=config.batch_size,
        parameter_count=workload.parameter_count,
        units_per_step=config.units_per_step,
        samples_ms=samples,
        throughput=throughput,
        peak_memory_bytes=workload.peak_memory() if metric == "peak_memory" else None,
        warmup_steps=warmup_steps if metric not in {"compile", "first_step"} else 0,
        measured_steps=measured_steps
        if metric not in {"compile", "first_step", "peak_memory"}
        else 1,
        seed=seed,
        device=implementation.device_name(),
        software=workload.software,
        **metadata,
    )


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("framework", choices=("ion", "equinox", "nnx", "pytorch"))
    parser.add_argument("model", choices=("mlp", "resnet", "gpt"))
    parser.add_argument("size", choices=("tiny", "small", "medium"))
    parser.add_argument(
        "metric",
        choices=(
            "forward",
            "forward_backward",
            "full_step",
            "compile",
            "first_step",
            "peak_memory",
        ),
    )
    parser.add_argument("--mode", choices=("compiled", "eager"), default="compiled")
    parser.add_argument("--warmup", type=int, default=WARMUP_STEPS)
    parser.add_argument("--steps", type=int, default=MEASURED_STEPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = run(
        args.framework,
        args.mode,
        args.model,
        args.size,
        args.metric,
        warmup_steps=args.warmup,
        measured_steps=args.steps,
        seed=args.seed,
    )
    if args.output is None:
        import dataclasses
        import json

        print(json.dumps(dataclasses.asdict(result), indent=2))
    else:
        result.write(args.output)


if __name__ == "__main__":
    main()
