"""Run a benchmark matrix using one subprocess per case."""

import argparse
import itertools
import subprocess
import sys
import time
from pathlib import Path

from .configs import MODELS, SIZES
from .protocol import FRAMEWORKS, METRICS


def main() -> None:
    """Run the benchmark matrix."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frameworks", nargs="+", choices=FRAMEWORKS, default=FRAMEWORKS)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--sizes", nargs="+", choices=SIZES, default=SIZES)
    parser.add_argument("--metrics", nargs="+", choices=METRICS, default=METRICS)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Collect incomplete cases before starting the suite
    cases = []
    completed = 0
    for framework, model, size, repetition in itertools.product(
        args.frameworks,
        args.models,
        args.sizes,
        range(args.repetitions),
    ):
        modes = ("eager", "compiled") if framework == "pytorch" else ("compiled",)
        for mode in modes:
            metrics = [
                metric for metric in args.metrics if mode == "compiled" or metric != "compile"
            ]
            pending = [
                metric
                for metric in metrics
                if args.overwrite
                or not (
                    args.output / framework / mode / model / size / f"{metric}-{repetition}.json"
                ).exists()
            ]
            if pending:
                cases.append((framework, mode, model, size, repetition, pending))
            elif metrics:
                completed += 1

    print(
        f"{completed + len(cases)} cases: {len(cases)} to run, {completed} complete "
        f"({args.repetitions} repetitions, {args.warmup} warm-ups, {args.steps} steps)",
        flush=True,
    )

    # Run each case in its own isolated process
    started = time.perf_counter()
    for index, (framework, mode, model, size, repetition, metrics) in enumerate(cases, start=1):
        print(
            f"[{index}/{len(cases)}] {framework} {mode} {model} {size} repetition {repetition + 1}",
            flush=True,
        )
        command = [
            sys.executable,
            "-m",
            "benchmarks.runner",
            framework,
            model,
            size,
            "all",
            "--mode",
            mode,
            "--warmup",
            str(args.warmup),
            "--steps",
            str(args.steps),
            "--repetition",
            str(repetition),
            "--metrics",
            *metrics,
            "--output",
            str(args.output),
        ]
        if args.overwrite:
            command.append("--overwrite")

        if args.dry_run:
            print(" ".join(command), flush=True)
            continue

        subprocess.run(command, check=True)
        elapsed = time.perf_counter() - started
        remaining = elapsed / index * (len(cases) - index)
        print(
            f"Completed in {elapsed / 60:.1f} min, "
            f"approximately {remaining / 60:.1f} min remaining",
            flush=True,
        )


if __name__ == "__main__":
    main()
