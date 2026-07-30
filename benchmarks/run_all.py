"""Run a benchmark matrix using one subprocess per case."""

import argparse
import itertools
import subprocess
import sys
from pathlib import Path

FRAMEWORKS = ("ion", "equinox", "nnx", "pytorch")
MODELS = ("mlp", "resnet", "gpt")
SIZES = ("tiny", "small", "medium")
METRICS = (
    "forward",
    "forward_backward",
    "full_step",
    "compile",
    "first_step",
    "peak_memory",
)


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frameworks", nargs="+", choices=FRAMEWORKS, default=FRAMEWORKS)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--sizes", nargs="+", choices=SIZES, default=SIZES)
    parser.add_argument("--metrics", nargs="+", choices=METRICS, default=METRICS)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cases = itertools.product(
        args.frameworks,
        args.models,
        args.sizes,
        args.metrics,
        range(args.repetitions),
    )
    for framework, model, size, metric, repetition in cases:
        modes = ("eager", "compiled") if framework == "pytorch" else ("compiled",)
        for mode in modes:
            if mode == "eager" and metric == "compile":
                continue
            output = args.output / framework / mode / model / size / f"{metric}-{repetition}.json"
            command = [
                sys.executable,
                "-m",
                "benchmarks.runner",
                framework,
                model,
                size,
                metric,
                "--mode",
                mode,
                "--warmup",
                str(args.warmup),
                "--steps",
                str(args.steps),
                "--seed",
                "0",
                "--output",
                str(output),
            ]
            print(" ".join(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
