"""Benchmark cold and warm model representation latency."""

import argparse
import gc
import statistics
import time
from collections.abc import Callable

import jax

import ion.nn as nn

from .configs import MODELS, SIZES, ModelConfig, ModelName, get_config
from .implementations.ion import gpt, mlp, resnet

_CREATE: dict[ModelName, Callable[..., nn.Module]] = {
    "mlp": mlp.create_model,
    "resnet": resnet.create_model,
    "gpt": gpt.create_model,
}


def measure(config: ModelConfig, warmup: int) -> tuple[float, float, int, int]:
    """Return cold and warm latency, parameter count and rendered character count."""
    model = _CREATE[config.model](config, key=jax.random.key(0))
    jax.block_until_ready(model)
    jax.clear_caches()

    start = time.perf_counter()
    rendered = repr(model)
    cold = time.perf_counter() - start

    samples = []
    for _ in range(warmup):
        start = time.perf_counter()
        rendered = repr(model)
        samples.append(time.perf_counter() - start)

    return cold, statistics.median(samples), model.num_params, len(rendered)


def main() -> None:
    """Measure selected Ion benchmark models and print CSV rows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--sizes", nargs="+", choices=SIZES, default=SIZES)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    if args.warmup <= 0:
        parser.error("--warmup must be positive")

    print("model,size,params,cold_ms,warm_ms,characters")
    for model_name in args.models:
        for size in args.sizes:
            config = get_config(model_name, size)
            cold, warm, num_params, characters = measure(config, args.warmup)
            print(
                f"{model_name},{size},{num_params},"
                f"{cold * 1000:.1f},{warm * 1000:.1f},{characters}"
            )
            gc.collect()
            jax.clear_caches()


if __name__ == "__main__":
    main()
