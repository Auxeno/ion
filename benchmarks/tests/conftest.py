"""Shared benchmark test fixtures."""

import pytest

from benchmarks.protocol import Result


@pytest.fixture
def result() -> Result:
    return Result(
        framework="ion",
        framework_version="test",
        mode="compiled",
        model="mlp",
        size="tiny",
        metric="full_step",
        dtype="bfloat16",
        batch_size=2,
        parameter_count=10,
        units_per_step=1000,
        samples_ms=[1.0, 2.0, 3.0],
        throughput=500_000,
        peak_memory_bytes=None,
        warmup_steps=1,
        measured_steps=3,
        seed=0,
        python="test",
        platform="test",
        device="test",
        software={},
        revision="test",
    )
