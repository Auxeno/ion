"""Tests for benchmark execution."""

from dataclasses import replace

import pytest

from benchmarks.configs import get_config
from benchmarks.runner import run_metrics


@pytest.fixture
def workload(monkeypatch):
    config = replace(
        get_config("mlp", "tiny"),
        batch_size=2,
        input_dim=8,
        width=16,
        depth=2,
        num_classes=4,
    )

    class Workload:
        device_name = "test"
        framework_version = "test"
        software = {}
        parameter_count = 1

        def __init__(self):
            self.config = config
            self.prepared = []
            self.calls = {"forward": 0, "forward_backward": 0, "full_step": 0}
            self.memory_resets = 0

        def prepare(self, metric, *, compiled):
            self.prepared.append((metric, compiled))

            def operation():
                self.calls[metric] += 1
                return self.calls[metric]

            return operation

        def synchronize(self, value):
            pass

        def peak_memory(self):
            return 1024

        def reset_peak_memory(self):
            self.memory_resets += 1

    instance = Workload()

    class Implementation:
        @staticmethod
        def Workload(config, *, seed):
            return instance

    monkeypatch.setattr("benchmarks.runner.importlib.import_module", lambda name: Implementation)
    monkeypatch.setattr(
        "benchmarks.runner.system_metadata",
        lambda: {"python": "test", "platform": "test", "revision": "test"},
    )
    return instance


def test_run_metrics_reuses_full_step(workload):
    results = run_metrics(
        "ion",
        "compiled",
        "mlp",
        "tiny",
        warmup_steps=2,
        measured_steps=3,
    )

    assert tuple(results) == (
        "full_step",
        "compile",
        "first_step",
        "peak_memory",
        "forward",
        "forward_backward",
    )
    assert workload.prepared == [
        ("full_step", True),
        ("forward", True),
        ("forward_backward", True),
    ]
    assert workload.calls == {"forward": 5, "forward_backward": 5, "full_step": 6}
    assert workload.memory_resets == 1
    assert len(results["full_step"].samples_ms) == 3
    assert results["full_step"].throughput is not None
    assert results["peak_memory"].peak_memory_bytes == 1024


def test_run_metrics_honours_filters(workload):
    results = run_metrics(
        "pytorch",
        "eager",
        "mlp",
        "tiny",
        warmup_steps=1,
        measured_steps=1,
        metrics=("forward", "compile"),
    )

    assert tuple(results) == ("forward",)
    assert workload.prepared == [("forward", False)]


def test_first_step_does_not_run_warmups(workload):
    result = run_metrics(
        "ion",
        "compiled",
        "mlp",
        "tiny",
        warmup_steps=5,
        measured_steps=10,
        metrics=("first_step",),
    )["first_step"]

    assert workload.calls["full_step"] == 1
    assert result.warmup_steps == 0
    assert result.measured_steps == 1


@pytest.mark.parametrize(
    ("framework", "mode", "metrics", "message"),
    [
        ("ion", "eager", ("forward",), "only benchmarked for PyTorch"),
        ("pytorch", "eager", ("compile",), "requires compiled mode"),
    ],
)
def test_invalid_modes(framework, mode, metrics, message):
    with pytest.raises(ValueError, match=message):
        run_metrics(framework, mode, "mlp", "tiny", metrics=metrics)


@pytest.mark.parametrize(("warmup", "steps"), [(-1, 1), (0, 0)])
def test_invalid_step_counts(warmup, steps):
    with pytest.raises(ValueError, match="warmup_steps"):
        run_metrics(
            "ion",
            "compiled",
            "mlp",
            "tiny",
            warmup_steps=warmup,
            measured_steps=steps,
        )
