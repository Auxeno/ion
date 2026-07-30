"""Shared benchmark protocol and result types."""

import dataclasses
import json
import platform
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Protocol

from .configs import ModelConfig

Framework = Literal["ion", "equinox", "nnx", "pytorch"]
Mode = Literal["compiled", "eager"]
Metric = Literal[
    "forward",
    "forward_backward",
    "full_step",
    "compile",
    "first_step",
    "peak_memory",
]


class Workload(Protocol):
    """A model and its framework-specific benchmark operations."""

    config: ModelConfig
    framework_version: str
    parameter_count: int
    software: dict[str, str]

    def prepare(self, metric: Metric, *, compiled: bool) -> Callable[[], Any]:
        """Return the operation measured for ``metric``."""
        raise NotImplementedError

    def synchronize(self, value: Any) -> None:
        """Wait until all device work producing ``value`` has completed."""
        raise NotImplementedError

    def peak_memory(self) -> int | None:
        """Return peak device memory in bytes when available."""
        raise NotImplementedError

    def reset_peak_memory(self) -> None:
        """Reset peak device memory statistics when supported."""
        raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class Result:
    """One isolated benchmark result."""

    framework: Framework
    framework_version: str
    mode: Mode
    model: str
    size: str
    metric: Metric
    dtype: str
    batch_size: int
    parameter_count: int
    units_per_step: int
    samples_ms: list[float]
    throughput: float | None
    peak_memory_bytes: int | None
    warmup_steps: int
    measured_steps: int
    seed: int
    python: str
    platform: str
    device: str
    software: dict[str, str]
    revision: str

    def write(self, path: Path) -> None:
        """Write the result as formatted JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dataclasses.asdict(self), indent=2) + "\n")


def system_metadata() -> dict[str, str]:
    """Return stable host metadata for a result."""
    try:
        revision = subprocess.run(
            ("git", "describe", "--always", "--dirty"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        revision = "unknown"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "revision": revision,
    }
