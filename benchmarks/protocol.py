"""Shared benchmark types."""

import json
import platform
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from .configs import ModelConfig, ModelName, ModelSize

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
Operation = Literal["forward", "forward_backward", "full_step"]

FRAMEWORKS: tuple[Framework, ...] = ("ion", "equinox", "nnx", "pytorch")
MODES: tuple[Mode, ...] = ("compiled", "eager")
METRICS: tuple[Metric, ...] = (
    "forward",
    "forward_backward",
    "full_step",
    "compile",
    "first_step",
    "peak_memory",
)


class Workload(Protocol):
    """Framework-specific benchmark operations."""

    config: ModelConfig
    device_name: str
    framework_version: str
    parameter_count: int
    software: dict[str, str]

    def prepare(self, operation: Operation, *, compiled: bool) -> Callable[[], Any]: ...

    def synchronize(self, value: Any) -> None: ...

    def peak_memory(self) -> int | None: ...

    def reset_peak_memory(self) -> None: ...


@dataclass(frozen=True)
class Result:
    """One isolated benchmark result."""

    framework: Framework
    framework_version: str
    mode: Mode
    model: ModelName
    size: ModelSize
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

    @classmethod
    def read(cls, path: Path) -> "Result":
        """Read a result from JSON."""
        return cls(**json.loads(path.read_text()))

    def write(self, path: Path) -> None:
        """Write the result as formatted JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")


def system_metadata() -> dict[str, str]:
    """Return host metadata recorded with each result."""
    # Record the current source revision when git is available
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
