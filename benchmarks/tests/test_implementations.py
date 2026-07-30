"""Smoke tests for installed benchmark frameworks."""

from dataclasses import replace
from importlib.util import find_spec

import pytest

from benchmarks.configs import get_config

CONFIGS = (
    replace(
        get_config("mlp", "tiny"),
        batch_size=2,
        input_dim=8,
        width=16,
        depth=3,
        num_classes=4,
    ),
    replace(
        get_config("resnet", "tiny"),
        batch_size=1,
        image_size=32,
        block_depths=(1, 1, 1, 1),
        num_classes=4,
    ),
    replace(
        get_config("gpt", "tiny"),
        batch_size=1,
        seq_len=8,
        vocab_size=32,
        width=16,
        depth=1,
        num_heads=4,
    ),
)


@pytest.mark.parametrize("config", CONFIGS, ids=lambda config: config.model)
def test_ion_steps(config):
    from benchmarks.implementations.ion import Workload

    workload = Workload(config, seed=0)
    initial_count = workload.parameter_count
    for metric in ("forward", "forward_backward", "full_step"):
        value = workload.prepare(metric, compiled=True)()
        workload.synchronize(value)
    assert workload.parameter_count == initial_count


@pytest.mark.skipif(find_spec("flax") is None, reason="Flax is not installed")
@pytest.mark.parametrize("config", CONFIGS, ids=lambda config: config.model)
def test_nnx_matches_ion_parameter_count(config):
    from benchmarks.implementations.ion import Workload as IonWorkload
    from benchmarks.implementations.nnx import Workload as NNXWorkload

    assert (
        IonWorkload(config, seed=0).parameter_count == NNXWorkload(config, seed=0).parameter_count
    )


@pytest.mark.skipif(find_spec("equinox") is None, reason="Equinox is not installed")
@pytest.mark.parametrize("config", CONFIGS, ids=lambda config: config.model)
def test_equinox_matches_ion_parameter_count(config):
    from benchmarks.implementations.equinox import Workload as EquinoxWorkload
    from benchmarks.implementations.ion import Workload as IonWorkload

    assert (
        IonWorkload(config, seed=0).parameter_count
        == EquinoxWorkload(config, seed=0).parameter_count
    )


@pytest.mark.skipif(find_spec("torch") is None, reason="PyTorch is not installed")
@pytest.mark.parametrize("config", CONFIGS, ids=lambda config: config.model)
def test_pytorch_matches_ion_parameter_count(config):
    from benchmarks.implementations.ion import Workload as IonWorkload
    from benchmarks.implementations.pytorch import Workload as PyTorchWorkload

    assert (
        IonWorkload(config, seed=0).parameter_count
        == PyTorchWorkload(config, seed=0).parameter_count
    )
