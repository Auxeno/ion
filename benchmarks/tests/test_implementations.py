"""Smoke tests for installed benchmark frameworks."""

from dataclasses import replace
from importlib.util import find_spec

import pytest

from benchmarks.configs import get_config


def _configs():
    yield replace(
        get_config("mlp", "tiny"),
        batch_size=2,
        input_dim=8,
        width=16,
        depth=3,
        num_classes=4,
    )
    yield replace(
        get_config("resnet", "tiny"),
        batch_size=1,
        image_size=32,
        block_depths=(1, 1, 1, 1),
        num_classes=4,
    )
    yield replace(
        get_config("gpt", "tiny"),
        batch_size=1,
        seq_len=8,
        vocab_size=32,
        width=16,
        depth=1,
        num_heads=4,
    )


@pytest.mark.parametrize("config", _configs())
def test_ion_steps(config):
    from benchmarks.implementations.ion import create_workload

    workload = create_workload(config, seed=0)
    initial_count = workload.parameter_count
    for metric in ("forward", "forward_backward", "full_step"):
        value = workload.prepare(metric, compiled=True)()
        workload.synchronize(value)
    assert workload.parameter_count == initial_count


@pytest.mark.skipif(find_spec("flax") is None, reason="Flax is not installed")
@pytest.mark.parametrize("config", _configs())
def test_nnx_matches_ion_parameter_count(config):
    from benchmarks.implementations.ion import create_workload as create_ion
    from benchmarks.implementations.nnx import create_workload as create_nnx

    assert create_ion(config, seed=0).parameter_count == create_nnx(config, seed=0).parameter_count


@pytest.mark.skipif(find_spec("equinox") is None, reason="Equinox is not installed")
@pytest.mark.parametrize("config", _configs())
def test_equinox_matches_ion_parameter_count(config):
    from benchmarks.implementations.equinox import create_workload as create_equinox
    from benchmarks.implementations.ion import create_workload as create_ion

    assert (
        create_ion(config, seed=0).parameter_count == create_equinox(config, seed=0).parameter_count
    )


@pytest.mark.skipif(find_spec("torch") is None, reason="PyTorch is not installed")
@pytest.mark.parametrize("config", _configs())
def test_pytorch_matches_ion_parameter_count(config):
    from benchmarks.implementations.ion import create_workload as create_ion
    from benchmarks.implementations.pytorch import create_workload as create_pytorch

    assert (
        create_ion(config, seed=0).parameter_count == create_pytorch(config, seed=0).parameter_count
    )
