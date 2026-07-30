"""Tests for benchmark configurations."""

import pytest

from benchmarks.configs import CONFIGS, get_config


@pytest.mark.parametrize("model", ("mlp", "resnet", "gpt"))
def test_sizes(model):
    assert set(CONFIGS[model]) == {"tiny", "small", "medium"}


def test_gpt_units_are_tokens():
    for size in ("tiny", "small", "medium"):
        config = get_config("gpt", size)
        assert config.units_per_step == 4096


def test_other_units_are_samples():
    config = get_config("resnet", "tiny")
    assert config.units_per_step == config.batch_size


def test_medium_sizes():
    mlp = get_config("mlp", "medium")
    resnet = get_config("resnet", "medium")
    gpt = get_config("gpt", "medium")

    assert (mlp.depth, mlp.width) == (12, 2048)
    assert (resnet.block_depths, resnet.resnet_width) == ((3, 4, 23, 3), 64)
    assert (gpt.depth, gpt.width, gpt.num_heads, gpt.seq_len) == (12, 768, 12, 512)
