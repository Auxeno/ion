"""Tests for benchmark reports."""

from dataclasses import replace

import pytest

from benchmarks.report import summarize


def test_summarize_formats_metrics(tmp_path, result):
    results = [
        replace(result, metric="forward", samples_ms=[0.5]),
        replace(result, metric="full_step", samples_ms=[1500]),
        replace(
            result,
            metric="peak_memory",
            samples_ms=[],
            throughput=None,
            peak_memory_bytes=2**30,
        ),
    ]
    for index, value in enumerate(results):
        value.write(tmp_path / f"{index}.json")

    report = summarize(tmp_path)

    assert "## MLP" in report
    assert "500 µs" in report
    assert "1.50 s" in report
    assert "500,000 /s" in report
    assert "1.00 GiB" in report


def test_summarize_requires_results(tmp_path):
    with pytest.raises(ValueError, match="No benchmark JSON"):
        summarize(tmp_path)
