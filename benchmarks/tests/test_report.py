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
    assert "667 /s" in report
    assert "1.00 GiB" in report


def test_summarize_requires_results(tmp_path):
    with pytest.raises(ValueError, match="No benchmark JSON"):
        summarize(tmp_path)


def test_summarize_balances_repetitions_and_samples(tmp_path, result):
    records = [
        replace(result, samples_ms=[1.0] * 30),
        replace(result, samples_ms=[1.0] * 30),
        replace(result, samples_ms=[100.0] * 100),
        replace(result, framework="equinox", samples_ms=[2.0] * 30),
        replace(result, framework="equinox", samples_ms=[2.0] * 30),
    ]
    for index, record in enumerate(records):
        record.write(tmp_path / f"{index}.json")

    report = summarize(tmp_path)

    assert "| ion | compiled |" in report
    assert "1.00 ms" in report
    assert "100.00 ms" not in report
