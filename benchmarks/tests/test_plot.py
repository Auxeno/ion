"""Tests for benchmark plotting."""

from dataclasses import replace

import pytest

from benchmarks.plot import balance_results, latency_figure, load_results


def test_load_results_round_trip(tmp_path, result):
    result.write(tmp_path / "result.json")
    assert load_results(tmp_path) == [result]


def test_load_results_requires_json(tmp_path):
    with pytest.raises(ValueError, match="No benchmark JSON"):
        load_results(tmp_path)


def test_balance_results_matches_repetitions_and_samples(result):
    records = [
        *(
            replace(
                result,
                samples_ms=[float(repetition * 100 + index) for index in range(100)],
            )
            for repetition in range(5)
        ),
        *(
            replace(
                result,
                framework="equinox",
                samples_ms=[float(repetition * 100 + index) for index in range(30)],
            )
            for repetition in range(2)
        ),
    ]

    balanced = balance_results(records)

    assert len(balanced) == 4
    assert {record.framework for record in balanced} == {"ion", "equinox"}
    assert all(len(record.samples_ms) == 30 for record in balanced)
    assert all(record.throughput != result.throughput for record in balanced)
    assert len(records[0].samples_ms) == 100


def test_latency_figure_uses_available_results(result):
    figure = latency_figure([result])
    assert len(figure.data) == 1
    assert figure.data[0].name == "Ion"
    assert list(figure.data[0].y) == [None, None, 2.0]
