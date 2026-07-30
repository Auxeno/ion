"""Tests for benchmark result serialization."""


def test_result_round_trip(tmp_path, result):
    path = tmp_path / "nested" / "result.json"
    result.write(path)

    assert type(result).read(path) == result
    assert path.read_text().endswith("\n")
