"""Tests for benchmark matrix execution."""

import sys

from benchmarks.run_all import main


def test_dry_run_only_requests_missing_metrics(tmp_path, monkeypatch, capsys):
    complete = tmp_path / "ion" / "compiled" / "mlp" / "tiny" / "forward-0.json"
    complete.parent.mkdir(parents=True)
    complete.touch()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmarks.run_all",
            "--frameworks",
            "ion",
            "--models",
            "mlp",
            "--sizes",
            "tiny",
            "--metrics",
            "forward",
            "compile",
            "--repetitions",
            "1",
            "--output",
            str(tmp_path),
            "--dry-run",
        ],
    )

    main()

    output = capsys.readouterr().out
    assert "1 cases: 1 to run, 0 complete" in output
    assert "--metrics compile --output" in output
