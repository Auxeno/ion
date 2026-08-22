"""Validates that copied Python snippets in the Overview run independently."""

import ast
import re
from pathlib import Path

import pytest

OVERVIEW = Path(__file__).parents[2] / "docs" / "overview.md"
PYTHON_FENCE = re.compile(r"^```python\n(.*?)^```$", re.MULTILINE | re.DOTALL)


def _snippets() -> list[tuple[int, str]]:
    source = OVERVIEW.read_text()
    snippets = []
    for match in PYTHON_FENCE.finditer(source):
        # Standalone examples import what they need, so a block without imports continues the page
        body = ast.parse(match.group(1)).body
        if not any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in body):
            continue
        snippets.append((source.count("\n", 0, match.start()) + 2, match.group(1)))

    return snippets


SNIPPETS = _snippets()


class TestOverviewSnippets:
    @pytest.mark.parametrize(
        ("line", "code"),
        SNIPPETS,
        ids=[f"line-{line}" for line, _ in SNIPPETS],
    )
    def test_runs_independently(self, line: int, code: str, tmp_path: Path, monkeypatch):
        """Each copied Python block runs in a fresh namespace and directory."""
        monkeypatch.chdir(tmp_path)
        namespace = {"__name__": "__main__"}
        exec(compile(code, f"{OVERVIEW}:{line}", "exec"), namespace)
