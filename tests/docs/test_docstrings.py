"""Validates supplementary docstring files in docs/docstrings/ against real signatures."""

import inspect
from pathlib import Path

import pytest

import ion
import ion.gnn
import ion.nn

griffe = pytest.importorskip("griffe")

DOCSTRING_FILES = sorted((Path(__file__).parents[2] / "docs" / "docstrings").glob("*.md"))


class TestSupplementaryDocstrings:
    @pytest.mark.parametrize("path", DOCSTRING_FILES, ids=lambda p: p.stem)
    def test_parameters_match_signature(self, path: Path):
        """Documented parameter names, order, and defaults match the real signature."""
        name = path.stem
        root, _, attr = name.partition(".")
        obj = next((getattr(m, root) for m in (ion.nn, ion, ion.gnn) if hasattr(m, root)), None)
        assert obj is not None, f"{name}: not found in ion, ion.nn, or ion.gnn"
        obj = getattr(obj, attr) if attr else obj

        sections = griffe.Docstring(path.read_text(), parser="numpy").parse()
        params = next((s for s in sections if s.kind.value == "parameters"), None)

        if isinstance(obj, property):
            assert params is None, f"{name}: properties take no parameters"
            return

        expected = [p for p in inspect.signature(obj).parameters.values() if p.name != "self"]

        if not expected:
            assert params is None, f"{name}: parameterless callable documents parameters"
            return

        assert params is not None, f"{name}: no Parameters section"

        documented = [p.name for p in params.value]
        assert documented == [p.name for p in expected], f"{name}: parameter mismatch"

        for doc, sig in zip(params.value, expected):
            if doc.default is not None:
                assert sig.default is not inspect.Parameter.empty, f"{name}: {doc.name} default"
                assert doc.default == repr(sig.default), f"{name}: {doc.name} default mismatch"
