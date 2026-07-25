"""Validates supplementary docstring files in docs/docstrings/ against real signatures."""

import ast
import inspect
import math
import numbers
import operator
from pathlib import Path

import pytest

import ion
import ion.gnn
import ion.nn

griffe = pytest.importorskip("griffe")

DOCSTRING_FILES = sorted((Path(__file__).parents[2] / "docs" / "docstrings").glob("*.md"))


def _parameter_name(param: inspect.Parameter) -> str:
    """Format a signature parameter as it appears in a NumPy docstring."""
    if param.kind is inspect.Parameter.VAR_POSITIONAL:
        return f"*{param.name}"
    if param.kind is inspect.Parameter.VAR_KEYWORD:
        return f"**{param.name}"
    return param.name


def _numeric_expression(node: ast.AST) -> int | float:
    """Evaluate a numeric default expression using a small, safe grammar."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name) and node.id == "pi":
        return math.pi
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _numeric_expression(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        operations = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
        }
        operation = operations.get(type(node.op))
        if operation is not None:
            return operation(_numeric_expression(node.left), _numeric_expression(node.right))
    raise ValueError("Not a supported numeric expression")


def _default_matches(documented: str, actual: object) -> bool:
    """Compare a readable documented default with its runtime value."""
    if documented == repr(actual):
        return True

    public_name = getattr(actual, "__name__", None)
    if public_name is not None and documented.rsplit(".", 1)[-1] == public_name:
        return True

    if isinstance(actual, numbers.Real):
        try:
            value = _numeric_expression(ast.parse(documented, mode="eval").body)
        except (SyntaxError, ValueError):
            return False
        return math.isclose(value, actual)

    return False


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
        assert documented == [_parameter_name(p) for p in expected], f"{name}: parameter mismatch"

        for doc, sig in zip(params.value, expected):
            if doc.default is not None:
                assert sig.default is not inspect.Parameter.empty, f"{name}: {doc.name} default"
                assert _default_matches(doc.default, sig.default), (
                    f"{name}: {doc.name} default mismatch"
                )
