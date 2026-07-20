"""Griffe extension that swaps in rich docstrings from docs/docstrings/.

Files are keyed by object name for module-level objects (Linear.md) and by
qualified name for members (Linear.__call__.md).
"""

from pathlib import Path
from typing import Any

import griffe

DOCSTRINGS = {p.stem: p for p in (Path(__file__).parent / "docstrings").glob("*.md")}


class RichDocstrings(griffe.Extension):
    def on_instance(self, *, obj: griffe.Object, **kwargs: Any) -> None:
        if obj.parent is None or obj.parent.is_module:
            key = obj.name
        else:
            key = f"{obj.parent.name}.{obj.name}"
        path = DOCSTRINGS.get(key)
        if path is not None:
            obj.docstring = griffe.Docstring(path.read_text(), parent=obj, parser="numpy")
