"""One architecture rule: each third-party boundary lives in exactly one module.

The domain -- actions, schemas, policy, storage -- must not know who the model
vendor is. That is the constraint worth failing a build over; the rest of the
layering is visible in the imports themselves and does not need a test to
restate it.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

PACKAGE = pathlib.Path(__file__).resolve().parent.parent / "coach_rl"

VENDOR_OWNERS = {"anthropic": "llm", "langgraph": "graph", "scipy": "stats"}

MODULES = [p for p in sorted(PACKAGE.rglob("*.py")) if p.name != "__init__.py"]


def module_name(path: pathlib.Path) -> str:
    return ".".join(path.relative_to(PACKAGE).with_suffix("").parts)


@pytest.mark.parametrize("path", MODULES, ids=module_name)
def test_vendor_imports_stay_in_the_module_that_owns_the_boundary(path):
    name = module_name(path)
    roots: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            roots.add(node.module.split(".")[0])
    for vendor, owner in VENDOR_OWNERS.items():
        if vendor in roots and name != owner:
            pytest.fail(
                f"{name} imports {vendor}; only {owner}.py may. "
                "Depend on the seam that module exposes instead."
            )
