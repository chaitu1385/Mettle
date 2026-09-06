"""Architecture tests: the dependency direction is a rule, not a convention.

Layered so that the domain (actions, schemas, policy, storage) never depends on
the delivery mechanism (the LLM vendor, LangGraph, the terminal). A new module
has to be placed in the stack deliberately, and vendor imports stay penned into
the one module that owns each boundary.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

PACKAGE = pathlib.Path(__file__).resolve().parent.parent / "coach_rl"

#: Lower ranks may not import higher ranks. Equal ranks may not import each other.
LAYERS: dict[str, int] = {
    "actions": 0,        # the action space: depends on nothing
    "config": 0,         # settings and thresholds
    "schemas": 1,        # domain objects
    "prompts": 1,        # prompt text + version
    "llm": 1,            # the model boundary (Protocol + Anthropic client)
    "storage": 2,        # persistence
    "policy": 2,         # decision rules
    "graph": 3,          # the decision loop
    "judge": 3,          # the reward call
    "stats": 3,          # analysis over stored rows
    "session": 4,        # orchestration
    "report": 4,         # rendering of stats
    "cli.run": 5,
    "cli.replay": 5,
    "cli.stats": 5,
}

#: Third-party imports that belong to exactly one module.
VENDOR_OWNERS = {"anthropic": {"llm"}, "langgraph": {"graph"}, "scipy": {"stats"}}


def module_name(path: pathlib.Path) -> str:
    relative = path.relative_to(PACKAGE).with_suffix("")
    return ".".join(relative.parts)


def modules() -> list[pathlib.Path]:
    return [p for p in sorted(PACKAGE.rglob("*.py")) if p.name != "__init__.py"]


def internal_imports(path: pathlib.Path) -> list[str]:
    """Names imported from within the package, e.g. `from .storage import ...`."""
    tree = ast.parse(path.read_text())
    return [
        node.module.lstrip(".")
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level and node.module
    ]


def third_party_roots(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text())
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_every_module_is_placed_in_a_layer():
    assert {module_name(p) for p in modules()} == set(LAYERS)


@pytest.mark.parametrize("path", modules(), ids=module_name)
def test_imports_only_point_downwards(path):
    importer = module_name(path)
    for imported in internal_imports(path):
        assert imported in LAYERS, f"{importer} imports unplaced module {imported}"
        assert LAYERS[imported] < LAYERS[importer], (
            f"{importer} (layer {LAYERS[importer]}) must not import "
            f"{imported} (layer {LAYERS[imported]})"
        )


@pytest.mark.parametrize("path", modules(), ids=module_name)
def test_vendor_imports_stay_in_the_module_that_owns_the_boundary(path):
    name = module_name(path)
    for vendor, owners in VENDOR_OWNERS.items():
        if vendor in third_party_roots(path) and name not in owners:
            pytest.fail(
                f"{name} imports {vendor}; only {sorted(owners)} may. "
                "Depend on the seam that module exposes instead."
            )
