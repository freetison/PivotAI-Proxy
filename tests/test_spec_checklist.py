"""The spec's test checklist must name tests that exist.

A checklist that drifts from the code is worse than no checklist: it says "done"
about names nobody can find. These tests read `plans/ai_cascade_proxy_specification.md`
and fail when a checked box points at a test that is not in the suite.

`plans/` is deliberately not committed (it lives in `.git/info/exclude`), so the
whole module skips where the document is absent — a deployment checkout has no
spec to check, and that is not a failure.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "plans" / "ai_cascade_proxy_specification.md"

CHECKED = re.compile(r"^- \[x\] \*\*Test [^*]+\*\*", re.MULTILINE)
UNCHECKED = re.compile(r"^- \[ \] \*\*Test ", re.MULTILINE)
NAMED_TEST = re.compile(r"`(test_[a-z0-9_]+)`")


def spec_text() -> str:
    if not SPEC.is_file():
        pytest.skip(f"no specification document at {SPEC}")
    return SPEC.read_text(encoding="utf-8")


def names_in_the_spec() -> list[str]:
    names: list[str] = []
    for line in spec_text().splitlines():
        if line.startswith(("- [x] **Test", "- [ ] **Test")):
            names.extend(NAMED_TEST.findall(line))
    return names


def names_in_the_suite() -> set[str]:
    found: set[str] = set()
    for path in (ROOT / "tests").rglob("test_*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("test_"):
                    found.add(node.name)
    return found


def test_every_checked_box_names_a_test_that_exists():
    missing = sorted(set(names_in_the_spec()) - names_in_the_suite())

    assert not missing, f"the spec checks boxes for tests that do not exist: {missing}"


def test_no_box_is_left_unchecked():
    """An unchecked box is either unfinished work or a stale line; both are debt."""
    unchecked = UNCHECKED.findall(spec_text())

    assert not unchecked, f"{len(unchecked)} unchecked test(s) remain in the checklist"


def test_the_checklist_was_not_emptied():
    """Guard against a green suite over a document with no tests in it."""
    assert len(CHECKED.findall(spec_text())) >= 40


def test_the_parser_actually_finds_the_document():
    """A skip must mean 'no document', never 'the regex stopped matching'."""
    if not SPEC.is_file():
        pytest.skip("no specification document")

    assert "Phase 1" in spec_text()
