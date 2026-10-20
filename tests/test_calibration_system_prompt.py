"""The calibration harness must score the same task the proxy scores.

`load_dataset` accepted no system prompt, so with ``jev.dialect:
typesafe_decisions`` every row was scored with ``system=""`` — a strictly easier
task than production, where an answer that ignores an instruction is a reject.
The threshold calibrated that way is optimistic, and nothing said so.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.calibrate import (
    DatasetRow,
    load_dataset,
    score_dataset,
    _cache_key,
)

SYSTEM = "Answer only with a single JSON object. Never mention competitors."


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def row(system: str | None = None, **overrides) -> dict:
    data = {
        "id": "r1",
        "prompt": "q",
        "candidate_response": "a",
        "human_label": "accept",
        "source": "hand",
        "notes": "",
    }
    if system is not None:
        data["system"] = system
    data.update(overrides)
    return data


# --------------------------------------------------------------------------- #
# Reading the field
# --------------------------------------------------------------------------- #


def test_a_row_without_a_system_prompt_reads_as_an_empty_string(tmp_path: Path):
    rows = load_dataset(write_jsonl(tmp_path / "d.jsonl", [row()]))

    assert rows[0].system == ""


def test_a_row_carries_its_system_prompt(tmp_path: Path):
    rows = load_dataset(write_jsonl(tmp_path / "d.jsonl", [row(SYSTEM)]))

    assert rows[0].system == SYSTEM


def test_a_null_system_prompt_is_treated_as_absent(tmp_path: Path):
    rows = load_dataset(write_jsonl(tmp_path / "d.jsonl", [row().copy() | {"system": None}]))

    assert rows[0].system == ""


def test_a_non_string_system_prompt_is_refused(tmp_path: Path):
    with pytest.raises(Exception) as excinfo:
        load_dataset(write_jsonl(tmp_path / "d.jsonl", [row() | {"system": ["nope"]}]))

    assert "system" in str(excinfo.value)
    assert ":1:" in str(excinfo.value)  # the line number, whatever the path


def test_a_system_prompt_of_only_spaces_is_refused(tmp_path: Path):
    with pytest.raises(Exception) as excinfo:
        load_dataset(write_jsonl(tmp_path / "d.jsonl", [row() | {"system": "   "}]))

    assert "system" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# Scoring with it
# --------------------------------------------------------------------------- #


async def test_the_scorer_receives_the_system_prompt(tmp_path: Path):
    rows = load_dataset(write_jsonl(tmp_path / "d.jsonl", [row(SYSTEM)]))
    seen: list[tuple[str, str, str]] = []

    async def scorer(prompt: str, response: str, system: str = "") -> float | None:
        seen.append((prompt, response, system))
        return 0.9

    await score_dataset(rows, scorer, cache_path=tmp_path / "scores.json")

    assert seen == [("q", "a", SYSTEM)]


async def test_a_scorer_that_ignores_the_system_prompt_still_works(tmp_path: Path):
    """The harness predates the field; a two-argument scorer must not break."""
    rows = load_dataset(write_jsonl(tmp_path / "d.jsonl", [row(SYSTEM)]))

    async def scorer(prompt: str, response: str) -> float | None:
        return 0.9

    scored = await score_dataset(rows, scorer, cache_path=tmp_path / "scores.json")

    assert scored[0].score == 0.9


async def test_the_same_prompt_with_different_systems_scores_separately(tmp_path: Path):
    """Otherwise a cached score would leak across two different tasks."""
    rows = load_dataset(write_jsonl(tmp_path / "d.jsonl", [
        row(SYSTEM, id="with"),
        row(None, id="without"),
    ]))
    calls: list[str] = []

    async def scorer(prompt: str, response: str, system: str = "") -> float | None:
        calls.append(system)
        return 0.9 if system else 0.2

    scored = await score_dataset(rows, scorer, cache_path=tmp_path / "scores.json")

    assert len(calls) == 2
    assert [entry.score for entry in scored] == [0.9, 0.2]


def test_the_cache_key_distinguishes_the_system_prompt():
    assert _cache_key("q", "a", SYSTEM) != _cache_key("q", "a")
    assert _cache_key("q", "a", SYSTEM) == _cache_key("q", "a", SYSTEM)


def test_the_cache_key_without_a_system_matches_the_old_one():
    """Existing score caches stay valid: nothing is re-billed on upgrade."""
    import hashlib

    digest = hashlib.sha256()
    digest.update(b"q")
    digest.update(b"\x00")
    digest.update(b"a")
    assert _cache_key("q", "a") == digest.hexdigest()
