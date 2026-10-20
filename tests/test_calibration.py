"""Phase 1: metrics of the JEV calibration harness, from hand-computed cases."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from eval.calibrate import (
    DatasetError,
    ScoredRow,
    compute_metrics,
    load_dataset,
    main,
    score_dataset,
    suggest_threshold,
    sweep,
)
from eval.calibrate import DatasetRow
from src.settings import Settings

SYNTHETIC = Path(__file__).resolve().parents[1] / "eval" / "datasets" / "synthetic_example.jsonl"


def scored(*triples: tuple[str, float]) -> list[ScoredRow]:
    """Build scored rows from (human_label, jev_score) pairs; order becomes the id."""
    return [
        ScoredRow(
            row=DatasetRow(
                id=f"row-{index}",
                prompt="prompt",
                candidate_response="response",
                human_label=label,
                source="test",
                notes="",
            ),
            score=score,
        )
        for index, (label, score) in enumerate(triples)
    ]


def test_metrics_match_hand_computed_case():
    # Two human rejects (0.9 accepted -> false accept, 0.2 rejected -> correct),
    # two human accepts (0.85 accepted -> correct, 0.6 rejected -> false reject).
    rows = scored(("reject", 0.9), ("reject", 0.2), ("accept", 0.85), ("accept", 0.6))

    metrics = compute_metrics(rows, threshold=0.8)

    assert metrics.threshold == 0.8
    assert metrics.total == 4
    assert metrics.human_rejects == 2
    assert metrics.human_accepts == 2
    assert metrics.true_rejects == 1
    assert metrics.true_accepts == 1
    assert metrics.false_accepts == 1
    assert metrics.false_rejects == 1
    assert metrics.correct_rejects == 1
    assert metrics.false_accept_rate == pytest.approx(0.5)
    assert metrics.false_reject_rate == pytest.approx(0.5)
    assert metrics.precision_reject == pytest.approx(0.5)
    assert metrics.recall_reject == pytest.approx(0.5)


def test_metrics_with_perfect_evaluator():
    rows = scored(("reject", 0.10), ("reject", 0.40), ("accept", 0.90), ("accept", 0.99))

    metrics = compute_metrics(rows, threshold=0.8)

    assert metrics.false_accepts == 0
    assert metrics.false_rejects == 0
    assert metrics.precision_reject == pytest.approx(1.0)
    assert metrics.recall_reject == pytest.approx(1.0)


def test_threshold_boundary_is_inclusive():
    rows = scored(("reject", 0.8), ("accept", 0.8))

    metrics = compute_metrics(rows, threshold=0.8)

    # score >= threshold approves, so both rows are approved.
    assert metrics.false_accepts == 1
    assert metrics.false_rejects == 0


def test_metrics_without_rejects_are_undefined_not_invented():
    rows = scored(("accept", 0.9), ("accept", 0.55))

    metrics = compute_metrics(rows, threshold=0.8)

    assert metrics.true_rejects == 0
    assert metrics.false_accept_rate is None
    assert metrics.recall_reject is None
    # Nothing rejected by JEV was a human reject, so the rejection precision is 0.
    assert metrics.precision_reject == pytest.approx(0.0)


def test_metrics_without_any_rejection_are_undefined():
    rows = scored(("reject", 0.95), ("accept", 0.90))

    metrics = compute_metrics(rows, threshold=0.80)

    assert metrics.rejected_by_jev == 0
    assert metrics.precision_reject is None


def test_sweep_covers_the_documented_range():
    rows = scored(("reject", 0.9), ("reject", 0.2), ("accept", 0.85), ("accept", 0.6))

    points = sweep(rows)

    assert [point.threshold for point in points] == [
        0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95
    ]


def test_sweep_is_monotonic_in_false_rejects_and_false_accepts():
    rows = scored(
        ("reject", 0.52), ("reject", 0.61), ("reject", 0.88), ("reject", 0.94),
        ("accept", 0.55), ("accept", 0.66), ("accept", 0.72), ("accept", 0.97),
    )

    points = sweep(rows)

    false_accepts = [point.false_accepts for point in points]
    false_rejects = [point.false_rejects for point in points]
    assert false_accepts == sorted(false_accepts, reverse=True)
    assert false_rejects == sorted(false_rejects)


def test_suggest_threshold_minimises_false_decisions():
    # 0.80 is the only grid point that misclassifies nothing:
    # 0.75 would approve the reject at 0.79, 0.85 would reject the accept at 0.82.
    rows = scored(
        ("reject", 0.20), ("reject", 0.79), ("accept", 0.82),
        ("accept", 0.90), ("accept", 0.95),
    )

    suggestion = suggest_threshold(sweep(rows))

    assert suggestion.threshold == 0.80
    assert suggestion.false_decisions == 0


def test_suggest_threshold_breaks_ties_towards_the_lowest_threshold():
    # Approving earlier is the point of a cascade, so a tie keeps the cheapest gate.
    rows = scored(("reject", 0.20), ("reject", 0.60), ("accept", 0.90))

    suggestion = suggest_threshold(sweep(rows))

    # Every grid point from 0.65 to 0.90 is equally correct; 0.65 wins the tie.
    assert suggestion.false_decisions == 0
    assert suggestion.threshold == 0.65


def test_load_dataset_reads_jsonl(tmp_path: Path):
    path = tmp_path / "rows.jsonl"
    path.write_text(
        "\n".join([
            json.dumps({"id": "a", "prompt": "p1", "candidate_response": "r1",
                        "human_label": "accept", "source": "hand", "notes": ""}),
            json.dumps({"id": "b", "prompt": "p2", "candidate_response": "r2",
                        "human_label": "reject", "source": "hand", "notes": "wrong math"}),
        ]),
        encoding="utf-8",
    )

    rows = load_dataset(path)

    assert [row.id for row in rows] == ["a", "b"]
    assert rows[1].human_label == "reject"
    assert rows[1].notes == "wrong math"


def test_load_dataset_rejects_invalid_label(tmp_path: Path):
    path = tmp_path / "rows.jsonl"
    path.write_text(
        json.dumps({"id": "a", "prompt": "p", "candidate_response": "r",
                    "human_label": "maybe", "source": "hand", "notes": ""}),
        encoding="utf-8",
    )

    with pytest.raises(DatasetError, match="human_label"):
        load_dataset(path)


def test_load_dataset_rejects_missing_fields(tmp_path: Path):
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps({"id": "a", "prompt": "p"}), encoding="utf-8")

    with pytest.raises(DatasetError, match="candidate_response"):
        load_dataset(path)


def test_load_dataset_rejects_duplicate_ids(tmp_path: Path):
    row = {"id": "dup", "prompt": "p", "candidate_response": "r",
           "human_label": "accept", "source": "hand", "notes": ""}
    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join([json.dumps(row), json.dumps(row)]), encoding="utf-8")

    with pytest.raises(DatasetError, match="duplicate id"):
        load_dataset(path)


def test_load_dataset_rejects_blank_required_text(tmp_path: Path):
    path = tmp_path / "rows.jsonl"
    path.write_text(
        json.dumps({"id": "a", "prompt": "  ", "candidate_response": "r",
                    "human_label": "accept", "source": "hand", "notes": ""}),
        encoding="utf-8",
    )

    with pytest.raises(DatasetError, match="prompt"):
        load_dataset(path)


@pytest.mark.asyncio
async def test_score_dataset_uses_the_injected_client_and_caches(tmp_path: Path):
    rows = load_dataset(SYNTHETIC)
    calls = []

    async def scorer(prompt: str, response: str) -> float | None:
        calls.append(response)
        return 0.9 if "geometry" in response else 0.1

    cache = tmp_path / "scores.json"

    first = await score_dataset(rows, scorer, cache_path=cache)
    second = await score_dataset(rows, scorer, cache_path=cache)

    assert len(first) == len(rows)
    assert len(calls) == len(rows)          # the cache prevents a second JEV round-trip
    assert [row.score for row in first] == [row.score for row in second]
    assert cache.is_file()


@pytest.mark.asyncio
async def test_score_dataset_records_unavailable_scores_as_none(tmp_path: Path):
    rows = load_dataset(SYNTHETIC)[:2]

    async def scorer(prompt: str, response: str) -> float | None:
        return None

    result = await score_dataset(rows, scorer, cache_path=tmp_path / "scores.json")
    metrics = compute_metrics(result, threshold=0.8)

    assert all(row.score is None for row in result)
    assert metrics.unavailable == 2
    assert metrics.rejected_by_jev == 0


def test_synthetic_dataset_is_small_and_labelled_synthetic():
    rows = load_dataset(SYNTHETIC)

    assert 1 <= len(rows) <= 10
    assert all("synthetic" in row.source for row in rows)
    assert {row.human_label for row in rows} == {"accept", "reject"}


def test_sweep_records_how_many_rows_were_actually_scored():
    rows = scored(("reject", 0.9), ("accept", 0.9))
    rows.append(ScoredRow(row=rows[0].row, score=None))

    points = sweep(rows)

    assert all(point.scored == 2 for point in points)
    assert all(point.unavailable == 1 for point in points)


def test_suggest_threshold_refuses_to_guess_when_nothing_was_scored():
    rows = [ScoredRow(row=scored(("accept", 0.9))[0].row, score=None)]

    with pytest.raises(DatasetError, match="no row was scored"):
        suggest_threshold(sweep(rows))


def test_main_fails_loudly_when_jev_cannot_score_anything(tmp_path: Path, capsys, monkeypatch):
    for name in ("JEV_API_URL", "JEV_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    exit_code = main(["--dataset", str(SYNTHETIC), "--scores", str(tmp_path / "scores.json")])

    captured = capsys.readouterr()
    assert exit_code == 3
    assert "not one row" in captured.err
    assert "suggested threshold" not in captured.out


def test_main_reports_metrics_for_a_fully_scored_dataset(tmp_path: Path, capsys, monkeypatch):
    dataset = tmp_path / "rows.jsonl"
    dataset.write_text(
        "\n".join([
            json.dumps({"id": "a", "prompt": "p1", "candidate_response": "good",
                        "human_label": "accept", "source": "hand", "notes": ""}),
            json.dumps({"id": "b", "prompt": "p2", "candidate_response": "bad",
                        "human_label": "reject", "source": "hand", "notes": ""}),
        ]),
        encoding="utf-8",
    )

    def fake_scorer(settings):
        async def scorer(prompt: str, response: str) -> float | None:
            return 0.95 if response == "good" else 0.05

        return scorer

    monkeypatch.setattr("eval.calibrate._real_scorer", fake_scorer)

    exit_code = main(["--dataset", str(dataset), "--scores", str(tmp_path / "scores.json")])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "suggested threshold" in captured.out
    assert "false accepts" in captured.out


@pytest.mark.asyncio
async def test_jev_evaluator_exposes_a_score_only_call():
    from src.jev_evaluator import JevEvaluator

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"score": 0.42}))
    )
    evaluator = JevEvaluator(
        Settings(jev_api_key="k", jev_api_url="https://jev.test/evaluate"), client=client
    )

    score = await evaluator.score("prompt", "response")
    await client.aclose()

    assert score == pytest.approx(0.42)


@pytest.mark.asyncio
async def test_jev_score_returns_none_instead_of_a_fabricated_number():
    from src.jev_evaluator import JevEvaluator

    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(503)))
    evaluator = JevEvaluator(
        Settings(jev_api_key="k", jev_api_url="https://jev.test/evaluate"), client=client
    )

    score = await evaluator.score("prompt", "response")
    await client.aclose()

    assert score is None
