"""JEV calibration harness.

Answers one question with data instead of intuition: *at which approval
threshold does the evaluator actually behave acceptably?*

The harness reads a JSONL dataset of hand-labelled rows (a prompt, one
candidate response, and whether a human would accept it), scores every row with
the real JEV endpoint, and reports false-accept rate, false-reject rate,
rejection precision/recall and a threshold sweep. Nothing is invented: without
a labelled dataset every number is undefined, and the script says so instead of
guessing.

Usage::

    python -m eval.calibrate --dataset eval/datasets/synthetic_example.jsonl
    python -m eval.calibrate --dataset my_labeled_rows.jsonl --scores .cache/scores.json
    python -m eval.calibrate --dataset my_labeled_rows.jsonl --sweep-only 0.7 0.8 0.9

The JEV endpoint and key come from ``JEV_API_URL`` / ``JEV_API_KEY`` (the same
variables the proxy uses). Scores are cached on disk, keyed by the exact
``(prompt, response)`` pair, so re-tuning thresholds costs no extra JEV calls.

Metric definitions (threshold ``t``; a row is *approved* when ``score >= t``):

* ``false_accepts`` — human said reject, JEV approved. The costly error: a bad
  answer is returned to the caller.
* ``false_rejects`` — human said accept, JEV rejected. The cheap error: the
  request escalates to a better (more expensive) model.
* ``false_accept_rate`` = false_accepts / human rejects.
* ``false_reject_rate`` = false_rejects / human accepts.
* ``precision_reject`` = correctly rejected / all rejected by JEV.
* ``recall_reject`` = correctly rejected / all human rejects.

Rates are ``None`` — never ``0.0`` — when their denominator is zero, because an
undefined rate is not a good score.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Iterable, Sequence

logger = logging.getLogger("ai_cascade_proxy.calibrate")

__all__ = [
    "call_scorer",
    "DatasetError",
    "DatasetRow",
    "ThresholdMetrics",
    "ScoredRow",
    "SweepPoint",
    "ThresholdSuggestion",
    "compute_metrics",
    "load_dataset",
    "score_dataset",
    "suggest_threshold",
    "sweep",
    "render_report",
    "SWEEP_THRESHOLDS",
]

HUMAN_LABELS = ("accept", "reject")
REQUIRED_FIELDS = ("id", "prompt", "candidate_response", "human_label", "source", "notes")
SWEEP_THRESHOLDS: tuple[float, ...] = (
    0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
)

#: ``(prompt, response)`` or ``(prompt, response, system)``. The harness calls the
#: three-argument form and falls back for scorers written before the field existed.
ScoreFn = Callable[..., Awaitable[float | None]]


class DatasetError(ValueError):
    """Raised when a dataset row is unusable; calibration must never silently skip rows."""


@dataclass(frozen=True)
class DatasetRow:
    id: str
    prompt: str
    candidate_response: str
    human_label: str
    source: str
    notes: str = ""
    #: The client's system prompt for this row. Empty means the request had none.
    #: Without it, a `typesafe_decisions` score measures an easier task than the
    #: one production scores, and the calibrated threshold comes out optimistic.
    system: str = ""

    @property
    def is_human_reject(self) -> bool:
        return self.human_label == "reject"


@dataclass(frozen=True)
class ScoredRow:
    row: DatasetRow
    score: float | None


@dataclass(frozen=True)
class ThresholdMetrics:
    threshold: float
    total: int
    human_rejects: int
    human_accepts: int
    true_rejects: int
    true_accepts: int
    false_accepts: int
    false_rejects: int
    unavailable: int

    @property
    def correct_rejects(self) -> int:
        return self.true_rejects

    @property
    def rejected_by_jev(self) -> int:
        return self.true_rejects + self.false_rejects

    @property
    def scored(self) -> int:
        return self.total - self.unavailable

    @property
    def false_accept_rate(self) -> float | None:
        return None if not self.human_rejects else self.false_accepts / self.human_rejects

    @property
    def false_reject_rate(self) -> float | None:
        return None if not self.human_accepts else self.false_rejects / self.human_accepts

    @property
    def precision_reject(self) -> float | None:
        return None if not self.rejected_by_jev else self.true_rejects / self.rejected_by_jev

    @property
    def recall_reject(self) -> float | None:
        return None if not self.human_rejects else self.true_rejects / self.human_rejects


@dataclass(frozen=True)
class SweepPoint:
    threshold: float
    true_rejects: int
    false_accepts: int
    false_rejects: int
    human_rejects: int
    scored: int = 0
    unavailable: int = 0

    @property
    def false_decisions(self) -> int:
        return self.false_accepts + self.false_rejects


@dataclass(frozen=True)
class ThresholdSuggestion:
    threshold: float
    false_decisions: int
    false_accepts: int
    false_rejects: int
    tied_thresholds: tuple[float, ...] = field(default_factory=tuple)


def _rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def load_dataset(path: str | Path) -> list[DatasetRow]:
    """Load and validate a JSONL dataset. Raises ``DatasetError`` on the first bad row."""
    path = Path(path)
    rows: list[DatasetRow] = []
    seen: set[str] = set()
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as exc:
        # A missing or unreadable dataset is a dataset problem, not a crash: the
        # callers turn DatasetError into "fix your dataset" and a distinct exit code.
        raise DatasetError(f"{path}: cannot be read ({exc.strerror or exc})") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            where = f"{path}:{line_number}"
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{where}: invalid JSON ({exc.msg})") from exc
            if not isinstance(raw, dict):
                raise DatasetError(f"{where}: each line must be a JSON object")
            missing = [name for name in REQUIRED_FIELDS if name not in raw]
            if missing:
                raise DatasetError(f"{where}: missing required field(s): {', '.join(missing)}")
            for name in ("id", "prompt", "candidate_response", "source"):
                value = raw[name]
                if not isinstance(value, str) or not value.strip():
                    raise DatasetError(f"{where}: {name} must be a non-empty string")
            if raw["human_label"] not in HUMAN_LABELS:
                raise DatasetError(
                    f"{where}: human_label must be one of {', '.join(HUMAN_LABELS)}, "
                    f"got {raw['human_label']!r}"
                )
            system = raw.get("system", "")
            if system is None:
                system = ""
            if not isinstance(system, str):
                raise DatasetError(f"{where}: system must be a string")
            if system and not system.strip():
                raise DatasetError(f"{where}: system must not be only whitespace")
            notes = raw["notes"]
            if notes is None:
                notes = ""
            if not isinstance(notes, str):
                raise DatasetError(f"{where}: notes must be a string")
            if raw["id"] in seen:
                raise DatasetError(f"{where}: duplicate id {raw['id']!r}")
            seen.add(raw["id"])
            rows.append(
                DatasetRow(
                    id=raw["id"],
                    prompt=raw["prompt"],
                    candidate_response=raw["candidate_response"],
                    human_label=raw["human_label"],
                    source=raw["source"],
                    notes=notes,
                    system=system,
                )
            )
    if not rows:
        raise DatasetError(f"{path}: dataset is empty; calibration needs at least one row")
    return rows


def _cache_key(prompt: str, response: str, system: str = "") -> str:
    """Digest of exactly what JEV is asked to judge.

    The system prompt is part of the question, so it is part of the key: two rows
    with the same prompt and answer under different instructions are two tasks. An
    empty system keeps the original digest, so existing score caches stay valid.
    """
    digest = hashlib.sha256()
    if system:
        digest.update(system.encode("utf-8"))
        digest.update(b"\x01")
    digest.update(prompt.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(response.encode("utf-8"))
    return digest.hexdigest()


async def call_scorer(scorer: ScoreFn, row: DatasetRow) -> float | None:
    """Give the scorer the system prompt, tolerating scorers that do not take it."""
    try:
        return await scorer(row.prompt, row.candidate_response, row.system)
    except TypeError as exc:
        if row.system:
            # A two-argument scorer cannot see the system prompt; that makes this
            # row easier than production, so say so instead of hiding it.
            logger.warning(
                "Scorer does not accept the system prompt; row %s was scored without it. "
                "Calibrated thresholds will be optimistic for traffic that relies on "
                "system instructions.", row.id,
            )
        if "system" in str(exc) or "positional argument" in str(exc):
            return await scorer(row.prompt, row.candidate_response)
        raise


def _load_cache(path: Path | None) -> dict[str, float | None]:
    if path is None or not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(key): value for key, value in raw.items()
            if value is None or isinstance(value, (int, float)) and not isinstance(value, bool)}


def _save_cache(path: Path | None, cache: dict[str, float | None]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")


async def score_dataset(
    rows: Sequence[DatasetRow],
    scorer: ScoreFn,
    *,
    cache_path: str | Path | None = None,
) -> list[ScoredRow]:
    """Score every row with ``scorer``, reusing cached scores when available."""
    cache_file = Path(cache_path) if cache_path is not None else None
    cache = _load_cache(cache_file)
    scored: list[ScoredRow] = []
    dirty = False
    for row in rows:
        key = _cache_key(row.prompt, row.candidate_response, row.system)
        if key in cache:
            scored.append(ScoredRow(row=row, score=cache[key]))
            continue
        score = await call_scorer(scorer, row)
        cache[key] = score
        dirty = True
        scored.append(ScoredRow(row=row, score=score))
    if dirty:
        _save_cache(cache_file, cache)
    return scored


def compute_metrics(rows: Iterable[ScoredRow], *, threshold: float) -> ThresholdMetrics:
    """Confusion counts for one threshold. ``score >= threshold`` means approved."""
    human_rejects = human_accepts = 0
    true_rejects = true_accepts = false_accepts = false_rejects = 0
    unavailable = total = 0
    for item in rows:
        total += 1
        if item.row.is_human_reject:
            human_rejects += 1
        else:
            human_accepts += 1
        if item.score is None:
            unavailable += 1
            continue
        approved = item.score >= threshold
        if item.row.is_human_reject:
            if approved:
                false_accepts += 1
            else:
                true_rejects += 1
        else:
            if approved:
                true_accepts += 1
            else:
                false_rejects += 1
    return ThresholdMetrics(
        threshold=threshold,
        total=total,
        human_rejects=human_rejects,
        human_accepts=human_accepts,
        true_rejects=true_rejects,
        true_accepts=true_accepts,
        false_accepts=false_accepts,
        false_rejects=false_rejects,
        unavailable=unavailable,
    )


def sweep(rows: Iterable[ScoredRow],
          thresholds: Sequence[float] = SWEEP_THRESHOLDS) -> list[SweepPoint]:
    """Evaluate the same scored rows across the threshold grid (0.50–0.95 by default)."""
    materialised = list(rows)
    points: list[SweepPoint] = []
    for threshold in thresholds:
        metrics = compute_metrics(materialised, threshold=threshold)
        points.append(
            SweepPoint(
                threshold=threshold,
                true_rejects=metrics.true_rejects,
                false_accepts=metrics.false_accepts,
                false_rejects=metrics.false_rejects,
                human_rejects=metrics.human_rejects,
                scored=metrics.scored,
                unavailable=metrics.unavailable,
            )
        )
    return points


def suggest_threshold(points: Sequence[SweepPoint]) -> ThresholdSuggestion:
    """Pick the grid point with the fewest false decisions.

    Ties keep the lowest threshold: approving earlier is the whole point of a
    cascade, so an equally accurate gate that escalates less is the better gate.

    Refuses to answer when no row was scored: with every score unavailable the
    sweep is all zeros, which would otherwise be reported as a flawless 0.50
    gate. An unmeasured gate is not a good gate.
    """
    if not points:
        raise ValueError("Cannot suggest a threshold from an empty sweep")
    if not any(point.scored for point in points):
        raise DatasetError(
            "Cannot suggest a threshold: no row was scored by JEV. Configure "
            "JEV_API_URL and JEV_API_KEY and check that the endpoint returns a numeric score."
        )
    best = min(points, key=lambda point: (point.false_decisions, point.threshold))
    tied = tuple(point.threshold for point in points
                 if point.false_decisions == best.false_decisions)
    return ThresholdSuggestion(
        threshold=best.threshold,
        false_decisions=best.false_decisions,
        false_accepts=best.false_accepts,
        false_rejects=best.false_rejects,
        tied_thresholds=tied,
    )


def render_report(
    metrics: ThresholdMetrics,
    points: Sequence[SweepPoint],
    suggestion: ThresholdSuggestion,
) -> str:
    lines = [
        f"rows: {metrics.total} ({metrics.human_rejects} human reject, "
        f"{metrics.human_accepts} human accept, {metrics.unavailable} unavailable)",
        "",
        f"threshold {metrics.threshold:.2f}:",
        f"  false accepts    {metrics.false_accepts:>4}   rate {_rate(metrics.false_accept_rate)}",
        f"  false rejects    {metrics.false_rejects:>4}   rate {_rate(metrics.false_reject_rate)}",
        f"  precision(reject)                       {_rate(metrics.precision_reject)}",
        f"  recall(reject)                          {_rate(metrics.recall_reject)}",
        "",
        "sweep   true_reject  false_accept  false_reject",
    ]
    for point in points:
        lines.append(
            f"  {point.threshold:.2f}   {point.true_rejects:>11}   "
            f"{point.false_accepts:>12}   {point.false_rejects:>12}"
        )
    lines += [
        "",
        f"suggested threshold: {suggestion.threshold:.2f} "
        f"({suggestion.false_decisions} false decisions; "
        f"false accepts {suggestion.false_accepts}, false rejects {suggestion.false_rejects})",
    ]
    if len(suggestion.tied_thresholds) > 1:
        lines.append(
            "  tied with: " + ", ".join(f"{value:.2f}" for value in suggestion.tied_thresholds)
        )
    lines += [
        "",
        "Negative numbers mean the cascade is worse than the configuration assumes;",
        "these metrics describe the evaluator on this dataset only, not a guarantee.",
    ]
    return "\n".join(lines)


def _real_scorer(settings) -> ScoreFn:
    """Wrap the production evaluator; unavailable evaluations become ``None``, not 0.0."""
    from src.jev_evaluator import JevEvaluator

    evaluator = JevEvaluator(settings)

    async def scorer(prompt: str, response: str, system: str = "") -> float | None:
        return await evaluator.score(prompt, response, system=system)

    return scorer


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m eval.calibrate",
        description="Calibrate JEV approval thresholds against a hand-labelled dataset.",
    )
    parser.add_argument("--dataset", required=True, help="JSONL dataset of labelled rows")
    parser.add_argument(
        "--scores",
        default=None,
        help="JSON cache of prompt/response scores, re-used across runs so a re-tune "
             "costs no extra JEV calls",
    )
    parser.add_argument("--threshold", type=float, default=None,
                        help="Report this threshold instead of the suggested one")
    parser.add_argument("--sweep-only", nargs="+", type=float, default=None,
                        help="Override the sweep grid, e.g. --sweep-only 0.7 0.8 0.9")
    args = parser.parse_args(argv)

    from src.settings import Settings

    try:
        rows = load_dataset(args.dataset)
    except DatasetError as exc:
        print(f"dataset error: {exc}", file=sys.stderr)
        return 2

    scorer = _real_scorer(Settings.from_env())
    scored = asyncio.run(score_dataset(rows, scorer, cache_path=args.scores))
    scored_rows = sum(1 for item in scored if item.score is not None)
    if scored_rows == 0:
        print(
            f"calibration failed: not one row of {len(rows)} could be scored by JEV. "
            "Configure JEV_API_URL and JEV_API_KEY, then re-run.",
            file=sys.stderr,
        )
        return 3

    thresholds = args.sweep_only or SWEEP_THRESHOLDS
    points = sweep(scored, thresholds)
    suggestion = suggest_threshold(points)
    threshold = args.threshold if args.threshold is not None else suggestion.threshold
    metrics = compute_metrics(scored, threshold=threshold)
    print(render_report(metrics, points, suggestion))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
