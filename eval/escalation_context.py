"""Escalation-context experiment (Phase 9).

One question: *does telling the next level why the previous answer was rejected
actually help?*

Production always sends the rejected candidate back with a short note
(`src.cascade_engine.failure_context`). This script replays the same escalations
twice over a labelled dataset, changing nothing but that context:

* ``clean`` — the original prompt alone.
* ``failure_context`` — the original prompt, the rejected candidate, and the
  production note verbatim.

Both arms call the same model level with the same parameters, and both are
judged by the same evaluator at the same threshold, so the only difference
between them is the context. The text of the note is *imported* from the engine,
not copied: an experiment measuring a stale copy of the production prompt would
answer the wrong question.

Nothing here changes proxy behaviour. This is offline tooling.

Usage::

    # What would be sent, without spending a single call
    python -m eval.escalation_context --dataset eval/datasets/synthetic_example.jsonl --dry-run

    # The real comparison, over the rows that escalate
    python -m eval.escalation_context --dataset eval/datasets/my_rows.jsonl \\
        --scores .cache/scores.json

Rows whose candidate the evaluator already approves never escalate and are
excluded: they have no second turn to compare.

Reading the report: ``approval`` is the share of second-level answers the same
evaluator would have accepted. A higher ``approval`` for ``failure_context`` at a
similar cost means the note earns its tokens; a ``distinct`` count near zero
means the second level mostly repeated the rejected text either way.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace

from eval.calibrate import DatasetError, DatasetRow, call_scorer, load_dataset
from src.cascade_config import (
    CascadeConfig,
    CascadeConfigError,
    CascadeLevel,
    load_cascade_config_from_env,
)
from src.cascade_engine import failure_context
from src.jev_evaluator import EvaluationResult
from src.telemetry import estimate_cost

__all__ = [
    "CONTEXT_MODES",
    "DEFAULT_LEVEL_NUMBER",
    "EscalationAnalysis",
    "ExperimentError",
    "ModeOutcome",
    "build_escalation_messages",
    "default_threshold",
    "main",
    "render_report",
    "run_experiment",
]

#: The two arms. Order is the order they are reported in.
CONTEXT_MODES: tuple[str, ...] = ("clean", "failure_context")

#: Escalation goes from level 1 to level 2, so that is where the comparison lives.
DEFAULT_LEVEL_NUMBER = 2

Completer = Callable[..., Awaitable[dict]]
#: ``(prompt, response)`` or ``(prompt, response, system)``; see ``call_scorer``.
Scorer = Callable[..., Awaitable[float | None]]


class ExperimentError(RuntimeError):
    """The experiment cannot answer its question with the data it was given."""


def build_escalation_messages(
    prompt: str,
    candidate: str,
    mode: str,
    *,
    evaluation: EvaluationResult | None = None,
) -> list[dict[str, str]]:
    """The messages the second level receives, per arm.

    ``failure_context`` needs the evaluation that caused the rejection: without
    it the note could only be invented, and an invented reason would make the
    two arms incomparable.
    """
    if mode not in CONTEXT_MODES:
        raise ValueError(f"unknown context mode {mode!r}; expected one of: {', '.join(CONTEXT_MODES)}")
    if mode == "clean":
        return [{"role": "user", "content": prompt}]
    if evaluation is None:
        raise ValueError("failure_context mode requires the evaluation that rejected the candidate")
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": candidate},
        {"role": "system", "content": failure_context(evaluation)},
    ]


@dataclass
class ModeOutcome:
    """What one arm produced, over the same set of escalated rows."""

    mode: str
    #: Rows this arm tried to answer.
    attempted: int = 0
    #: Rows where the provider returned a usable answer.
    answered: int = 0
    #: Rows whose new answer the evaluator approved.
    approved: int = 0
    #: Rows whose new answer differs from the rejected candidate.
    distinct: int = 0
    cost_usd: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def approval_rate(self) -> float | None:
        """Share of answers this evaluator would accept. ``None`` when nothing answered."""
        return self.approved / self.answered if self.answered else None

    @property
    def distinct_rate(self) -> float | None:
        """Share of answers that are not a copy of the rejected candidate."""
        return self.distinct / self.answered if self.answered else None


@dataclass
class EscalationAnalysis:
    """One run of both arms over one dataset."""

    threshold: float
    rows: int
    escalated_rows: int
    level: str
    outcomes: dict[str, ModeOutcome]
    #: The escalated rows and the score that rejected them, for reporting.
    escalations: list[tuple[DatasetRow, float | None]] = field(default_factory=list)

    @property
    def total_cost_usd(self) -> float:
        return sum(outcome.cost_usd for outcome in self.outcomes.values())


def _answer_text(payload: dict) -> str | None:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        )
    return None


def _normalise(text: str) -> str:
    return " ".join(text.split()).casefold()


async def run_experiment(
    rows: Sequence[DatasetRow],
    *,
    threshold: float,
    completer: Completer,
    scorer: Scorer,
    config: CascadeConfig,
    level_number: int = DEFAULT_LEVEL_NUMBER,
) -> EscalationAnalysis:
    """Replay every escalating row through both arms.

    ``scorer`` decides both which rows escalate and whether a new answer is
    approved; ``completer`` is the only thing that spends provider money.
    """
    level = config.level_for_number(level_number)
    escalations: list[tuple[DatasetRow, float | None, EvaluationResult]] = []
    for row in rows:
        score = await call_scorer(scorer, row)
        # A missing score is treated exactly like the default failure policy:
        # escalate. The row still has a second turn worth comparing.
        if score is not None and score >= threshold:
            continue
        evaluation = EvaluationResult(approved=False, score=score, status="rejected")
        escalations.append((row, score, evaluation))

    outcomes = {mode: ModeOutcome(mode=mode) for mode in CONTEXT_MODES}
    for mode in CONTEXT_MODES:
        outcome = outcomes[mode]
        for row, _, evaluation in escalations:
            outcome.attempted += 1
            messages = build_escalation_messages(
                row.prompt, row.candidate_response, mode, evaluation=evaluation
            )
            try:
                payload = await completer(level=level, messages=messages)
            except Exception as exc:  # noqa: BLE001 - recorded, never swallowed silently
                outcome.errors.append(f"{row.id}: {type(exc).__name__}: {exc}")
                continue
            answer = _answer_text(payload)
            if answer is None:
                outcome.errors.append(f"{row.id}: no completion text in the response")
                continue
            outcome.answered += 1
            usage = payload.get("usage")
            outcome.cost_usd += estimate_cost(level, usage if isinstance(usage, dict) else {})
            if _normalise(answer) != _normalise(row.candidate_response):
                outcome.distinct += 1
            score = await call_scorer(scorer, replace(row, candidate_response=answer))
            if score is not None and score >= threshold:
                outcome.approved += 1

    return EscalationAnalysis(
        threshold=threshold,
        rows=len(rows),
        escalated_rows=len(escalations),
        level=level.full_name,
        outcomes=outcomes,
        escalations=[(row, score) for row, score, _ in escalations],
    )


def _rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def render_report(analysis: EscalationAnalysis) -> str:
    lines = [
        f"level under test: {analysis.level}",
        f"threshold {analysis.threshold:.2f}    rows: {analysis.rows}    "
        f"escalated rows: {analysis.escalated_rows}",
    ]
    if analysis.escalated_rows == 0:
        lines += [
            "",
            "no row would escalate: every candidate is already approved at this threshold,",
            "so there is no second turn to compare. Nothing was called and nothing is reported.",
        ]
        return "\n".join(lines)

    lines += [
        "",
        f"{'mode':<17}{'answered':>9}{'approved':>10}{'approval':>10}"
        f"{'distinct':>10}{'cost_usd':>11}",
    ]
    for mode in CONTEXT_MODES:
        outcome = analysis.outcomes[mode]
        lines.append(
            f"{mode:<17}{outcome.answered:>9}{outcome.approved:>10}"
            f"{_rate(outcome.approval_rate):>10}{_rate(outcome.distinct_rate):>10}"
            f"{outcome.cost_usd:>11.6f}"
        )
    for mode in CONTEXT_MODES:
        for error in analysis.outcomes[mode].errors:
            lines.append(f"  {mode}: {error}")

    lines += [
        "",
        f"total cost of the experiment: {analysis.total_cost_usd:.6f} USD",
        "approval is measured by the same evaluator and threshold on the second-level answer;",
        "a difference smaller than the dataset can resolve is not evidence either way.",
    ]
    return "\n".join(lines)


def default_threshold(config: CascadeConfig) -> float:
    """Calibrate the evaluator at the threshold level 1 uses: that is what escalates."""
    threshold = config.threshold_for_level(1)
    if threshold is None:  # pragma: no cover - level 1 always defines one
        raise ExperimentError("level 1 does not define a jev_threshold")
    return threshold


def _real_completer():
    from src.provider import ModelProvider
    from src.settings import Settings

    provider = ModelProvider(Settings.from_env())

    async def completer(*, level: CascadeLevel, messages: list[dict]) -> dict:
        return await provider.complete(
            level.full_name,
            messages,
            timeout_ms=level.params.timeout_ms,
            **level.provider_params,
        )

    return completer


def _real_scorer():
    from src.jev_evaluator import JevEvaluator
    from src.settings import Settings

    evaluator = JevEvaluator(Settings.from_env())

    async def scorer(prompt: str, response: str, system: str = "") -> float | None:
        return await evaluator.score(prompt, response, system=system)

    return scorer


def _load_config(path: str | None) -> CascadeConfig:
    if path is None:
        return load_cascade_config_from_env()
    from src.cascade_config import load_cascade_config

    return load_cascade_config(path)


def main(
    argv: Sequence[str] | None = None,
    *,
    config: CascadeConfig | None = None,
    completer: Completer | None = None,
    scorer: Scorer | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m eval.escalation_context",
        description="Compare escalation with a failure note against escalation with a clean prompt.",
    )
    parser.add_argument("--dataset", required=True, help="JSONL dataset of labelled rows")
    parser.add_argument("--cascade-config", default=None,
                        help="cascade YAML (default: CASCADE_CONFIG_PATH or config/cascade.yaml)")
    parser.add_argument("--level", type=int, default=DEFAULT_LEVEL_NUMBER,
                        help=f"level that answers the escalation (default {DEFAULT_LEVEL_NUMBER})")
    parser.add_argument("--threshold", type=float, default=None,
                        help="approval threshold for both arms (default: the level-1 threshold)")
    parser.add_argument("--limit", type=int, default=None,
                        help="use only the first N rows (a quick, partial answer)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the two requests per row without calling anything")
    args = parser.parse_args(argv)

    try:
        rows = load_dataset(args.dataset)
    except DatasetError as exc:
        print(f"dataset error: {exc}", file=sys.stderr)
        return 2
    if args.limit is not None:
        rows = rows[: args.limit]

    if config is None:
        try:
            config = _load_config(args.cascade_config)
        except CascadeConfigError as exc:
            print(f"cascade configuration error: {exc}", file=sys.stderr)
            return 4

    if args.dry_run:
        print(_render_dry_run(rows, config, args.level))
        return 0

    threshold = args.threshold if args.threshold is not None else default_threshold(config)
    analysis = asyncio.run(run_experiment(
        rows,
        threshold=threshold,
        completer=completer or _real_completer(),
        scorer=scorer or _real_scorer(),
        config=config,
        level_number=args.level,
    ))
    print(render_report(analysis))
    return 0


def _render_dry_run(rows: Sequence[DatasetRow], config: CascadeConfig, level_number: int) -> str:
    """Show exactly what each arm would send, for the first row, and count the rest."""
    level = config.level_for_number(level_number)
    example = rows[0] if rows else None
    lines = [
        f"level under test: {level.full_name}",
        f"rows: {len(rows)} (dry run: which rows escalate depends on the evaluator)",
        f"requests if every row escalated: {len(rows)} per arm, "
        f"{len(rows) * len(CONTEXT_MODES)} in total",
    ]
    if example is None:
        return "\n".join(lines + ["", "the dataset is empty: nothing to send"])

    for mode in CONTEXT_MODES:
        messages = build_escalation_messages(
            example.prompt,
            example.candidate_response,
            mode,
            evaluation=EvaluationResult(approved=False, score=None, status="unavailable"),
        )
        lines += ["", f"--- {mode} (row {example.id}) ---"]
        for message in messages:
            lines.append(f"[{message['role']}] {message['content']}")
    lines += ["", "no provider or evaluator call was made."]
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
