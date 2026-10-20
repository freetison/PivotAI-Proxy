"""The two JEV wire dialects, isolated from the evaluator that consumes them.

``simple`` is the contract this proxy started with::

    POST {JEV_API_URL}  {"prompt": ..., "response": ...}  ->  {"score": 0.87}

No real JEV endpoint serves it, but tests, fixtures and any in-house evaluator do,
so it stays as the default and keeps working unchanged.

``typesafe_decisions`` is TypeSafe's System One API, which OpenRouter exposes at
``https://openrouter.ai/api/v1/systemone``::

    POST {"model": "jev-1.13",
          "state": {"system": ..., "question": ..., "answer": ...},
          "questions": {"correct": {"type": "noul", "instructions": ...}}}
      -> {"answers": {"correct": {"type": "noul", "noul": 0.93}},
          "usage": {"input_tokens": 275, "output_tokens": 20, "cost": 0.00003}}

Both dialects reduce to one number in [0, 1] or ``None``. That is the whole point:
the per-level threshold, the failure policy and the calibration harness keep their
meaning regardless of which wire format produced the number.

Why ``noul`` and not ``choice``: a ``noul`` answer *is* "probability that this is
true", which is what ``score`` has always meant here. A ``choice`` over three
labels splits its confidence three ways and would silently rescale every
threshold in ``cascade.yaml``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

SIMPLE = "simple"
TYPESAFE_DECISIONS = "typesafe_decisions"
JEV_DIALECTS: tuple[str, ...] = (SIMPLE, TYPESAFE_DECISIONS)
DEFAULT_JEV_DIALECT = SIMPLE

#: Question asked when ``cascade.yaml`` does not define one.
DEFAULT_QUESTION_NAME = "correct"
DEFAULT_QUESTION = "Is the answer correct and complete for the question it answers?"


def build_simple_payload(prompt: str, response: str) -> dict[str, Any]:
    """The legacy two-string contract. Kept byte-identical on purpose."""
    return {"prompt": prompt, "response": response}


def build_system_one_body(*, model: str, system: str, prompt: str, response: str,
                          questions: Mapping[str, str]) -> dict[str, Any]:
    """A System One request: typed questions about a structured state.

    ``state`` is an object rather than a concatenated string so the client's
    system prompt reaches JEV. The legacy dialect concatenated user turns only,
    which let JEV approve an answer that ignored the instructions it was given.
    """
    return {
        "model": model,
        "state": {"system": system, "question": prompt, "answer": response},
        "questions": {
            name: {"type": "noul", "instructions": instructions}
            for name, instructions in questions.items()
        },
    }


def dialect_payload(dialect: str, *, model: str, system: str, prompt: str,
                    response: str, questions: Mapping[str, str]) -> dict[str, Any]:
    """Build the request body for ``dialect``.

    Raises:
        ValueError: the dialect is not one of :data:`JEV_DIALECTS`. Configuration
            validation catches this at startup; reaching it here would mean a
            request silently scored nothing.
    """
    if dialect == SIMPLE:
        return build_simple_payload(prompt, response)
    if dialect == TYPESAFE_DECISIONS:
        return build_system_one_body(
            model=model, system=system, prompt=prompt, response=response,
            questions=questions,
        )
    raise ValueError(
        f"unknown JEV dialect {dialect!r}; expected one of: {', '.join(JEV_DIALECTS)}"
    )


def _usable_score(value: Any) -> float | None:
    """A score must be a real number in [0, 1]. Booleans are not numbers here."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not 0 <= value <= 1:
        return None
    return float(value)


def score_from_simple(data: Any) -> float | None:
    """Extract ``score`` from the legacy response body, or None."""
    if not isinstance(data, Mapping):
        return None
    return _usable_score(data.get("score"))


def score_from_answers(data: Any) -> float | None:
    """Reduce a System One response to one score, or None.

    Every configured question is required to have produced a usable ``noul``
    answer, and the result is the **minimum**: two questions mean two
    requirements, and an answer that fails one of them is not approved. Averaging
    would let a confident "correct" hide a failed "well formatted".
    """
    if not isinstance(data, Mapping):
        return None
    answers = data.get("answers")
    if not isinstance(answers, Mapping) or not answers:
        return None

    scores: list[float] = []
    for answer in answers.values():
        if not isinstance(answer, Mapping):
            return None
        if answer.get("type") != "noul":
            return None
        score = _usable_score(answer.get("noul"))
        if score is None:
            return None
        scores.append(score)
    return min(scores)


def cost_from_answers(data: Any) -> float:
    """OpenRouter's billed cost for the verification, or 0.0 when not reported.

    Never estimated: an unknown cost is reported as zero and stays visible as a
    zero, rather than being filled in with a made-up price.
    """
    if not isinstance(data, Mapping):
        return 0.0
    usage = data.get("usage")
    if not isinstance(usage, Mapping):
        return 0.0
    cost = usage.get("cost")
    if isinstance(cost, bool) or not isinstance(cost, (int, float)):
        return 0.0
    return max(0.0, float(cost))
