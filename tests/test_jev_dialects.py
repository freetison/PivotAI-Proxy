"""Two JEV dialects: the legacy `simple` contract and TypeSafe's System One.

The proxy originally spoke ``{"prompt","response"} -> {"score"}``, which no real
JEV endpoint serves. TypeSafe/OpenRouter answer *typed questions about state*
instead: ``{"model","state","questions"} -> {"answers":{name:{"noul":0.93}}}``.

Both dialects must produce the same thing to the rest of the code — a float in
[0, 1] or ``None`` — so the threshold, the failure policy and the calibration
harness keep their existing meaning. The dialect is configuration, so these
tests pin the exact wire payload of each one.
"""

from __future__ import annotations

import json

import httpx
import httpx2
import pytest

from src.jev_dialects import (
    JEV_DIALECTS,
    SIMPLE,
    TYPESAFE_DECISIONS,
    build_simple_payload,
    build_system_one_body,
    dialect_payload,
    score_from_answers,
    score_from_simple,
)

# --------------------------------------------------------------------------- #
# Dialect names are a closed set: a typo must be caught by configuration, not
# by a request that silently scores nothing.
# --------------------------------------------------------------------------- #


def test_the_dialects_are_a_closed_set():
    assert JEV_DIALECTS == (SIMPLE, TYPESAFE_DECISIONS)


# --------------------------------------------------------------------------- #
# simple: unchanged, byte for byte.
# --------------------------------------------------------------------------- #


def test_the_simple_payload_is_unchanged():
    """Existing deployments and the calibration harness depend on this shape."""
    assert build_simple_payload("the question", "the answer") == {
        "prompt": "the question",
        "response": "the answer",
    }


def test_the_simple_payload_survives_a_json_round_trip():
    payload = build_simple_payload("q", "a")

    assert json.loads(json.dumps(payload)) == {"prompt": "q", "response": "a"}


@pytest.mark.parametrize("score", [0.0, 0.42, 1.0, 1])
def test_a_simple_score_in_range_is_returned_as_a_float(score):
    assert score_from_simple({"score": score}) == float(score)


@pytest.mark.parametrize(
    "body",
    [
        {"score": 1.5},
        {"score": -0.1},
        {"score": 3},
        {"score": "0.9"},
        {"score": True},
        {"score": None},
        {},
        [],
        "not an object",
    ],
)
def test_a_simple_score_that_cannot_be_trusted_is_no_measurement(body):
    assert score_from_simple(body) is None


# --------------------------------------------------------------------------- #
# typesafe_decisions: the payload TypeSafe/OpenRouter actually accepts.
# --------------------------------------------------------------------------- #


def test_the_system_one_body_asks_one_named_noul_question():
    body = build_system_one_body(
        model="jev-1.13",
        system="You are terse.",
        prompt="What is the capital of France?",
        response="Paris.",
        questions={"correct": "Is the answer correct and complete?"},
    )

    assert body == {
        "model": "jev-1.13",
        "state": {
            "system": "You are terse.",
            "question": "What is the capital of France?",
            "answer": "Paris.",
        },
        "questions": {
            "correct": {"type": "noul", "instructions": "Is the answer correct and complete?"},
        },
    }


def test_the_state_keeps_the_system_prompt_that_the_simple_dialect_dropped():
    """`_user_prompt` sends only user turns; JEV would approve answers that
    ignore the client's instructions. The object state carries them."""
    body = build_system_one_body(
        model="jev-1.13", system="Always answer in JSON.", prompt="hi", response="hi there",
        questions={"correct": "Is it correct?"},
    )

    assert body["state"]["system"] == "Always answer in JSON."


def test_an_empty_system_prompt_is_still_sent_as_a_string():
    body = build_system_one_body(
        model="jev-1.13", system="", prompt="hi", response="hello",
        questions={"correct": "Is it correct?"},
    )

    assert body["state"] == {"system": "", "question": "hi", "answer": "hello"}


def test_the_questions_are_emitted_in_configuration_order():
    """Insertion order is the user's order; nothing is sorted behind their back."""
    body = build_system_one_body(
        model="jev-1.13", system="", prompt="p", response="r",
        questions={"first": "one?", "second": "two?"},
    )

    assert list(body["questions"]) == ["first", "second"]


def test_a_noul_answer_becomes_the_score():
    assert score_from_answers({"answers": {"correct": {"type": "noul", "noul": 0.93}}}) == pytest.approx(0.93)


def test_the_score_is_the_lowest_answer_so_every_question_must_pass():
    """Two questions, no averaging: an answer that fails one of them is not approved."""
    data = {"answers": {
        "correct": {"type": "noul", "noul": 0.95},
        "complete": {"type": "noul", "noul": 0.40},
    }}

    assert score_from_answers(data) == pytest.approx(0.40)


def test_extra_response_fields_are_tolerated():
    """OpenRouter adds `id` and `provider`; the SDK's models ignore them."""
    data = {
        "id": "gen-dec-1",
        "provider": "TypeSafe",
        "model": "typesafe/jev-1.13-20260917",
        "answers": {"correct": {"type": "noul", "noul": 0.88}},
        "usage": {"input_tokens": 275, "output_tokens": 20, "cost": 0.00003},
    }

    assert score_from_answers(data) == pytest.approx(0.88)


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"answers": {}},
        {"answers": {"correct": {"type": "noul", "noul": 1.4}}},
        {"answers": {"correct": {"type": "noul", "noul": -0.2}}},
        {"answers": {"correct": {"type": "noul", "noul": "high"}}},
        {"answers": {"correct": {"type": "noul", "noul": True}}},
        {"answers": {"correct": {"type": "choice", "choice": "supported", "confidence": 0.9}}},
        {"answers": "nope"},
        [],
    ],
)
def test_an_answer_that_is_not_a_usable_noul_is_no_measurement(data):
    """Never invent a score: an unparseable answer must not become 0 or 1."""
    assert score_from_answers(data) is None


def test_one_unusable_answer_among_several_is_no_measurement():
    """The minimum over questions requires every question to have answered."""
    data = {"answers": {
        "correct": {"type": "noul", "noul": 0.95},
        "complete": {"type": "choice", "choice": "supported", "confidence": 0.9},
    }}

    assert score_from_answers(data) is None


# --------------------------------------------------------------------------- #
# One entry point so the evaluator never branches on the dialect itself.
# --------------------------------------------------------------------------- #


def test_the_dispatcher_builds_the_simple_payload():
    built = dialect_payload(
        SIMPLE, model="", system="", prompt="q", response="a",
        questions={"correct": "Is it correct?"},
    )

    assert built == {"prompt": "q", "response": "a"}


def test_the_dispatcher_builds_the_system_one_body():
    built = dialect_payload(
        TYPESAFE_DECISIONS, model="jev-1.13", system="s", prompt="q", response="a",
        questions={"correct": "Is it correct?"},
    )

    assert built["model"] == "jev-1.13"
    assert built["questions"]["correct"]["type"] == "noul"


def test_an_unknown_dialect_is_a_configuration_error_not_a_silent_no_score():
    with pytest.raises(ValueError, match="unknown JEV dialect"):
        dialect_payload(
            "typesafe_choices", model="jev-1.13", system="", prompt="q", response="a",
            questions={"correct": "Is it correct?"},
        )


def test_the_score_is_the_minimum_answer():
    """Questions are requirements, not votes: one low answer rejects the candidate."""
    from src.jev_dialects import score_from_answers

    data = {"answers": {
        "correct": {"type": "noul", "noul": 0.99},
        "on_brief": {"type": "noul", "noul": 0.1},
    }}

    assert score_from_answers(data) == 0.1


def test_a_single_answer_is_its_own_minimum():
    from src.jev_dialects import score_from_answers

    assert score_from_answers({"answers": {"correct": {"type": "noul", "noul": 0.7}}}) == 0.7
