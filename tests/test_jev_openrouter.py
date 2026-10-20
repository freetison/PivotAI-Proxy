"""The ``typesafe_decisions`` dialect end to end, through the official SDK.

These tests drive ``JevEvaluator`` with the real ``AsyncTypeSafeClient`` pointed
at ``httpx2.MockTransport``. No network: what is being verified is the wire
payload, the response parsing, and the guarantees the rest of the proxy depends
on — no hidden retries, no invented score, and a cost that is reported rather
than guessed.
"""

from __future__ import annotations

import json

import httpx
import httpx2
import pytest

from src.cascade_config import load_cascade_config
from src.jev_evaluator import JevEvaluator
from src.settings import Settings

from tests.conftest import base_config_dict, load_config, write_config_data

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def openrouter_settings(**overrides) -> Settings:
    values = {
        "jev_api_key": "or-key",
        "jev_api_url": "https://openrouter.ai/api",
        "jev_dialect": "typesafe_decisions",
        "jev_model": "jev-1.13",
    }
    values.update(overrides)
    return Settings(**values)


def noul_response(score: float, *, cost: float | None = 0.00003) -> dict:
    usage: dict = {"input_tokens": 275, "output_tokens": 20}
    if cost is not None:
        usage["cost"] = cost
    return {
        "id": "gen-dec-1789738314-X5e5eKGQdvR9rblyX250",
        "model": "typesafe/jev-1.13-20260917",
        "provider": "TypeSafe",
        "answers": {"correct": {"type": "noul", "noul": score}},
        "usage": usage,
    }


class Recording:
    """A ``httpx2.MockTransport`` handler that records every attempt."""

    def __init__(self, *responses: dict | int):
        self.responses = list(responses)
        self.requests: list[dict] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append({
            "url": str(request.url),
            "method": request.method,
            "headers": dict(request.headers),
            "body": json.loads(request.content) if request.content else None,
        })
        # The last response repeats, so a test can assert on attempt count.
        item = self.responses[min(len(self.requests), len(self.responses)) - 1]
        if isinstance(item, int):
            return httpx2.Response(item, json={"error": {"message": "upstream failure"}})
        return httpx2.Response(200, json=item)


def evaluator_for(settings: Settings, transport: Recording, config=None) -> JevEvaluator:
    return JevEvaluator(settings, config, transport=httpx2.MockTransport(transport))


def config_with(env: dict | None = None):
    import pathlib
    import tempfile

    path = write_config_data(pathlib.Path(tempfile.mkdtemp()), base_config_dict())
    return load_config(path, env=env or {}, api_keys={"openai": "k", "anthropic": "k"})


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


def test_the_dialect_defaults_to_the_existing_simple_contract():
    """Nothing changes for a deployment that does not opt in."""
    config = config_with()

    assert config.jev_dialect == "simple"


def test_the_dialect_can_be_set_in_the_yaml():
    import pathlib
    import tempfile

    data = base_config_dict()
    data["jev"].update({"dialect": "typesafe_decisions", "model": "jev-1.13"})
    config = load_config(write_config_data(pathlib.Path(tempfile.mkdtemp()), data))

    assert config.jev_dialect == "typesafe_decisions"
    assert config.jev_model == "jev-1.13"


def test_the_environment_overrides_the_dialect():
    config = config_with({"JEV_DIALECT": "typesafe_decisions", "JEV_MODEL": "jev-1.14"})

    assert config.jev_dialect == "typesafe_decisions"
    assert config.jev_model == "jev-1.14"


def test_an_unknown_dialect_is_refused_at_startup():
    with pytest.raises(Exception) as caught:
        config_with({"JEV_DIALECT": "systemone"})

    assert "JEV_DIALECT must be one of" in str(caught.value)


def test_typesafe_decisions_without_a_model_is_refused_at_startup():
    """Without a model the API returns 400 on the first request; catch it at boot."""
    with pytest.raises(Exception) as caught:
        config_with({"JEV_DIALECT": "typesafe_decisions"})

    assert "jev.model" in str(caught.value)


def test_the_questions_are_configurable_and_default_to_one_question():
    config = config_with()

    assert list(config.jev_questions) == ["correct"]
    assert config.jev_questions["correct"]


def test_the_question_set_can_be_overridden_from_the_yaml():
    import pathlib
    import tempfile

    data = base_config_dict()
    data["jev"]["questions"] = {
        "correct": "Is the answer correct?",
        "complete": "Does the answer address every part of the question?",
    }
    config = load_config(write_config_data(pathlib.Path(tempfile.mkdtemp()), data))

    assert list(config.jev_questions) == ["correct", "complete"]


def test_an_empty_question_set_is_refused_at_startup():
    import pathlib
    import tempfile

    data = base_config_dict()
    data["jev"]["questions"] = {}
    with pytest.raises(Exception) as caught:
        load_config(write_config_data(pathlib.Path(tempfile.mkdtemp()), data))

    assert "questions" in str(caught.value)


# --------------------------------------------------------------------------- #
# The request
# --------------------------------------------------------------------------- #


async def test_the_request_goes_to_the_system_one_endpoint():
    transport = Recording(noul_response(0.9))
    evaluator = evaluator_for(openrouter_settings(), transport)

    await evaluator.score("What is the capital of France?", "Paris.", system="Be terse.")

    assert transport.requests[0]["url"] == "https://openrouter.ai/api/v1/systemone"
    assert transport.requests[0]["method"] == "POST"


async def test_the_openrouter_key_is_sent_as_a_bearer_token():
    transport = Recording(noul_response(0.9))

    await evaluator_for(openrouter_settings(), transport).score("q", "a")

    assert transport.requests[0]["headers"]["authorization"] == "Bearer or-key"


async def test_the_body_asks_a_noul_question_about_the_state():
    transport = Recording(noul_response(0.9))

    await evaluator_for(openrouter_settings(), transport).score(
        "What is the capital of France?", "Paris.", system="Be terse.",
    )

    body = transport.requests[0]["body"]
    assert body["model"] == "jev-1.13"
    assert body["state"] == {
        "system": "Be terse.",
        "question": "What is the capital of France?",
        "answer": "Paris.",
    }
    assert body["questions"]["correct"]["type"] == "noul"


async def test_the_system_prompt_reaches_jev_instead_of_being_dropped():
    transport = Recording(noul_response(0.2))

    await evaluator_for(openrouter_settings(), transport).score(
        "Say hi", "Hello!", system="Answer in JSON only.",
    )

    assert transport.requests[0]["body"]["state"]["system"] == "Answer in JSON only."


async def test_no_hidden_retries_are_made():
    """The cascade owns the outbound budget; a retrying SDK would break it."""
    transport = Recording(500, noul_response(0.9))

    score = await evaluator_for(openrouter_settings(), transport).score("q", "a")

    assert score is None
    assert len(transport.requests) == 1


# --------------------------------------------------------------------------- #
# The response
# --------------------------------------------------------------------------- #


async def test_the_noul_probability_is_the_score():
    transport = Recording(noul_response(0.93))

    assert await evaluator_for(openrouter_settings(), transport).score("q", "a") == pytest.approx(0.93)


async def test_the_score_is_the_minimum_of_every_question():
    transport = Recording({"answers": {
        "correct": {"type": "noul", "noul": 0.95},
        "complete": {"type": "noul", "noul": 0.55},
    }})

    score = await evaluator_for(openrouter_settings(), transport).score("q", "a")

    assert score == pytest.approx(0.55)


async def test_a_transport_failure_is_no_measurement():
    class Broken:
        def __call__(self, request):
            raise httpx2.ConnectError("unreachable")

    evaluator = JevEvaluator(openrouter_settings(), None,
                             transport=httpx2.MockTransport(Broken()))

    assert await evaluator.score("q", "a") is None


@pytest.mark.parametrize("status", [400, 402, 404, 429, 500, 503])
async def test_an_unsuccessful_response_is_no_measurement(status):
    transport = Recording(status)

    assert await evaluator_for(openrouter_settings(), transport).score("q", "a") is None
    assert len(transport.requests) == 1


async def test_a_malformed_answer_is_no_measurement():
    transport = Recording({"answers": {"correct": {"type": "noul", "noul": "very likely"}}})

    assert await evaluator_for(openrouter_settings(), transport).score("q", "a") is None


# --------------------------------------------------------------------------- #
# The cost of the verification
# --------------------------------------------------------------------------- #


async def test_the_cost_reported_by_openrouter_is_returned_not_guessed():
    transport = Recording(noul_response(0.9, cost=0.000031))

    measurement = await evaluator_for(openrouter_settings(), transport).measure("q", "a")

    assert measurement.score == pytest.approx(0.9)
    assert measurement.cost_usd == pytest.approx(0.000031)


async def test_a_missing_cost_is_zero_and_never_an_invented_price():
    transport = Recording(noul_response(0.9, cost=None))

    measurement = await evaluator_for(openrouter_settings(), transport).measure("q", "a")

    assert measurement.cost_usd == 0.0


async def test_an_unavailable_evaluation_costs_nothing():
    transport = Recording(500)

    measurement = await evaluator_for(openrouter_settings(), transport).measure("q", "a")

    assert measurement.score is None
    assert measurement.cost_usd == 0.0


# --------------------------------------------------------------------------- #
# The failure policy is unchanged by the dialect
# --------------------------------------------------------------------------- #


async def test_fail_closed_still_refuses_when_the_sdk_cannot_answer():
    from src.jev_evaluator import JevUnavailable

    config = config_with({"JEV_FAILURE_POLICY": "fail_closed"})
    evaluator = evaluator_for(openrouter_settings(), Recording(503), config)

    with pytest.raises(JevUnavailable):
        await evaluator.evaluate("q", "a", threshold=0.8)


async def test_fail_open_still_approves_when_the_sdk_cannot_answer():
    config = config_with({"JEV_FAILURE_POLICY": "fail_open"})
    evaluator = evaluator_for(openrouter_settings(), Recording(503), config)

    result = await evaluator.evaluate("q", "a", threshold=0.8)

    assert result.approved is True
    assert result.status == "unavailable"
    assert result.score is None
