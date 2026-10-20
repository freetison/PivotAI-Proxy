"""The client's system prompt must reach JEV, and its cost must reach telemetry.

Two things the dialect work alone does not fix:

1. ``_user_prompt`` concatenates ``role == "user"`` turns only, so an answer that
   ignores the client's instructions was evaluated against a prompt that no longer
   contained them. The System One dialect can carry them, so the engine must pass
   them.
2. OpenRouter bills the verification and reports ``usage.cost``. Leaving it out of
   ``cost_usd`` makes the savings figure flattering — the cost of deciding is a
   real cost.
"""

from __future__ import annotations

import json

import httpx
import httpx2
import pytest

from src.cascade_engine import CascadeEngine
from src.jev_evaluator import JevEvaluator
from src.settings import Settings

from tests.conftest import base_config_dict, load_config, write_config_data


def openrouter_settings(**overrides) -> Settings:
    values = {
        "jev_api_key": "or-key",
        "jev_api_url": "https://openrouter.ai/api",
        "jev_dialect": "typesafe_decisions",
        "jev_model": "jev-1.13",
    }
    values.update(overrides)
    return Settings(**values)


class RecordingJev:
    """Records every System One body and replies with a fixed noul answer."""

    def __init__(self, *, noul: float = 0.99, cost: float | None = 0.00004, status: int = 200):
        self.noul = noul
        self.cost = cost
        self.status = status
        self.bodies: list[dict] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.bodies.append(json.loads(request.content))
        if self.status != 200:
            return httpx2.Response(self.status, json={"error": {"message": "nope"}})
        usage = {"input_tokens": 100, "output_tokens": 10}
        if self.cost is not None:
            usage["cost"] = self.cost
        return httpx2.Response(200, json={
            "model": "typesafe/jev-1.13-20260917",
            "answers": {"correct": {"type": "noul", "noul": self.noul}},
            "usage": usage,
        })


class FixedProvider:
    """One cheap answer per call, with usage so cost has something to price."""

    def __init__(self, text: str = "The answer."):
        self.text = text
        self.calls: list[str] = []

    async def complete(self, model, messages, *, timeout_ms=None, **kwargs):
        self.calls.append(model)
        return {
            "choices": [{"message": {"role": "assistant", "content": self.text},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 500},
        }


def config_with(env: dict | None = None):
    import pathlib
    import tempfile

    data = base_config_dict()
    # The cascade config is the source of truth, so the dialect lives there too.
    data["jev"].update({"dialect": "typesafe_decisions", "model": "jev-1.13"})
    path = write_config_data(pathlib.Path(tempfile.mkdtemp()), data)
    return load_config(path, env=env or {}, api_keys={"openai": "k", "anthropic": "k"})


def engine_with(config, *, provider=None, jev=None, settings=None):
    settings = settings or openrouter_settings()
    evaluator = JevEvaluator(settings, config, transport=httpx2.MockTransport(jev or RecordingJev()))
    return CascadeEngine(settings, config, provider=provider or FixedProvider(), evaluator=evaluator)


# --------------------------------------------------------------------------- #
# The system prompt
# --------------------------------------------------------------------------- #

SYSTEM = "Always answer in JSON. Never mention competitors."


async def test_the_system_prompt_is_sent_to_jev():
    jev = RecordingJev()
    engine = engine_with(config_with(), jev=jev)

    await engine.run([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "What is the capital of France?"},
    ])

    assert jev.bodies[0]["state"]["system"] == SYSTEM


async def test_the_user_turns_are_still_concatenated_into_the_question():
    jev = RecordingJev()
    engine = engine_with(config_with(), jev=jev)

    await engine.run([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ignored"},
        {"role": "user", "content": "second"},
    ])

    assert jev.bodies[0]["state"]["question"] == "first\nsecond"


async def test_several_system_turns_are_joined_in_order():
    jev = RecordingJev()
    engine = engine_with(config_with(), jev=jev)

    await engine.run([
        {"role": "system", "content": "one"},
        {"role": "system", "content": "two"},
        {"role": "user", "content": "q"},
    ])

    assert jev.bodies[0]["state"]["system"] == "one\ntwo"


async def test_no_system_prompt_is_an_empty_string_not_a_missing_field():
    jev = RecordingJev()
    engine = engine_with(config_with(), jev=jev)

    await engine.run([{"role": "user", "content": "q"}])

    assert jev.bodies[0]["state"]["system"] == ""


async def test_the_answer_under_evaluation_is_the_candidate_not_the_prompt():
    jev = RecordingJev()
    engine = engine_with(config_with(), jev=jev, provider=FixedProvider("candidate text"))

    await engine.run([{"role": "user", "content": "q"}])

    assert jev.bodies[0]["state"]["answer"] == "candidate text"


# --------------------------------------------------------------------------- #
# The cost of the verification
# --------------------------------------------------------------------------- #


async def test_the_verification_cost_is_part_of_the_request_cost(caplog):
    import logging

    caplog.set_level(logging.INFO, logger="ai_cascade_proxy.telemetry")
    engine = engine_with(config_with(), jev=RecordingJev(noul=0.99, cost=0.00004))

    result = await engine.run([{"role": "user", "content": "q"}])

    # One level-1 call at the fixture prices: 1000 in @ 1.0/M + 500 out @ 2.0/M = 0.002
    # plus JEV's 0.00004 — the cascade alone would report 0.002.
    assert result.cost_usd == pytest.approx(0.00204)
    events = [json.loads(record.getMessage()) for record in caplog.records]
    events = [event for event in events if "cost_usd" in event]
    assert events[-1]["cost_usd"] == pytest.approx(0.00204)


async def test_a_jev_call_that_reports_no_cost_adds_nothing(caplog):
    engine = engine_with(config_with(), jev=RecordingJev(noul=0.99, cost=None))

    result = await engine.run([{"role": "user", "content": "q"}])

    assert result.cost_usd == pytest.approx(0.002)


async def test_an_unavailable_jev_call_adds_no_cost():
    # fail_open keeps the request on level 1, so the total is one provider call
    # and nothing else: an evaluation that failed must not be billed.
    config = config_with(env={"JEV_FAILURE_POLICY": "fail_open"})
    engine = engine_with(config, jev=RecordingJev(status=500))

    result = await engine.run([{"role": "user", "content": "q"}])

    assert result.cost_usd == pytest.approx(0.002)


async def test_the_jev_cost_is_reported_separately_for_auditing(caplog):
    """`cost_usd` is the total; `jev_cost_usd` isolates what the decisions cost."""
    import logging

    caplog.set_level(logging.INFO, logger="ai_cascade_proxy.telemetry")
    engine = engine_with(config_with(), jev=RecordingJev(noul=0.99, cost=0.00004))

    await engine.run([{"role": "user", "content": "q"}])

    events = [json.loads(record.getMessage()) for record in caplog.records]
    events = [event for event in events if "cost_usd" in event]
    assert events[-1]["jev_cost_usd"] == pytest.approx(0.00004)


async def test_every_evaluated_level_charges_its_own_verification(caplog):
    """Two rejections mean two JEV calls, so two verifications are billed."""
    import logging

    caplog.set_level(logging.INFO, logger="ai_cascade_proxy.telemetry")
    jev = RecordingJev(noul=0.10, cost=0.00004)
    engine = engine_with(config_with(), jev=jev)

    result = await engine.run([{"role": "user", "content": "q"}])

    assert len(jev.bodies) == 2
    # Two rejections escalate to the final level: L1 0.002 + L2 0.006 +
    # L3 (1000 in @9.0 + 500 out @12.0 = 0.015) + two verifications at 0.00004.
    assert result.cost_usd == pytest.approx(0.02308)


# --------------------------------------------------------------------------- #
# The other two paths
# --------------------------------------------------------------------------- #


async def test_a_bypass_carries_no_verification_cost():
    """A bypass never asks JEV, so it must never add a verification cost."""
    config = config_with()
    engine = engine_with(config, jev=RecordingJev(noul=0.99, cost=0.5))

    result = await engine.run_bypass([{"role": "user", "content": "q"}], level_number=1)

    assert result.jev_cost_usd == 0.0
    assert result.cost_usd == pytest.approx(0.002)


async def test_shadow_mode_reports_the_verification_cost_of_the_cascade(caplog):
    import logging

    caplog.set_level(logging.INFO, logger="ai_cascade_proxy.telemetry")
    settings = openrouter_settings(cascade_mode="shadow")
    config = config_with()
    jev = RecordingJev(noul=0.10, cost=0.00004)
    engine = engine_with(config, jev=jev, settings=settings)

    result = await engine.run([{"role": "user", "content": "q"}])

    assert result.cascade_mode == "shadow"
    # Two rejections before the cascade falls through to the final level.
    assert len(jev.bodies) == 2
    events = [json.loads(record.getMessage()) for record in caplog.records]
    events = [event for event in events if "cost_usd" in event]
    assert events[-1]["jev_cost_usd"] == pytest.approx(0.00008)
