"""Phase 6: the numbers a finance owner would ask for.

Three things were still missing from the telemetry:

* ``escalated`` — did the request leave the first tier at all?
* ``latency_ms_by_stage`` — where the time went, per level, with failures included.
* ``LOG_CONTENT`` — off by default, because prompts are the sensitive part.

And the costs must be *real*: computed from the ``usage`` the provider reported,
never from a percentage.
"""

from __future__ import annotations

import json
import logging

import pytest

from src.cascade_engine import CascadeEngine
from src.jev_evaluator import EvaluationResult
from src.provider import ProviderError
from src.settings import Settings
from src.telemetry import create_event

from tests.conftest import base_config_dict, load_config, write_config_data


class FakeEvaluator:
    def __init__(self, scores):
        self.scores = iter(scores)
        self.calls = 0

    async def evaluate(self, prompt, response, *, threshold=None, system=""):
        self.calls += 1
        score = next(self.scores)
        return EvaluationResult(approved=score >= threshold, score=score)


class FakeProvider:
    """Answers per model; a value that is an exception is raised instead."""

    def __init__(self, responses: dict[str, list]):
        self.responses = {model: list(items) for model, items in responses.items()}
        self.calls = []

    async def complete(self, model, messages, timeout_ms=None, **kwargs):
        self.calls.append(model)
        item = self.responses[model].pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def configured(tmp_path):
    return load_config(write_config_data(tmp_path, base_config_dict()))


def engine_for(config, provider, evaluator):
    # Settings carries LOG_CONTENT and the outbound-call cap; the engine owns both.
    return CascadeEngine(Settings(), config, provider=provider, evaluator=evaluator)


def completion(text: str, *, prompt_tokens: int = 1000, completion_tokens: int = 500):
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "provider-report-model",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


# ------------------------------------------------------- escalated / by-stage

async def test_a_first_tier_approval_is_not_an_escalation(tmp_path):
    config = configured(tmp_path)
    provider = FakeProvider({"openai/model-one": [completion("answer")]})
    engine = engine_for(config, provider, FakeEvaluator([0.9]))

    result = await engine.run([{"role": "user", "content": "q"}])

    assert result.hops_count == 1
    assert result.escalated is False
    assert list(result.latency_ms_by_stage) == ["openai/model-one"]


async def test_a_rejected_first_tier_is_an_escalation(tmp_path):
    config = configured(tmp_path)
    provider = FakeProvider({
        "openai/model-one": [completion("weak")],
        "anthropic/model-two": [completion("better")],
    })
    engine = engine_for(config, provider, FakeEvaluator([0.2, 0.9]))

    result = await engine.run([{"role": "user", "content": "q"}])

    assert result.hops_count == 2
    assert result.escalated is True
    assert list(result.latency_ms_by_stage) == ["openai/model-one", "anthropic/model-two"]


async def test_a_failed_level_is_still_a_stage(tmp_path):
    """A level that errored consumed time; hiding it would hide the real latency."""
    config = configured(tmp_path)
    provider = FakeProvider({
        "openai/model-one": [ProviderError("boom", provider="openai", upstream_status=503)],
        "anthropic/model-two": [completion("ok")],
    })
    engine = engine_for(config, provider, FakeEvaluator([0.9]))

    result = await engine.run([{"role": "user", "content": "q"}])

    assert list(result.latency_ms_by_stage) == ["openai/model-one", "anthropic/model-two"]
    assert all(value >= 0.0 for value in result.latency_ms_by_stage.values())
    assert result.escalated is True


async def test_stage_latencies_do_not_exceed_the_total(tmp_path):
    config = configured(tmp_path)
    provider = FakeProvider({
        "openai/model-one": [completion("weak")],
        "anthropic/model-two": [completion("ok")],
    })
    engine = engine_for(config, provider, FakeEvaluator([0.1, 0.9]))

    result = await engine.run([{"role": "user", "content": "q"}])

    assert sum(result.latency_ms_by_stage.values()) <= result.total_latency_ms + 0.001


async def test_a_bypass_records_one_stage(tmp_path):
    config = configured(tmp_path)
    provider = FakeProvider({"anthropic/model-three": [completion("tool call")]})
    engine = engine_for(config, provider, FakeEvaluator([]))

    result = await engine.run_bypass([{"role": "user", "content": "q"}], level_number=3)

    assert result.escalated is False
    assert list(result.latency_ms_by_stage) == ["anthropic/model-three"]


# --------------------------------------------------------------- real numbers

def test_cost_comes_from_reported_tokens_and_configured_prices(tmp_path):
    config = configured(tmp_path)

    event = create_event(
        config,
        model_used="openai/model-one",
        hops_count=1,
        jev_score=0.9,
        total_latency_ms=12.0,
        usage={"prompt_tokens": 1000, "completion_tokens": 500},
        candidate_usages=[(config.levels[0], {"prompt_tokens": 1000, "completion_tokens": 500})],
    )

    # Fixture prices: 1.0/Mtok in, 2.0/Mtok out -> 1000*1e-6 + 500*2e-6 = 0.002
    assert event.cost_usd == pytest.approx(0.002)
    # Baseline is the same usage priced at level 3: 9.0/Mtok in, 12.0/Mtok out
    assert event.baseline_cost_usd == pytest.approx(0.015)
    assert event.net_savings_usd == pytest.approx(0.013)


def test_rejected_candidates_are_paid_for(tmp_path):
    """A rejected answer was still generated and still costs money."""
    config = configured(tmp_path)
    usage = {"prompt_tokens": 1000, "completion_tokens": 500}

    event = create_event(
        config,
        model_used="anthropic/model-two",
        hops_count=2,
        jev_score=0.9,
        total_latency_ms=12.0,
        usage=usage,
        candidate_usages=[(config.levels[0], usage), (config.levels[1], usage)],
    )

    # L1: 1000*1e-6 + 500*2e-6 = 0.002 ; L2: 1000*3e-6 + 500*6e-6 = 0.006
    assert event.cost_usd == pytest.approx(0.002 + 0.006)


def test_savings_can_be_negative(tmp_path):
    """Escalating all the way costs more than going straight to the last tier."""
    config = configured(tmp_path)
    usage = {"prompt_tokens": 1_000_000, "completion_tokens": 0}

    event = create_event(
        config,
        model_used="anthropic/model-three",
        hops_count=3,
        jev_score=None,
        total_latency_ms=12.0,
        usage=usage,
        candidate_usages=[(config.levels[0], usage), (config.levels[1], usage),
                          (config.levels[2], usage)],
    )

    # 1Mtok in at each tier: 1.0 + 3.0 + 9.0 spent, 9.0 for the baseline alone.
    assert event.cost_usd == pytest.approx(13.0)
    assert event.baseline_cost_usd == pytest.approx(9.0)
    assert event.net_savings_usd == pytest.approx(-4.0)


def test_an_event_never_carries_tokens_it_was_not_given(tmp_path):
    config = configured(tmp_path)

    event = create_event(
        config,
        model_used="openai/model-one",
        hops_count=1,
        jev_score=0.9,
        total_latency_ms=12.0,
        usage={},
        candidate_usages=[(config.levels[0], {})],
    )

    assert event.cost_usd == 0.0
    assert event.baseline_cost_usd == 0.0
    assert event.net_savings_usd == 0.0


# ------------------------------------------------------------------ LOG_CONTENT

def _logged(caplog, event):
    event.emit()
    return json.loads(caplog.records[-1].getMessage())


def test_content_is_absent_by_default(tmp_path, caplog):
    config = configured(tmp_path)
    event = create_event(
        config, model_used="openai/model-one", hops_count=1, jev_score=0.9,
        total_latency_ms=12.0, usage={}, candidate_usages=[(config.levels[0], {})],
        prompt_text="my secret prompt", response_text="my secret answer",
    )

    with caplog.at_level(logging.INFO, logger="ai_cascade_proxy.telemetry"):
        payload = _logged(caplog, event)

    assert "prompt" not in payload
    assert "response" not in payload
    assert "my secret" not in json.dumps(payload)


def test_content_is_logged_when_explicitly_enabled(tmp_path, caplog):
    config = configured(tmp_path)
    event = create_event(
        config, model_used="openai/model-one", hops_count=1, jev_score=0.9,
        total_latency_ms=12.0, usage={}, candidate_usages=[(config.levels[0], {})],
        prompt_text="prompt", response_text="answer", log_content=True,
    )

    with caplog.at_level(logging.INFO, logger="ai_cascade_proxy.telemetry"):
        payload = _logged(caplog, event)

    assert payload["prompt"] == "prompt"
    assert payload["response"] == "answer"


def test_logged_content_is_truncated(tmp_path, caplog):
    config = configured(tmp_path)
    event = create_event(
        config, model_used="openai/model-one", hops_count=1, jev_score=0.9,
        total_latency_ms=12.0, usage={}, candidate_usages=[(config.levels[0], {})],
        prompt_text="x" * 5000, response_text="y" * 5000, log_content=True,
    )

    with caplog.at_level(logging.INFO, logger="ai_cascade_proxy.telemetry"):
        payload = _logged(caplog, event)

    assert len(payload["prompt"]) < 5000
    assert len(payload["response"]) < 5000


def test_the_event_carries_no_percentage(tmp_path, caplog):
    """`estimated_savings_pct` is gone for good: it was a guess, not a cost."""
    config = configured(tmp_path)
    event = create_event(
        config, model_used="openai/model-one", hops_count=1, jev_score=0.9,
        total_latency_ms=12.0, usage={"prompt_tokens": 10, "completion_tokens": 10},
        candidate_usages=[(config.levels[0], {"prompt_tokens": 10, "completion_tokens": 10})],
    )

    with caplog.at_level(logging.INFO, logger="ai_cascade_proxy.telemetry"):
        payload = _logged(caplog, event)

    assert "estimated_savings_pct" not in payload
    assert payload["escalated"] is False
    assert payload["latency_ms_by_stage"] == {"openai/model-one": pytest.approx(
        payload["total_latency_ms"], abs=1e-6)}


# ------------------------------------------------------------- end-to-end API

def make_client(cascade_config, settings=None):
    """An app wired to fakes that approve everything, for one API-level assertion."""
    from fastapi.testclient import TestClient

    from src.main import create_app

    class ApiProvider(FakeProvider):
        async def complete(self, model, messages, timeout_ms=None, **kwargs):
            return completion("the answer")

    class ApiEvaluator(FakeEvaluator):
        async def evaluate(self, prompt, response, *, threshold=None, system=""):
            return EvaluationResult(approved=True, score=0.99)

    return TestClient(create_app(
        settings=settings or Settings(),
        cascade_config=cascade_config,
        provider=ApiProvider({}),
        evaluator=ApiEvaluator([]),
    ))


def test_the_response_does_not_leak_cost_telemetry_to_clients(tmp_path):
    """Costs are an operator concern: the OpenAI-shaped body stays OpenAI-shaped."""
    client = make_client(configured(tmp_path))

    body = client.post("/v1/chat/completions", json={
        "model": "auto", "messages": [{"role": "user", "content": "q"}]}).json()

    assert set(body) == {"id", "object", "created", "model", "choices", "usage"}


def test_log_content_defaults_to_off_in_settings(tmp_path):
    # The flag exists, and its default is the safe one.
    assert Settings().log_content is False


def test_settings_can_enable_content_logging_from_the_environment(monkeypatch):
    monkeypatch.setenv("LOG_CONTENT", "true")

    assert Settings.from_env().log_content is True
