"""Phase 3: provider failures consume a hop, and every outbound call is counted.

Two invariants are under test:

1. A 429/5xx/timeout from level N consumes the hop and escalates to level N+1;
   if the last level fails, the client gets an OpenAI-shaped error, never an
   empty answer and never a 200.
2. Retries are ours and counted. Only 429 may be retried, at most once, and the
   total number of outbound provider calls per request has a hard cap that is
   enforced even when the hop count stays within the three configured levels.
"""

from __future__ import annotations

import asyncio
import json
import logging

import httpx
import pytest
from fastapi.testclient import TestClient

from src.cascade_config import CascadeConfig
from src.cascade_engine import CascadeEngine, OutboundCallLimitExceeded, UpstreamCascadeFailed
from src.main import create_app
from src.metrics import Metrics
from src.provider import ModelProvider, ProviderError
from src.settings import Settings

from tests.conftest import base_config_dict, load_config, write_config_data

L1, L2, L3 = "openai/model-one", "anthropic/model-two", "anthropic/model-three"


def config_with(tmp_path, **overrides) -> CascadeConfig:
    payload = base_config_dict()
    payload["cascade"].update(overrides)
    return load_config(write_config_data(tmp_path, payload),
                       env={"JEV_FAILURE_POLICY": "escalate"})


def completion(text: str) -> dict:
    return {
        "id": "provider-id",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


class ScriptedProvider:
    """Answers per model with a scripted outcome: a dict payload or an int HTTP status."""

    def __init__(self, script: dict[str, list], *, timeout_models: tuple[str, ...] = ()):
        self.script = {model: list(outcomes) for model, outcomes in script.items()}
        self.timeout_models = set(timeout_models)
        self.calls: list[str] = []

    @property
    def outbound_calls(self) -> int:
        return len(self.calls)

    async def complete(self, model, messages, timeout_ms=None, **kwargs):
        self.calls.append(model)
        if model in self.timeout_models:
            raise ProviderError(f"Provider {model} request timed out", provider=model.split("/")[0])
        outcomes = self.script.get(model) or [503]
        outcome = outcomes.pop(0) if len(outcomes) > 1 else outcomes[0]
        if isinstance(outcome, dict):
            return outcome
        raise ProviderError(
            f"Upstream provider {model.split('/')[0]} returned an error",
            status_code=outcome, provider=model.split("/")[0],
            upstream_status=outcome, retryable=outcome == 429,
        )


class ApprovingEvaluator:
    """Approves whatever it is given: isolates provider behaviour from JEV decisions."""

    async def evaluate(self, prompt, response, *, threshold=None, system=""):
        from src.jev_evaluator import EvaluationResult
        return EvaluationResult(approved=True, score=0.99, status="approved")


def engine_for(config: CascadeConfig, provider, **kwargs) -> CascadeEngine:
    return CascadeEngine(Settings(), config,
                         provider=provider, evaluator=ApprovingEvaluator(), **kwargs)


# ------------------------------------------------------------------ hop consumption

@pytest.mark.asyncio
async def test_level_1_failure_escalates_and_level_2_answers(tmp_path):
    provider = ScriptedProvider({L1: [503], L2: [completion("answer from two")]})
    engine = engine_for(config_with(tmp_path), provider)

    result = await engine.run([{"role": "user", "content": "Q"}])

    assert provider.calls == [L1, L2]
    assert result.model_used == L2
    assert result.hops_count == 2
    assert result.outbound_calls == 2
    assert result.response["choices"][0]["message"]["content"] == "answer from two"


@pytest.mark.asyncio
async def test_level_1_and_2_failure_reach_level_3(tmp_path):
    provider = ScriptedProvider({L1: [500], L2: [502], L3: [completion("answer from three")]})
    engine = engine_for(config_with(tmp_path), provider)

    result = await engine.run([{"role": "user", "content": "Q"}])

    assert provider.calls == [L1, L2, L3]
    assert result.model_used == L3
    assert result.hops_count == 3


@pytest.mark.asyncio
async def test_a_timeout_consumes_the_hop_like_any_other_failure(tmp_path):
    provider = ScriptedProvider({L2: [completion("answer from two")]}, timeout_models=(L1,))
    engine = engine_for(config_with(tmp_path), provider)

    result = await engine.run([{"role": "user", "content": "Q"}])

    assert provider.calls == [L1, L2]
    assert result.model_used == L2


@pytest.mark.asyncio
async def test_the_last_level_failing_raises_instead_of_returning_an_empty_answer(tmp_path):
    provider = ScriptedProvider({L1: [503], L2: [503], L3: [503]})
    engine = engine_for(config_with(tmp_path), provider)

    with pytest.raises(UpstreamCascadeFailed) as caught:
        await engine.run([{"role": "user", "content": "Q"}])

    assert caught.value.status_code == 502
    assert provider.calls == [L1, L2, L3]
    assert caught.value.failures == {L1: 503, L2: 503, L3: 503}


@pytest.mark.asyncio
async def test_the_error_names_every_level_that_was_tried(tmp_path):
    provider = ScriptedProvider({L1: [503], L2: [503], L3: [503]})
    engine = engine_for(config_with(tmp_path), provider)

    with pytest.raises(UpstreamCascadeFailed) as caught:
        await engine.run([{"role": "user", "content": "Q"}])

    message = str(caught.value)
    assert "openai/model-one" in message and "anthropic/model-two" in message
    assert "anthropic/model-three" in message


# -------------------------------------------------------------------------- retries

@pytest.mark.asyncio
async def test_a_429_is_retried_once_on_the_same_level(tmp_path):
    provider = ScriptedProvider({L1: [429, completion("answer after retry")]})
    engine = engine_for(config_with(tmp_path), provider, retry_backoff_ms=0)

    result = await engine.run([{"role": "user", "content": "Q"}])

    assert provider.calls == [L1, L1]
    assert result.model_used == L1
    assert result.hops_count == 1
    assert result.outbound_calls == 2
    assert result.response["choices"][0]["message"]["content"] == "answer after retry"


@pytest.mark.asyncio
async def test_a_second_429_is_not_retried_again(tmp_path):
    provider = ScriptedProvider({L1: [429], L2: [completion("answer from two")]})
    engine = engine_for(config_with(tmp_path), provider, retry_backoff_ms=0)

    result = await engine.run([{"role": "user", "content": "Q"}])

    # One retry of L1 (both 429), then the hop moves on to L2: 3 outbound calls total.
    assert provider.calls == [L1, L1, L2]
    assert result.model_used == L2


@pytest.mark.parametrize("status", [500, 502, 503, 504])
@pytest.mark.asyncio
async def test_server_errors_are_not_retried_inside_a_level(tmp_path, status):
    provider = ScriptedProvider({L1: [status], L2: [completion("answer from two")]})
    engine = engine_for(config_with(tmp_path), provider, retry_backoff_ms=0)

    await engine.run([{"role": "user", "content": "Q"}])

    assert provider.calls == [L1, L2]


@pytest.mark.asyncio
async def test_a_timeout_is_not_retried_inside_a_level(tmp_path):
    provider = ScriptedProvider({L2: [completion("answer from two")]}, timeout_models=(L1,))
    engine = engine_for(config_with(tmp_path), provider, retry_backoff_ms=0)

    await engine.run([{"role": "user", "content": "Q"}])

    assert provider.calls == [L1, L2]


@pytest.mark.asyncio
async def test_the_retry_waits_before_calling_again(tmp_path):
    provider = ScriptedProvider({L1: [429, completion("ok")]})
    engine = engine_for(config_with(tmp_path), provider, retry_backoff_ms=40)

    started = asyncio.get_running_loop().time()
    await engine.run([{"role": "user", "content": "Q"}])
    elapsed = asyncio.get_running_loop().time() - started

    assert elapsed >= 0.04


# -------------------------------------------------------------- the hard call cap

@pytest.mark.asyncio
async def test_the_outbound_call_cap_aborts_even_when_hops_stay_within_three(tmp_path):
    # Every level returns 429, so each one retries once: without a cap this would
    # make 6 outbound calls while using only 3 hops.
    provider = ScriptedProvider({L1: [429], L2: [429], L3: [429]})
    engine = engine_for(config_with(tmp_path), provider, retry_backoff_ms=0,
                        max_outbound_calls=4)

    with pytest.raises(OutboundCallLimitExceeded) as caught:
        await engine.run([{"role": "user", "content": "Q"}])

    assert caught.value.limit == 4
    assert provider.calls == [L1, L1, L2, L2]
    assert len(provider.calls) <= 4


@pytest.mark.asyncio
async def test_the_cap_allows_exactly_the_configured_number_of_calls(tmp_path):
    provider = ScriptedProvider({L1: [429, completion("ok")]})
    engine = engine_for(config_with(tmp_path), provider, retry_backoff_ms=0,
                        max_outbound_calls=2)

    result = await engine.run([{"role": "user", "content": "Q"}])

    assert provider.calls == [L1, L1]
    assert result.outbound_calls == 2


@pytest.mark.asyncio
async def test_the_default_cap_scales_with_max_hops(tmp_path):
    config = config_with(tmp_path)
    engine = engine_for(config, ScriptedProvider({}))

    # Three levels, at most one retry each: six is the documented default ceiling.
    assert engine.max_outbound_calls == 2 * config.max_hops


@pytest.mark.asyncio
async def test_the_cap_cannot_be_disabled_by_a_huge_value(tmp_path, monkeypatch):
    monkeypatch.setenv("MAX_OUTBOUND_CALLS", "1000")

    with pytest.raises(ValueError, match="MAX_OUTBOUND_CALLS"):
        Settings.from_env()


# ------------------------------------------------------------------- observability

@pytest.mark.asyncio
async def test_metrics_count_outbound_calls(tmp_path):
    metrics = Metrics()
    provider = ScriptedProvider({L1: [503], L2: [completion("answer from two")]})
    engine = engine_for(config_with(tmp_path), provider, metrics=metrics)

    await engine.run([{"role": "user", "content": "Q"}])

    snapshot = metrics.snapshot()
    assert snapshot["outbound_calls_total"] == 2
    assert snapshot["provider_failures_total"] == 1


@pytest.mark.asyncio
async def test_telemetry_records_outbound_calls_and_failed_levels(tmp_path, caplog):
    provider = ScriptedProvider({L1: [503], L2: [completion("answer from two")]})
    engine = engine_for(config_with(tmp_path), provider)

    with caplog.at_level(logging.INFO, logger="ai_cascade_proxy.telemetry"):
        await engine.run([{"role": "user", "content": "Q"}])

    event = json.loads(caplog.records[-1].getMessage())
    assert event["outbound_calls"] == 2
    assert event["failed_levels"] == ["openai/model-one"]


# ------------------------------------------------------------------------- the API

def app_for(tmp_path, provider, **app_kwargs) -> TestClient:
    config = config_with(tmp_path)
    engine = CascadeEngine(
        app_kwargs.pop("settings", Settings()),
        config,
        provider=provider,
        evaluator=ApprovingEvaluator(),
        retry_backoff_ms=0,
        **app_kwargs,
    )
    return TestClient(create_app(settings=Settings(), cascade_config=config, engine=engine))


def test_all_levels_failing_returns_a_502_with_an_openai_error_body(tmp_path):
    provider = ScriptedProvider({L1: [503], L2: [503], L3: [503]})
    with app_for(tmp_path, provider) as client:
        response = client.post("/v1/chat/completions",
                               json={"model": "auto",
                                     "messages": [{"role": "user", "content": "Q"}]})

    assert response.status_code == 502
    body = response.json()
    assert body["error"]["type"] == "upstream_error"
    assert body["error"]["code"] == "cascade_failed"
    assert "choices" not in body


def test_exceeding_the_call_cap_returns_an_openai_error_body(tmp_path):
    provider = ScriptedProvider({L1: [429], L2: [429], L3: [429]})
    with app_for(tmp_path, provider, max_outbound_calls=3) as client:
        response = client.post("/v1/chat/completions",
                               json={"model": "auto",
                                     "messages": [{"role": "user", "content": "Q"}]})

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "outbound_call_limit"
    assert len(provider.calls) <= 3


def test_a_successful_escalation_still_returns_200(tmp_path):
    provider = ScriptedProvider({L1: [503], L2: [completion("answer from two")]})
    with app_for(tmp_path, provider) as client:
        response = client.post("/v1/chat/completions",
                               json={"model": "auto",
                                     "messages": [{"role": "user", "content": "Q"}]})

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "answer from two"
    assert response.json()["model"] == L2
