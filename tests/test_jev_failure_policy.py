"""Phase 2: what the cascade does when JEV cannot produce a score.

Every test uses a fake evaluator or a mock HTTP transport: no real JEV call.
The invariant under test: an unavailable evaluation is never turned into a
score, and the configured policy alone decides what happens next.
"""

from __future__ import annotations

import json
import logging

import httpx
import pytest
from fastapi.testclient import TestClient

from src.cascade_config import CascadeConfig, CascadeConfigError, configuration_warnings
from src.cascade_engine import CascadeEngine
from src.jev_evaluator import JevEvaluator, JevUnavailable
from src.main import create_app
from src.metrics import Metrics
from src.settings import Settings

from tests.conftest import base_config_dict, load_config, write_config_data

JEV_SETTINGS = Settings(jev_api_key="k", jev_api_url="https://jev.test/evaluate")


def config_with_policy(directory, policy: str) -> CascadeConfig:
    """The reference fixture configuration, with one failure policy."""
    payload = base_config_dict()
    payload["jev"]["failure_policy"] = policy
    return load_config(write_config_data(directory, payload))


def provider_response(text: str) -> dict:
    return {
        "id": "provider-id",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


class RecordingProvider:
    """Returns the nth scripted answer; extra calls repeat the last one."""

    def __init__(self, texts: list[str]):
        self.texts = list(texts)
        self.models: list[str] = []

    async def complete(self, model, messages, timeout_ms=None, **kwargs):
        self.models.append(model)
        index = min(len(self.models) - 1, len(self.texts) - 1)
        return provider_response(self.texts[index])


def down_evaluator(config) -> JevEvaluator:
    """The real evaluator, with an HTTP client that always fails: JEV is down."""

    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("JEV is unreachable", request=request)

    return JevEvaluator(JEV_SETTINGS, config,
                        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


def scored_evaluator(config, score: float) -> JevEvaluator:
    """The real evaluator, with an HTTP client that returns a fixed score."""
    return JevEvaluator(
        JEV_SETTINGS, config,
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"score": score}))
        ),
    )


# ---------------------------------------------------------------- configuration

@pytest.mark.parametrize("policy", ["fail_open", "escalate", "fail_closed"])
def test_policy_is_read_from_the_yaml(tmp_path, policy):
    assert config_with_policy(tmp_path, policy).failure_policy == policy


def test_policy_defaults_to_escalate_when_the_key_is_absent(tmp_path):
    payload = base_config_dict()
    del payload["jev"]["failure_policy"]

    config = load_config(write_config_data(tmp_path, payload))

    assert config.failure_policy == "escalate"


def test_env_overrides_the_policy(tmp_path):
    payload = base_config_dict()
    payload["jev"]["failure_policy"] = "fail_open"

    config = load_config(write_config_data(tmp_path, payload),
                         env={"JEV_FAILURE_POLICY": "fail_closed"})

    assert config.failure_policy == "fail_closed"


def test_invalid_policy_in_env_fails_with_a_specific_message(tmp_path):
    with pytest.raises(CascadeConfigError, match="JEV_FAILURE_POLICY must be one of"):
        load_config(write_config_data(tmp_path, base_config_dict()),
                    env={"JEV_FAILURE_POLICY": "ignore"})


def test_no_warning_claims_the_policy_is_unimplemented(cascade_config):
    warnings = configuration_warnings(cascade_config)

    assert not [w for w in warnings if "still fails open" in w]


def test_fail_closed_warns_about_the_client_visible_outage(tmp_path):
    warnings = configuration_warnings(config_with_policy(tmp_path, "fail_closed"))

    assert any("fail_closed" in warning and "503" in warning for warning in warnings)


# ------------------------------------------------------------------- evaluator

@pytest.mark.asyncio
async def test_fail_open_approves_an_unavailable_evaluation(tmp_path):
    evaluator = JevEvaluator(JEV_SETTINGS, config_with_policy(tmp_path, "fail_open"))

    result = await evaluator.evaluate("q", "a", threshold=0.8)

    assert result.approved is True
    assert result.score is None
    assert result.fallback is True
    assert result.status == "unavailable"


@pytest.mark.asyncio
async def test_escalate_never_approves_an_unavailable_evaluation(tmp_path):
    evaluator = JevEvaluator(JEV_SETTINGS, config_with_policy(tmp_path, "escalate"))

    result = await evaluator.evaluate("q", "a", threshold=0.8)

    assert result.approved is False
    assert result.score is None
    assert result.fallback is True
    assert result.status == "unavailable"


@pytest.mark.asyncio
async def test_fail_closed_raises_instead_of_approving_or_rejecting(tmp_path):
    evaluator = JevEvaluator(JEV_SETTINGS, config_with_policy(tmp_path, "fail_closed"))

    with pytest.raises(JevUnavailable, match="unavailable"):
        await evaluator.evaluate("q", "a", threshold=0.8)


@pytest.mark.asyncio
async def test_a_real_score_still_decides_regardless_of_policy(tmp_path):
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"score": 0.4}))
    )
    evaluator = JevEvaluator(JEV_SETTINGS, config_with_policy(tmp_path, "fail_closed"),
                             client=client)

    result = await evaluator.evaluate("q", "a", threshold=0.8)
    await client.aclose()

    assert result.approved is False
    assert result.status == "rejected"
    assert result.score == pytest.approx(0.4)


@pytest.mark.asyncio
async def test_a_timeout_is_unavailable_under_every_policy(tmp_path):
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("JEV timed out", request=request)

    for policy in ("fail_open", "escalate"):
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        evaluator = JevEvaluator(JEV_SETTINGS, config_with_policy(tmp_path, policy), client=client)
        result = await evaluator.evaluate("q", "a", threshold=0.8)
        await client.aclose()

        assert result.status == "unavailable"
        assert result.score is None


# ---------------------------------------------------------------------- engine

@pytest.mark.asyncio
async def test_escalate_policy_moves_to_the_next_level_when_jev_is_down(tmp_path):
    config = config_with_policy(tmp_path, "escalate")
    provider = RecordingProvider(["first answer", "second answer"])
    engine = CascadeEngine(Settings(), config, provider=provider,
                           evaluator=down_evaluator(config))

    result = await engine.run([{"role": "user", "content": "Question"}])

    # Both evaluated levels were unavailable, so the request reached the final tier.
    assert provider.models == ["openai/model-one", "anthropic/model-two", "anthropic/model-three"]
    assert result.model_used == "anthropic/model-three"
    assert result.jev_score is None
    assert result.jev_status == "unavailable"


@pytest.mark.asyncio
async def test_fail_open_policy_keeps_the_first_answer(tmp_path):
    config = config_with_policy(tmp_path, "fail_open")
    provider = RecordingProvider(["first answer", "second answer"])
    engine = CascadeEngine(Settings(), config, provider=provider,
                           evaluator=down_evaluator(config))

    result = await engine.run([{"role": "user", "content": "Question"}])

    assert provider.models == ["openai/model-one"]
    assert result.jev_score is None
    assert result.jev_status == "unavailable"


@pytest.mark.asyncio
async def test_fail_closed_policy_aborts_without_a_second_provider_call(tmp_path):
    config = config_with_policy(tmp_path, "fail_closed")
    provider = RecordingProvider(["first answer", "second answer"])
    engine = CascadeEngine(Settings(), config, provider=provider,
                           evaluator=down_evaluator(config))

    with pytest.raises(JevUnavailable):
        await engine.run([{"role": "user", "content": "Question"}])

    assert provider.models == ["openai/model-one"]


@pytest.mark.asyncio
async def test_successful_evaluation_reports_its_status(cascade_config):
    engine = CascadeEngine(Settings(), cascade_config,
                           provider=RecordingProvider(["first answer"]),
                           evaluator=scored_evaluator(cascade_config, 0.95))

    result = await engine.run([{"role": "user", "content": "Question"}])

    assert result.jev_status == "approved"
    assert result.jev_score == pytest.approx(0.95)


@pytest.mark.asyncio
async def test_a_rejected_candidate_reports_the_rejected_status(cascade_config):
    provider = RecordingProvider(["weak answer", "better answer"])
    engine = CascadeEngine(Settings(), cascade_config, provider=provider,
                           evaluator=scored_evaluator(cascade_config, 0.10))

    result = await engine.run([{"role": "user", "content": "Question"}])

    # Both evaluated candidates were rejected; the last evaluation decides the status.
    assert provider.models == ["openai/model-one", "anthropic/model-two", "anthropic/model-three"]
    assert result.jev_status == "rejected"
    assert result.jev_score == pytest.approx(0.10)


# ------------------------------------------------------------------- telemetry

@pytest.mark.asyncio
async def test_telemetry_logs_a_null_score_and_the_unavailable_status(tmp_path, caplog):
    config = config_with_policy(tmp_path, "escalate")
    engine = CascadeEngine(Settings(), config,
                           provider=RecordingProvider(["first answer", "second answer"]),
                           evaluator=down_evaluator(config))

    with caplog.at_level(logging.INFO, logger="ai_cascade_proxy.telemetry"):
        await engine.run([{"role": "user", "content": "Question"}])

    event = json.loads(caplog.records[-1].getMessage())
    assert event["jev_score"] is None
    assert event["jev_status"] == "unavailable"


@pytest.mark.asyncio
async def test_telemetry_logs_the_approved_status_for_a_scored_evaluation(cascade_config, caplog):
    engine = CascadeEngine(Settings(), cascade_config,
                           provider=RecordingProvider(["first answer"]),
                           evaluator=scored_evaluator(cascade_config, 0.95))

    with caplog.at_level(logging.INFO, logger="ai_cascade_proxy.telemetry"):
        await engine.run([{"role": "user", "content": "Question"}])

    event = json.loads(caplog.records[-1].getMessage())
    assert event["jev_status"] == "approved"
    assert event["jev_score"] == pytest.approx(0.95)


# --------------------------------------------------------------------- metrics

def test_fallback_rate_counts_evaluations_without_a_score():
    metrics = Metrics()

    metrics.record_jev_evaluation(status="approved")
    metrics.record_jev_evaluation(status="unavailable")
    metrics.record_jev_evaluation(status="unavailable")
    snapshot = metrics.snapshot()

    assert snapshot["jev_evaluations_total"] == 3
    assert snapshot["jev_unavailable_total"] == 2
    assert snapshot["jev_fallback_rate"] == pytest.approx(2 / 3)


def test_fallback_rate_is_zero_not_undefined_before_any_evaluation():
    snapshot = Metrics().snapshot()

    assert snapshot["jev_evaluations_total"] == 0
    assert snapshot["jev_fallback_rate"] == 0.0


@pytest.mark.asyncio
async def test_the_engine_records_every_evaluation_in_the_metrics(tmp_path):
    metrics = Metrics()
    config = config_with_policy(tmp_path, "escalate")
    engine = CascadeEngine(Settings(), config,
                           provider=RecordingProvider(["first answer", "second answer"]),
                           evaluator=down_evaluator(config), metrics=metrics)

    await engine.run([{"role": "user", "content": "Question"}])

    snapshot = metrics.snapshot()
    # Levels 1 and 2 are both evaluated and both unavailable.
    assert snapshot["jev_evaluations_total"] == 2
    assert snapshot["jev_unavailable_total"] == 2
    assert snapshot["jev_fallback_rate"] == pytest.approx(1.0)
    assert snapshot["requests_total"] == 1
    assert snapshot["escalations_total"] == 1


@pytest.mark.asyncio
async def test_metrics_survive_many_requests(cascade_config):
    metrics = Metrics()
    engine = CascadeEngine(Settings(), cascade_config,
                           provider=RecordingProvider(["answer"] * 5),
                           evaluator=scored_evaluator(cascade_config, 0.99), metrics=metrics)

    for _ in range(5):
        await engine.run([{"role": "user", "content": "Question"}])

    snapshot = metrics.snapshot()
    assert snapshot["jev_evaluations_total"] == 5
    assert snapshot["jev_unavailable_total"] == 0


# ------------------------------------------------------------------------- API

def app_for(tmp_path, policy: str, *, jev_score: float | None, settings=None):
    """App wired to the real evaluator: ``jev_score=None`` means JEV is down."""
    config = config_with_policy(tmp_path, policy)
    evaluator = down_evaluator(config) if jev_score is None else scored_evaluator(config, jev_score)
    return TestClient(
        create_app(
            settings=settings or Settings(),
            cascade_config=config,
            provider=RecordingProvider(["first answer", "second answer"]),
            evaluator=evaluator,
        )
    )


def test_metrics_endpoint_exposes_the_fallback_rate(tmp_path):
    with app_for(tmp_path, "escalate", jev_score=None) as client:
        client.post("/v1/chat/completions",
                    json={"model": "auto", "messages": [{"role": "user", "content": "Hi"}]})
        response = client.get("/metrics")

    assert response.status_code == 200
    body = response.json()
    assert body["jev_evaluations_total"] == 2
    assert body["jev_unavailable_total"] == 2
    assert body["jev_fallback_rate"] == pytest.approx(1.0)
    assert body["requests_total"] == 1


def test_metrics_endpoint_requires_the_client_api_key_when_configured(tmp_path):
    with app_for(tmp_path, "escalate", jev_score=0.95,
                 settings=Settings(api_key="client-secret")) as client:
        missing = client.get("/metrics")
        ok = client.get("/metrics", headers={"Authorization": "Bearer client-secret"})

    assert missing.status_code == 401
    assert ok.status_code == 200


def test_fail_closed_returns_an_openai_error_and_no_answer(tmp_path):
    with app_for(tmp_path, "fail_closed", jev_score=None) as client:
        response = client.post("/v1/chat/completions",
                               json={"model": "auto",
                                     "messages": [{"role": "user", "content": "Hi"}]})
        metrics = client.get("/metrics").json()

    assert response.status_code == 503
    error = response.json()["error"]
    assert error["type"] == "service_unavailable"
    assert error["code"] == "jev_unavailable"
    assert "choices" not in response.json()
    assert metrics["jev_unavailable_total"] == 1


def test_fail_open_returns_the_first_answer_when_jev_is_down(tmp_path):
    with app_for(tmp_path, "fail_open", jev_score=None) as client:
        response = client.post("/v1/chat/completions",
                               json={"model": "auto",
                                     "messages": [{"role": "user", "content": "Hi"}]})

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "first answer"
