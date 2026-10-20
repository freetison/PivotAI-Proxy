"""One live request through the whole proxy, against real models and a real JEV.

This is the smoke test the README describes but nothing executed: it answers the
only question the unit suite cannot — does this configuration actually work when
the network is real? It asserts on *shape and accounting*, never on the content
of an answer, because a model's wording is not a contract.

See ``tests/integration/conftest.py`` for how to opt in.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.cascade_config import CascadeConfig
from src.jev_evaluator import JevEvaluator
from src.main import create_app
from src.provider import ModelProvider
from src.settings import Settings

pytestmark = pytest.mark.integration


def live_app(settings: Settings, config: CascadeConfig) -> TestClient:
    """Build the app exactly as the proxy does, with real collaborators."""
    provider = ModelProvider(settings)
    evaluator = JevEvaluator(settings, config)
    return TestClient(create_app(settings, config, provider=provider, evaluator=evaluator))


def test_a_real_request_is_answered_by_a_configured_level(live_settings, live_config, live_payload):
    app = live_app(live_settings, live_config)

    response = app.post("/v1/chat/completions", json=live_payload,
                        headers=_auth(live_settings))

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["choices"][0]["message"]["content"].strip()
    assert body["model"] in live_config.model_names


def test_the_telemetry_prices_the_real_tokens(live_settings, live_config, live_payload, caplog):
    """The cost must come from the usage the provider reported, not from a guess."""
    import json
    import logging

    app = live_app(live_settings, live_config)
    caplog.set_level(logging.INFO, logger="ai_cascade_proxy.telemetry")

    response = app.post("/v1/chat/completions", json=live_payload,
                        headers=_auth(live_settings))
    assert response.status_code == 200, response.text

    events = [json.loads(record.getMessage()) for record in caplog.records]
    events = [event for event in events if "cost_usd" in event]
    assert events, "the proxy emitted no telemetry event for a completed request"

    event = events[-1]
    assert event["cost_usd"] is not None and event["cost_usd"] > 0
    assert event["outbound_calls"] >= 1
    assert event["hops_count"] >= 1


def test_the_reported_models_are_the_configured_ones(live_settings, live_config):
    app = live_app(live_settings, live_config)

    response = app.get("/v1/models", headers=_auth(live_settings))

    assert response.status_code == 200, response.text
    reported = {item["id"] for item in response.json()["data"]}
    assert reported == set(live_config.model_names)


def test_the_listed_models_exist_at_the_provider(live_settings, live_config):
    """`VALIDATE_MODELS_ON_STARTUP` exists because a typo in a model name is silent.

    A level naming a model the provider does not serve fails every request routed
    to it, and nothing in a mocked test suite can notice. This is the test that
    notices.
    """
    from src.cascade_config import validate_models_against_providers

    provider = ModelProvider(live_settings)
    try:
        import asyncio

        asyncio.run(validate_models_against_providers(live_config, provider))
    except Exception as exc:  # noqa: BLE001 - a validated failure is the finding
        pytest.fail(f"the cascade names a model the provider does not serve: {exc}")


def test_streaming_ends_with_the_done_sentinel(live_settings, live_config, live_payload):
    app = live_app(live_settings, live_config)

    with app.stream("POST", "/v1/chat/completions", json={**live_payload, "stream": True},
                    headers=_auth(live_settings)) as response:
        assert response.status_code == 200

        body = b"".join(response.iter_bytes()).decode("utf-8")

    assert body.rstrip().endswith("data: [DONE]")
    assert "chat.completion.chunk" in body


def test_an_unauthenticated_request_is_refused_before_anything_is_spent(live_settings, live_config):
    """The cheapest possible failure: rejected at the door, no provider call."""
    if not live_settings.client_keys:
        pytest.skip("no client keys configured; every caller is anonymous in this mode")

    app = live_app(live_settings, live_config)

    response = app.post("/v1/chat/completions", json={"model": "auto",
                                                     "messages": [{"role": "user", "content": "hi"}]})

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "authentication_error"


def _auth(settings: Settings) -> dict[str, str]:
    if not settings.client_keys:
        return {}
    _label, key = settings.client_keys[0]
    return {"Authorization": f"Bearer {key}"}


def test_a_health_check_needs_no_credential_and_no_money(live_settings, live_config):
    """Liveness must not depend on a client key."""
    app = live_app(live_settings, live_config)

    response = app.get("/health")

    assert response.status_code == 200
