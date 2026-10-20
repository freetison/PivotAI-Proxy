import json
import logging

from fastapi.testclient import TestClient

from src.jev_evaluator import EvaluationResult
from src.main import create_app
from src.settings import Settings


class FakeProvider:
    def __init__(self):
        self.calls = []

    async def complete(self, model, messages, **kwargs):
        self.calls.append(model)
        return {
            "id": "provider-id",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }


class FakeEvaluator:
    async def evaluate(self, prompt, response, *, threshold=None, system=""):
        return EvaluationResult(approved=True, score=0.95)


def make_client(cascade_config):
    settings = Settings(api_key="client-secret")
    return TestClient(
        create_app(
            settings=settings,
            cascade_config=cascade_config,
            provider=FakeProvider(),
            evaluator=FakeEvaluator(),
        )
    )


def test_openai_valid_payload_structure(cascade_config):
    with make_client(cascade_config) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer client-secret"},
            json={"model": "auto", "messages": [{"role": "user", "content": "Hi"}]},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["id"].startswith("chatcmpl-")
    assert body["object"] == "chat.completion"
    assert isinstance(body["created"], int)
    assert body["model"] == "openai/model-one"
    assert body["choices"][0]["message"]["content"] == "Hello!"
    assert body["usage"]["total_tokens"] == 5


def test_models_endpoint_derives_from_config(cascade_config):
    with make_client(cascade_config) as client:
        response = client.get("/v1/models", headers={"Authorization": "Bearer client-secret"})

    assert response.status_code == 200
    assert response.json()["object"] == "list"
    assert [model["id"] for model in response.json()["data"]] == [
        "auto",
        "openai/model-one",
        "anthropic/model-two",
        "anthropic/model-three",
    ]


def test_api_key_is_required_and_validated(cascade_config):
    with make_client(cascade_config) as client:
        missing = client.get("/v1/models")
        invalid = client.get("/v1/models", headers={"Authorization": "Bearer wrong"})

    assert missing.status_code == 401
    assert invalid.status_code == 401
    assert missing.json()["error"]["type"] == "authentication_error"


def test_unsupported_model_returns_openai_error(cascade_config):
    with make_client(cascade_config) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer client-secret"},
            json={"model": "not-configured/model", "messages": [{"role": "user", "content": "Hi"}]},
        )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_invalid_chat_payload_returns_openai_error(cascade_config):
    with make_client(cascade_config) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer client-secret"},
            json={"model": "auto", "messages": []},
        )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_telemetry_and_cost_logging(cascade_config, caplog):
    with caplog.at_level(logging.INFO, logger="ai_cascade_proxy.telemetry"):
        with make_client(cascade_config) as client:
            response = client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer client-secret"},
                json={"model": "auto", "messages": [{"role": "user", "content": "Hi"}]},
            )

    assert response.status_code == 200
    records = [record.getMessage() for record in caplog.records]
    telemetry = next(json.loads(message) for message in records if '"model_used"' in message)
    assert telemetry["model_used"] == "openai/model-one"
    assert telemetry["hops_count"] == 1
    assert telemetry["jev_score"] == 0.95
    assert telemetry["total_latency_ms"] >= 0
    assert telemetry["cost_usd"] >= 0
    assert telemetry["baseline_cost_usd"] >= 0
    assert "estimated_savings_pct" not in telemetry
    assert "prompt" not in telemetry
