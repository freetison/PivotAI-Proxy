"""Phase 5: payloads the cascade cannot reason about are bypassed, not evaluated.

`tools`, `tool_choice`, `response_format`, image content and `n > 1` change the
meaning of a response (tool calls, enforced schemas, multimodal input, multiple
choices). JEV scores a single text answer, so scoring any of these would produce
a meaningless decision. They are therefore sent straight to one configured level
and never evaluated; the response says so with `X-Cascade-Bypass`.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from src.cascade_config import CascadeConfig
from src.cascade_engine import CascadeEngine
from src.cascade_engine import UpstreamCascadeFailed  # noqa: F401  (imported for parity)
from src.jev_evaluator import EvaluationResult
from src.main import create_app
from src.payload import BYPASS_HEADER, detect_unsupported
from src.settings import Settings

from tests.conftest import base_config_dict, load_config, write_config_data

L1, L2, L3 = "openai/model-one", "anthropic/model-two", "anthropic/model-three"

TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
IMAGE_CONTENT = [
    {"type": "text", "text": "What is this?"},
    {"type": "image_url", "image_url": {"url": "https://example.test/cat.png"}},
]


def config(tmp_path) -> CascadeConfig:
    return load_config(write_config_data(tmp_path, base_config_dict()))


def completion(text: str) -> dict:
    return {
        "id": "provider-id",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


class RecordingProvider:
    def __init__(self, text: str = "answer"):
        self.text = text
        self.calls: list[tuple[str, dict]] = []

    async def complete(self, model, messages, timeout_ms=None, **kwargs):
        self.calls.append((model, kwargs))
        return completion(self.text)

    @property
    def models(self) -> list[str]:
        return [model for model, _ in self.calls]


class CountingEvaluator:
    def __init__(self):
        self.calls = 0

    async def evaluate(self, prompt, response, *, threshold=None, system=""):
        self.calls += 1
        return EvaluationResult(approved=True, score=0.99, status="approved")


def payload(**overrides) -> dict:
    body = {"model": "auto", "messages": [{"role": "user", "content": "Hi"}]}
    body.update(overrides)
    return body


# ------------------------------------------------------------------ detection

def test_a_plain_payload_needs_no_bypass():
    assert detect_unsupported(payload()) == ()


def test_tools_are_detected():
    assert detect_unsupported(payload(tools=TOOLS)) == ("tools",)


def test_tool_choice_alone_is_detected():
    assert detect_unsupported(payload(tool_choice="auto")) == ("tools",)


def test_an_empty_tools_list_is_not_a_tool_request():
    # Some clients send `tools: []`; that is not a tool-calling payload.
    assert detect_unsupported(payload(tools=[])) == ()


def test_response_format_is_detected():
    assert detect_unsupported(payload(response_format={"type": "json_object"})) == ("response_format",)


def test_image_content_is_detected():
    body = payload(messages=[{"role": "user", "content": IMAGE_CONTENT}])

    assert detect_unsupported(body) == ("image_content",)


def test_plain_text_blocks_are_not_multimodal():
    body = payload(messages=[{"role": "user", "content": [{"type": "text", "text": "Hi"}]}])

    assert detect_unsupported(body) == ()


def test_multiple_choices_are_detected():
    assert detect_unsupported(payload(n=2)) == ("choices",)


def test_a_single_choice_is_the_normal_case():
    assert detect_unsupported(payload(n=1)) == ()


def test_every_reason_is_reported_once_in_a_stable_order():
    body = payload(tools=TOOLS, response_format={"type": "json_object"}, n=3,
                   messages=[{"role": "user", "content": IMAGE_CONTENT}])

    assert detect_unsupported(body) == ("tools", "response_format", "image_content", "choices")


# ---------------------------------------------------------------- configuration

def test_the_bypass_level_defaults_to_the_final_level(monkeypatch):
    monkeypatch.delenv("UNSUPPORTED_FEATURES_LEVEL", raising=False)

    assert Settings.from_env().unsupported_features_level == 3


@pytest.mark.parametrize("raw", ["1", "2", "3"])
def test_the_bypass_level_is_configurable(monkeypatch, raw):
    monkeypatch.setenv("UNSUPPORTED_FEATURES_LEVEL", raw)

    assert Settings.from_env().unsupported_features_level == int(raw)


@pytest.mark.parametrize("raw", ["0", "4", "nine"])
def test_an_invalid_bypass_level_fails_at_startup(monkeypatch, raw):
    monkeypatch.setenv("UNSUPPORTED_FEATURES_LEVEL", raw)

    with pytest.raises(ValueError, match="UNSUPPORTED_FEATURES_LEVEL"):
        Settings.from_env()


# ------------------------------------------------------------------------ engine

@pytest.mark.asyncio
async def test_the_bypass_calls_exactly_one_level_and_never_evaluates(tmp_path):
    provider = RecordingProvider("tool answer")
    evaluator = CountingEvaluator()
    engine = CascadeEngine(Settings(), config(tmp_path), provider=provider, evaluator=evaluator)

    result = await engine.run_bypass([{"role": "user", "content": "Hi"}], level_number=3)

    assert provider.models == [L3]
    assert evaluator.calls == 0
    assert result.hops_count == 1
    assert result.jev_score is None
    assert result.jev_status is None


@pytest.mark.asyncio
async def test_the_bypass_forwards_the_client_parameters(tmp_path):
    provider = RecordingProvider()
    engine = CascadeEngine(Settings(), config(tmp_path), provider=provider,
                           evaluator=CountingEvaluator())

    await engine.run_bypass([{"role": "user", "content": "Hi"}], level_number=3,
                            tools=TOOLS, n=2)

    model, kwargs = provider.calls[0]
    assert kwargs["tools"] == TOOLS
    assert kwargs["n"] == 2


@pytest.mark.asyncio
async def test_the_bypass_ignores_the_cascade_even_for_an_unsupported_model(tmp_path):
    provider = RecordingProvider()
    engine = CascadeEngine(Settings(), config(tmp_path), provider=provider,
                           evaluator=CountingEvaluator())

    # `model: "auto"` normally means "run the cascade"; a bypass always uses one level.
    await engine.run_bypass([{"role": "user", "content": "Hi"}], level_number=2)

    assert provider.models == [L2]


# ------------------------------------------------------------------------- the API

def app_for(tmp_path, provider, *, evaluator=None, settings=None) -> TestClient:
    cascade = config(tmp_path)
    engine = CascadeEngine(settings or Settings(), cascade, provider=provider,
                           evaluator=evaluator or CountingEvaluator(), retry_backoff_ms=0)
    return TestClient(create_app(settings=settings or Settings(), cascade_config=cascade,
                                 engine=engine))


@pytest.mark.parametrize(
    "overrides, reason",
    [
        ({"tools": TOOLS}, "tools"),
        ({"tool_choice": "required"}, "tools"),
        ({"response_format": {"type": "json_object"}}, "response_format"),
        ({"n": 2}, "choices"),
        ({"messages": [{"role": "user", "content": IMAGE_CONTENT}]}, "image_content"),
    ],
)
def test_each_unsupported_feature_bypasses_the_cascade(tmp_path, overrides, reason):
    provider = RecordingProvider()
    evaluator = CountingEvaluator()
    with app_for(tmp_path, provider, evaluator=evaluator) as client:
        response = client.post("/v1/chat/completions", json=payload(**overrides))

    assert response.status_code == 200
    assert response.headers[BYPASS_HEADER] == reason
    assert provider.models == [L3]
    assert evaluator.calls == 0


def test_several_reasons_are_joined_in_the_header(tmp_path):
    provider = RecordingProvider()
    with app_for(tmp_path, provider) as client:
        response = client.post("/v1/chat/completions",
                              json=payload(tools=TOOLS, n=2))

    assert response.headers[BYPASS_HEADER] == "tools,choices"


def test_a_normal_request_has_no_bypass_header(tmp_path):
    provider = RecordingProvider()
    with app_for(tmp_path, provider) as client:
        response = client.post("/v1/chat/completions", json=payload())

    assert BYPASS_HEADER not in response.headers
    assert provider.models == [L1]  # the cascade started at level 1


def test_the_bypass_level_is_configurable_per_deployment(tmp_path):
    provider = RecordingProvider()
    settings = Settings(unsupported_features_level=1)
    with app_for(tmp_path, provider, settings=settings) as client:
        response = client.post("/v1/chat/completions", json=payload(tools=TOOLS))

    assert response.status_code == 200
    assert provider.models == [L1]


def test_a_failed_bypass_does_not_escalate(tmp_path):
    class FailingProvider(RecordingProvider):
        async def complete(self, model, messages, timeout_ms=None, **kwargs):
            from src.provider import ProviderError
            self.calls.append((model, kwargs))
            raise ProviderError("boom", status_code=502, provider="openai",
                                upstream_status=502)

    provider = FailingProvider()
    with app_for(tmp_path, provider) as client:
        response = client.post("/v1/chat/completions", json=payload(tools=TOOLS))

    # No JEV, no escalation: the bypassed level is the only answer available.
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "provider_error"
    assert provider.models == [L3]


def test_a_bypass_is_reported_in_telemetry(tmp_path, caplog):
    provider = RecordingProvider()
    import logging
    with app_for(tmp_path, provider) as client:
        with caplog.at_level(logging.INFO, logger="ai_cascade_proxy.telemetry"):
            client.post("/v1/chat/completions", json=payload(tools=TOOLS))

    event = json.loads(caplog.records[-1].getMessage())
    assert event["hops_count"] == 1
    assert event["jev_score"] is None
    assert event["jev_status"] is None
    assert event["model_used"] == L3


def test_a_bypass_still_streams_when_asked(tmp_path):
    provider = RecordingProvider("streamed answer")
    with app_for(tmp_path, provider) as client:
        with client.stream("POST", "/v1/chat/completions",
                           json=payload(tools=TOOLS, stream=True)) as response:
            body = "".join(response.iter_text())

    assert response.headers[BYPASS_HEADER] == "tools"
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "streamed answer" in body
    assert body.rstrip().endswith("data: [DONE]")


def test_an_unknown_bypass_level_is_a_configuration_error_not_a_silent_level(tmp_path):
    provider = RecordingProvider()
    engine = CascadeEngine(Settings(), config(tmp_path), provider=provider,
                           evaluator=CountingEvaluator())

    with pytest.raises(ValueError, match="Unsupported cascade level: 7"):
        engine.level_by_number(7)


@pytest.mark.parametrize("block_type", ["input_image", "file", "input_file", "audio", "input_audio"])
def test_every_non_text_block_type_bypasses(block_type):
    body = payload(messages=[{"role": "user",
                              "content": [{"type": block_type, "url": "https://x.test/a"}]}])

    assert detect_unsupported(body) == ("image_content",)


def test_a_malformed_content_block_is_ignored_rather_than_raising():
    body = payload(messages=[{"role": "user", "content": ["just a string", 42, {"type": "text", "text": "hi"}]}])

    assert detect_unsupported(body) == ()


def test_messages_that_are_not_a_list_do_not_crash_detection():
    assert detect_unsupported({"messages": "not-a-list"}) == ()
