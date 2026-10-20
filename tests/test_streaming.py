"""Phase 4: `stream: true`.

Two modes, both documented in the README:

* ``buffered_cascade`` (default): the cascade runs exactly as for a normal request,
  the winning answer is evaluated by JEV, and only then is it replayed as an
  OpenAI chunk stream. The client waits for the whole cascade, so time-to-first-token
  equals the cascade latency, but nothing unevaluated ever reaches the wire.
* ``passthrough_l3``: the final level is streamed directly, with no JEV evaluation.
  TTFT matches the provider's, at the cost of skipping evaluation entirely.

The chunk format is the one clients already parse: a role delta, content deltas,
a finish delta, and ``data: [DONE]``.
"""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from src.cascade_config import CascadeConfig
from src.cascade_engine import CascadeEngine
from src.jev_evaluator import EvaluationResult
from src.main import create_app
from src.provider import ModelProvider
from src.settings import STREAM_MODES, Settings
from src.streaming import buffered_stream, sse_done, sse_event

from tests.conftest import base_config_dict, load_config, write_config_data

L1, L2, L3 = "openai/model-one", "anthropic/model-two", "anthropic/model-three"


def config(tmp_path) -> CascadeConfig:
    return load_config(write_config_data(tmp_path, base_config_dict()))


def completion(text: str) -> dict:
    return {
        "id": "provider-id",
        "model": "whatever",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


class ScriptedProvider:
    def __init__(self, script: dict[str, list]):
        self.script = {model: list(outcomes) for model, outcomes in script.items()}
        self.calls: list[str] = []

    async def complete(self, model, messages, timeout_ms=None, **kwargs):
        self.calls.append(model)
        outcomes = self.script.get(model) or [completion("default")]
        outcome = outcomes.pop(0) if len(outcomes) > 1 else outcomes[0]
        if isinstance(outcome, dict):
            return outcome
        from src.provider import ProviderError
        raise ProviderError(
            f"Upstream provider {model.split('/')[0]} returned an error",
            status_code=outcome, provider=model.split("/")[0],
            upstream_status=outcome, retryable=outcome == 429,
        )


class ApprovingEvaluator:
    def __init__(self):
        self.calls = 0

    async def evaluate(self, prompt, response, *, threshold=None, system="", **kwargs):
        self.calls += 1
        return EvaluationResult(approved=True, score=0.99, status="approved")


def parse_events(body: str) -> list:
    """Split an SSE body into payloads; the terminator is returned as the string '[DONE]'."""
    events = []
    for block in body.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        assert block.startswith("data: "), f"unexpected SSE frame: {block!r}"
        payload = block[len("data: "):]
        events.append("[DONE]" if payload == "[DONE]" else json.loads(payload))
    return events


# ------------------------------------------------------------------ chunk building

def test_the_terminator_is_exactly_the_openai_one():
    assert sse_done() == "data: [DONE]\n\n"


def test_a_chunk_is_a_single_data_frame_with_compact_json():
    frame = sse_event({"a": 1, "b": "x"})

    assert frame == 'data: {"a":1,"b":"x"}\n\n'


@pytest.mark.asyncio
async def test_the_buffered_stream_emits_role_content_finish_and_done(tmp_path):
    provider = ScriptedProvider({L1: [completion("Hello there")]})
    engine = CascadeEngine(Settings(), config(tmp_path), provider=provider,
                           evaluator=ApprovingEvaluator())
    result = await engine.run([{"role": "user", "content": "Hi"}])

    frames = [frame async for frame in buffered_stream(result, "auto")]
    events = parse_events("".join(frames))

    assert events[-1] == "[DONE]"
    chunk_id = events[0]["id"]
    created = events[0]["created"]
    assert events[0] == {
        "id": chunk_id, "object": "chat.completion.chunk", "created": created,
        "model": L1,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    assert events[1] == {
        "id": chunk_id, "object": "chat.completion.chunk", "created": created,
        "model": L1,
        "choices": [{"index": 0, "delta": {"content": "Hello there"}, "finish_reason": None}],
    }
    assert events[2]["choices"] == [{"index": 0, "delta": {}, "finish_reason": "stop"}]
    # Every chunk shares one id and timestamp: clients group them by request.
    assert {event["id"] for event in events[:-1]} == {chunk_id}
    assert {event["created"] for event in events[:-1]} == {created}


@pytest.mark.asyncio
async def test_a_final_level_answer_streams_with_its_own_model_name(tmp_path):
    provider = ScriptedProvider({L1: [completion("weak")], L2: [completion("still weak")],
                                 L3: [completion("frontier answer")]})
    rejecting = type("Rejecting", (), {
        "calls": 0,
        "evaluate": lambda self, p, r, *, threshold=None, system="": _rejected(),
    })()
    engine = CascadeEngine(Settings(), config(tmp_path), provider=provider, evaluator=rejecting)

    result = await engine.run([{"role": "user", "content": "Hi"}])
    events = parse_events("".join([frame async for frame in buffered_stream(result, "auto")]))

    assert result.model_used == L3
    assert events[1]["choices"][0]["delta"]["content"] == "frontier answer"
    assert events[1]["model"] == L3


async def _rejected():
    return EvaluationResult(approved=False, score=0.0, status="rejected")


# ----------------------------------------------------------------- configuration

@pytest.mark.parametrize("mode", STREAM_MODES)
def test_every_documented_mode_is_selectable(monkeypatch, mode):
    monkeypatch.setenv("STREAM_MODE", mode)

    assert Settings.from_env().stream_mode == mode


def test_the_default_mode_buffers_the_cascade(monkeypatch):
    monkeypatch.delenv("STREAM_MODE", raising=False)

    assert Settings.from_env().stream_mode == "buffered_cascade"


def test_an_unknown_mode_fails_at_startup(monkeypatch):
    monkeypatch.setenv("STREAM_MODE", "chunked_everything")

    with pytest.raises(ValueError, match="STREAM_MODE"):
        Settings.from_env()


# ------------------------------------------------------------------------ the API

def app_for(tmp_path, provider, *, evaluator=None, settings=None) -> TestClient:
    cascade = config(tmp_path)
    engine = CascadeEngine(settings or Settings(), cascade, provider=provider,
                           evaluator=evaluator or ApprovingEvaluator(), retry_backoff_ms=0)
    return TestClient(create_app(settings=settings or Settings(), cascade_config=cascade,
                                 engine=engine))


def test_streaming_returns_event_stream_and_the_buffered_chunks(tmp_path):
    provider = ScriptedProvider({L1: [completion("Hello there")]})
    with app_for(tmp_path, provider) as client:
        with client.stream("POST", "/v1/chat/completions",
                           json={"model": "auto", "stream": True,
                                 "messages": [{"role": "user", "content": "Hi"}]}) as response:
            body = "".join(response.iter_text())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_events(body)
    assert events[-1] == "[DONE]"
    assert events[1]["choices"][0]["delta"]["content"] == "Hello there"


def test_a_buffered_stream_replays_the_answer_jev_approved(tmp_path):
    evaluator = ApprovingEvaluator()
    provider = ScriptedProvider({L1: [completion("Hello there")]})
    with app_for(tmp_path, provider, evaluator=evaluator) as client:
        client.post("/v1/chat/completions",
                    json={"model": "auto", "stream": True,
                          "messages": [{"role": "user", "content": "Hi"}]})

    # The default mode is a cascade: the answer was evaluated before being sent.
    assert evaluator.calls == 1


def test_a_failing_cascade_still_returns_a_json_error_not_a_broken_stream(tmp_path):
    provider = ScriptedProvider({L1: [503], L2: [503], L3: [503]})
    with app_for(tmp_path, provider) as client:
        response = client.post("/v1/chat/completions",
                               json={"model": "auto", "stream": True,
                                     "messages": [{"role": "user", "content": "Hi"}]})

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "cascade_failed"


def test_a_non_streaming_request_is_unaffected(tmp_path):
    provider = ScriptedProvider({L1: [completion("Hello there")]})
    with app_for(tmp_path, provider) as client:
        response = client.post("/v1/chat/completions",
                               json={"model": "auto",
                                     "messages": [{"role": "user", "content": "Hi"}]})

    assert response.status_code == 200
    assert response.json()["object"] == "chat.completion"


# ---------------------------------------------------------------- passthrough_l3

UPSTREAM_SSE = (
    'data: {"id":"up-1","object":"chat.completion.chunk","created":1,"model":"model-three",'
    '"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
    'data: {"id":"up-1","object":"chat.completion.chunk","created":1,"model":"model-three",'
    '"choices":[{"index":0,"delta":{"content":"direct "},"finish_reason":null}]}\n\n'
    'data: {"id":"up-1","object":"chat.completion.chunk","created":1,"model":"model-three",'
    '"choices":[{"index":0,"delta":{"content":"from three"},"finish_reason":"stop"}]}\n\n'
    "data: [DONE]\n\n"
)


def streaming_settings() -> Settings:
    return Settings(stream_mode="passthrough_l3")


def openai_provider(handler) -> ModelProvider:
    """A provider whose final level is OpenAI-compatible, so passthrough is allowed."""
    return ModelProvider(
        Settings(openai_base_url="https://provider.test/v1",
                 provider_base_urls={"openai": "https://provider.test/v1"},
                 openai_api_key="k"),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def passthrough_config(tmp_path) -> CascadeConfig:
    """A cascade whose last level is an OpenAI-compatible provider (streamable)."""
    payload = base_config_dict()
    payload["cascade"]["levels"][2]["provider"] = "openai"
    payload["cascade"]["levels"][2]["model"] = "model-three"
    return load_config(write_config_data(tmp_path, payload))


def test_passthrough_forwards_the_upstream_frames_verbatim(tmp_path):
    evaluator = ApprovingEvaluator()

    async def handler(request):
        assert request.url == "https://provider.test/v1/chat/completions"
        assert json.loads(request.read())["stream"] is True
        return httpx.Response(200, text=UPSTREAM_SSE,
                              headers={"content-type": "text/event-stream"})

    provider = openai_provider(handler)
    engine = CascadeEngine(streaming_settings(), passthrough_config(tmp_path),
                           provider=provider, evaluator=evaluator)
    with TestClient(create_app(settings=streaming_settings(),
                               cascade_config=passthrough_config(tmp_path),
                               engine=engine)) as client:
        with client.stream("POST", "/v1/chat/completions",
                           json={"model": "auto", "stream": True,
                                 "messages": [{"role": "user", "content": "Hi"}]}) as response:
            body = "".join(response.iter_text())

    assert response.status_code == 200
    assert body == UPSTREAM_SSE
    # Passthrough means no evaluation at all: that is the trade being made.
    assert evaluator.calls == 0


def test_passthrough_adds_the_terminator_when_the_provider_omits_it(tmp_path):
    async def handler(request):
        return httpx.Response(200, text=UPSTREAM_SSE.replace("data: [DONE]\n\n", ""))

    provider = openai_provider(handler)
    engine = CascadeEngine(streaming_settings(), passthrough_config(tmp_path),
                           provider=provider, evaluator=ApprovingEvaluator())
    with TestClient(create_app(settings=streaming_settings(),
                               cascade_config=passthrough_config(tmp_path),
                               engine=engine)) as client:
        with client.stream("POST", "/v1/chat/completions",
                           json={"model": "auto", "stream": True,
                                 "messages": [{"role": "user", "content": "Hi"}]}) as response:
            body = "".join(response.iter_text())

    assert parse_events(body)[-1] == "[DONE]"


def test_passthrough_reports_an_upstream_error_as_a_json_error(tmp_path):
    async def handler(request):
        return httpx.Response(429, json={"error": "slow down"})

    provider = openai_provider(handler)
    engine = CascadeEngine(streaming_settings(), passthrough_config(tmp_path),
                           provider=provider, evaluator=ApprovingEvaluator())
    with TestClient(create_app(settings=streaming_settings(),
                               cascade_config=passthrough_config(tmp_path),
                               engine=engine)) as client:
        response = client.post("/v1/chat/completions",
                               json={"model": "auto", "stream": True,
                                     "messages": [{"role": "user", "content": "Hi"}]})

    # The failure happens before any byte of body, so a normal error is still possible.
    assert response.status_code == 429
    assert response.json()["error"]["code"] == "provider_error"


def test_passthrough_refuses_a_final_level_that_cannot_stream_in_openai_format(tmp_path):
    async def handler(request):  # pragma: no cover - never reached
        return httpx.Response(200, text=UPSTREAM_SSE)

    provider = openai_provider(handler)
    engine = CascadeEngine(streaming_settings(), config(tmp_path),
                           provider=provider, evaluator=ApprovingEvaluator())
    with TestClient(create_app(settings=streaming_settings(), cascade_config=config(tmp_path),
                               engine=engine)) as client:
        response = client.post("/v1/chat/completions",
                               json={"model": "auto", "stream": True,
                                     "messages": [{"role": "user", "content": "Hi"}]})

    assert response.status_code == 400
    body = response.json()["error"]
    assert body["code"] == "streaming_unsupported"
    assert "OpenAI-compatible" in body["message"]
