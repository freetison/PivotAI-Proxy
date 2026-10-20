"""Phase 8: hardening — client keys, per-key rate limiting, payload size cap.

Three separate promises, tested separately:

* a client key is required when keys are configured, and several keys can coexist;
* the rate limit is *per key*, so one noisy client cannot starve another;
* an oversized body is refused before it is parsed, not after it is in memory.

Plus the thing none of the above proves on its own: fifty requests at once must
leave the counters exactly right, with no state shared between them.
"""

from __future__ import annotations

import asyncio
import json
import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.jev_evaluator import EvaluationResult
from src.main import _read_body, create_app
from src.metrics import Metrics
from src.ratelimit import RateLimiter
from src.settings import Settings

from tests.conftest import base_config_dict, load_config, write_config_data

CHAT = "/v1/chat/completions"
BODY = {"model": "auto", "messages": [{"role": "user", "content": "q"}]}


def configured(tmp_path):
    return load_config(write_config_data(tmp_path, base_config_dict()))


class FakeProvider:
    def __init__(self, delay: float = 0.0):
        self.calls: list[str] = []
        self.delay = delay

    async def complete(self, model, messages, timeout_ms=None, **kwargs):
        if self.delay:
            await asyncio.sleep(self.delay)
        self.calls.append(model)
        return {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }


class FakeEvaluator:
    async def evaluate(self, prompt, response, *, threshold=None, system=""):
        return EvaluationResult(approved=True, score=0.99)


def client(config, *, settings=None, provider=None, metrics=None, raise_server_exceptions=True):
    return TestClient(create_app(
        settings=settings or Settings(),
        cascade_config=config,
        provider=provider or FakeProvider(),
        evaluator=FakeEvaluator(),
        metrics=metrics,
    ), raise_server_exceptions=raise_server_exceptions)


def post(client, *, key: str | None = None, body=None, headers=None, **kwargs):
    all_headers = dict(headers or {})
    if key is not None:
        all_headers["Authorization"] = f"Bearer {key}"
    return client.post(CHAT, json=body if body is not None else BODY, headers=all_headers, **kwargs)


# ------------------------------------------------------------- client keys

def test_no_keys_configured_means_no_auth_required(tmp_path):
    with client(configured(tmp_path)) as api:
        assert post(api).status_code == 200


def test_a_single_key_still_works_as_before(tmp_path):
    with client(configured(tmp_path), settings=Settings(api_key="only-key")) as api:
        assert post(api, key="only-key").status_code == 200
        assert post(api).status_code == 401
        assert post(api, key="wrong").status_code == 401


def test_several_keys_are_all_accepted(tmp_path):
    settings = Settings(client_keys=(("team-a", "key-a"), ("team-b", "key-b")))
    with client(configured(tmp_path), settings=settings) as api:
        assert post(api, key="key-a").status_code == 200
        assert post(api, key="key-b").status_code == 200
        assert post(api, key="key-c").status_code == 401


def test_a_key_is_matched_exactly(tmp_path):
    settings = Settings(client_keys=(("team-a", "key-a"),))
    with client(configured(tmp_path), settings=settings) as api:
        assert post(api, key="KEY-A").status_code == 401
        assert post(api, key="key-a ").status_code == 401
        assert post(api, key="key").status_code == 401


def test_keys_are_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("API_KEYS", "team-a:key-a,team-b:key-b")

    settings = Settings.from_env()

    assert settings.client_keys == (("team-a", "key-a"), ("team-b", "key-b"))


def test_a_bare_key_gets_a_generated_label(monkeypatch):
    monkeypatch.setenv("API_KEYS", "key-only")

    settings = Settings.from_env()

    assert settings.client_keys == (("key-1", "key-only"),)


def test_api_keys_win_over_the_legacy_single_key(monkeypatch):
    monkeypatch.setenv("API_KEY", "legacy")
    monkeypatch.setenv("API_KEYS", "team-a:key-a")

    settings = Settings.from_env()

    assert settings.client_keys == (("team-a", "key-a"),)


def test_the_legacy_key_becomes_the_default_label(monkeypatch):
    monkeypatch.setenv("API_KEY", "legacy")
    monkeypatch.delenv("API_KEYS", raising=False)

    settings = Settings.from_env()

    assert settings.client_keys == (("default", "legacy"),)


def test_a_malformed_key_list_fails_at_startup(monkeypatch):
    monkeypatch.setenv("API_KEYS", "team-a:,key-b")

    with pytest.raises(ValueError, match="API_KEYS"):
        Settings.from_env()


def test_a_duplicate_label_fails_at_startup(monkeypatch):
    monkeypatch.setenv("API_KEYS", "team-a:key-a,team-a:key-b")

    with pytest.raises(ValueError, match="duplicate label"):
        Settings.from_env()


def test_keys_never_appear_in_the_settings_repr():
    settings = Settings(client_keys=(("team-a", "super-secret-key"),))

    assert "super-secret-key" not in repr(settings)


# ------------------------------------------------------------ rate limiting

def test_the_limiter_allows_up_to_the_limit():
    limiter = RateLimiter(limit=3, window_seconds=60, clock=lambda: 0.0)

    assert [limiter.check("k").allowed for _ in range(3)] == [True, True, True]
    assert limiter.check("k").allowed is False


def test_the_refusal_says_how_long_to_wait():
    limiter = RateLimiter(limit=1, window_seconds=10, clock=lambda: 0.0)
    limiter.check("k")

    decision = limiter.check("k")

    assert decision.allowed is False
    assert decision.retry_after == pytest.approx(10.0)


def test_the_bucket_refills_after_the_window():
    now = [0.0]
    limiter = RateLimiter(limit=1, window_seconds=10, clock=lambda: now[0])

    assert limiter.check("k").allowed is True
    now[0] = 9.9
    assert limiter.check("k").allowed is False
    now[0] = 10.1
    assert limiter.check("k").allowed is True


def test_each_key_has_its_own_bucket():
    limiter = RateLimiter(limit=1, window_seconds=60, clock=lambda: 0.0)

    assert limiter.check("a").allowed is True
    assert limiter.check("a").allowed is False
    assert limiter.check("b").allowed is True


def test_the_remaining_count_is_reported():
    limiter = RateLimiter(limit=2, window_seconds=60, clock=lambda: 0.0)

    assert limiter.check("k").remaining == 1
    assert limiter.check("k").remaining == 0


def test_a_request_over_the_limit_is_refused_with_retry_after(tmp_path):
    settings = Settings(client_keys=(("team-a", "key-a"),),
                        rate_limit_requests=2, rate_limit_window_seconds=60)
    with client(configured(tmp_path), settings=settings) as api:
        assert post(api, key="key-a").status_code == 200
        assert post(api, key="key-a").status_code == 200
        refused = post(api, key="key-a")

    assert refused.status_code == 429
    assert int(refused.headers["Retry-After"]) >= 1
    error = refused.json()["error"]
    assert error["type"] == "rate_limit_error"
    assert error["code"] == "rate_limit_exceeded"


def test_the_limit_is_per_key_not_global(tmp_path):
    settings = Settings(
        client_keys=(("team-a", "key-a"), ("team-b", "key-b")),
        rate_limit_requests=1, rate_limit_window_seconds=60,
    )
    with client(configured(tmp_path), settings=settings) as api:
        assert post(api, key="key-a").status_code == 200
        assert post(api, key="key-a").status_code == 429
        # team-b must be unaffected by team-a exhausting its own budget.
        assert post(api, key="key-b").status_code == 200


def test_no_limit_by_default(tmp_path):
    with client(configured(tmp_path), settings=Settings(client_keys=(("a", "k"),))) as api:
        for _ in range(20):
            assert post(api, key="k").status_code == 200


def test_anonymous_clients_share_one_bucket(tmp_path):
    """No keys configured: one shared bucket, because there is nobody to tell apart."""
    settings = Settings(rate_limit_requests=1, rate_limit_window_seconds=60)
    with client(configured(tmp_path), settings=settings) as api:
        assert post(api).status_code == 200
        assert post(api).status_code == 429


def test_the_bearer_scheme_is_required(tmp_path):
    settings = Settings(client_keys=(("a", "key-a"),))
    with client(configured(tmp_path), settings=settings) as api:
        assert post(api, headers={"Authorization": "key-a"}).status_code == 401
        assert post(api, headers={"Authorization": "Basic key-a"}).status_code == 401


def test_the_rate_limit_must_be_positive(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "-1")

    with pytest.raises(ValueError, match="RATE_LIMIT_REQUESTS"):
        Settings.from_env()


def test_a_non_integer_rate_limit_fails_at_startup(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "lots")

    with pytest.raises(ValueError, match="RATE_LIMIT_REQUESTS must be an integer"):
        Settings.from_env()


def test_a_non_numeric_window_fails_at_startup(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_WINDOW_SECONDS", "a minute")

    with pytest.raises(ValueError, match="RATE_LIMIT_WINDOW_SECONDS must be a number"):
        Settings.from_env()


def test_a_zero_window_fails_at_startup(monkeypatch):
    """A window of zero would refill every instant, silently disabling the limit."""
    monkeypatch.setenv("RATE_LIMIT_WINDOW_SECONDS", "0")

    with pytest.raises(ValueError, match="RATE_LIMIT_WINDOW_SECONDS must be positive"):
        Settings.from_env()


def test_zero_disables_the_rate_limit_explicitly(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "0")

    settings = Settings.from_env()

    assert settings.rate_limit_requests == 0
    assert settings.rate_limit_window_seconds == 60.0


def test_the_limiter_refuses_a_degenerate_configuration():
    with pytest.raises(ValueError, match="at least 1"):
        RateLimiter(limit=0, window_seconds=60)
    with pytest.raises(ValueError, match="window must be positive"):
        RateLimiter(limit=1, window_seconds=0)


def test_an_empty_key_entry_fails_at_startup(monkeypatch):
    monkeypatch.setenv("API_KEYS", "a:key-a,,b:key-b")

    with pytest.raises(ValueError, match="empty entries"):
        Settings.from_env()


def test_a_labelless_entry_with_only_a_colon_fails_at_startup(monkeypatch):
    monkeypatch.setenv("API_KEYS", ":key-a")

    with pytest.raises(ValueError, match="label:key"):
        Settings.from_env()


def test_health_is_never_rate_limited(tmp_path):
    settings = Settings(rate_limit_requests=1, rate_limit_window_seconds=60)
    with client(configured(tmp_path), settings=settings) as api:
        for _ in range(5):
            assert api.get("/health").status_code == 200


def test_a_rate_limited_request_is_counted(tmp_path):
    metrics = Metrics()
    settings = Settings(rate_limit_requests=1, rate_limit_window_seconds=60)
    with client(configured(tmp_path), settings=settings, metrics=metrics) as api:
        post(api)
        post(api)

    assert metrics.snapshot()["rate_limited_total"] == 1


# ------------------------------------------------------------ payload limit

def test_a_body_over_the_limit_is_refused(tmp_path):
    settings = Settings(max_payload_bytes=200)
    big = {"model": "auto", "messages": [{"role": "user", "content": "x" * 500}]}
    with client(configured(tmp_path), settings=settings) as api:
        response = post(api, body=big)

    assert response.status_code == 413
    error = response.json()["error"]
    assert error["code"] == "payload_too_large"
    assert error["type"] == "invalid_request_error"


def test_a_body_at_the_limit_is_accepted(tmp_path):
    body = json.dumps(BODY)
    settings = Settings(max_payload_bytes=len(body))
    with client(configured(tmp_path), settings=settings) as api:
        response = api.post(CHAT, content=body,
                            headers={"Content-Type": "application/json"})

    assert response.status_code == 200


def test_an_oversized_content_length_is_refused_without_buffering_it(tmp_path):
    settings = Settings(max_payload_bytes=100)
    with client(configured(tmp_path), settings=settings) as api:
        response = api.post(CHAT, content=b'{"model": 1}',
                            headers={"Content-Type": "application/json",
                                     "Content-Length": "10000000"})

    assert response.status_code == 413


def test_an_overflowing_content_length_falls_through_to_the_stream_check(tmp_path):
    """A garbage header must not be trusted in either direction."""
    settings = Settings(max_payload_bytes=10_000)
    with client(configured(tmp_path), settings=settings) as api:
        response = api.post(CHAT, content=json.dumps(BODY),
                            headers={"Content-Type": "application/json",
                                     "Content-Length": "not-a-number"})

    assert response.status_code == 200


def test_an_oversized_body_is_counted(tmp_path):
    metrics = Metrics()
    settings = Settings(max_payload_bytes=100)
    big = {"model": "auto", "messages": [{"role": "user", "content": "x" * 500}]}
    with client(configured(tmp_path), settings=settings, metrics=metrics) as api:
        post(api, body=big)

    assert metrics.snapshot()["payload_too_large_total"] == 1


def test_the_default_payload_limit_is_one_mebibyte():
    assert Settings().max_payload_bytes == 1_048_576


def test_a_bad_payload_limit_fails_at_startup(monkeypatch):
    monkeypatch.setenv("MAX_PAYLOAD_BYTES", "0")

    with pytest.raises(ValueError, match="MAX_PAYLOAD_BYTES"):
        Settings.from_env()


# -------------------------------------------------------------- concurrency

async def _concurrent(api, count, *, body=None, key=None):
    """Fire ``count`` requests at once, through the real ASGI app."""

    async def one(index):
        return await asyncio.to_thread(
            post, api, key=key, body={**BODY, "user": index} if body is None else body
        )

    return await asyncio.gather(*(one(index) for index in range(count)))


def test_fifty_concurrent_requests_all_succeed_with_consistent_counters(tmp_path, caplog):
    metrics = Metrics()
    provider = FakeProvider(delay=0.01)
    with client(configured(tmp_path), provider=provider, metrics=metrics) as api:
        with caplog.at_level(logging.INFO, logger="ai_cascade_proxy.telemetry"):
            responses = asyncio.run(_concurrent(api, 50))

    assert [response.status_code for response in responses] == [200] * 50
    snapshot = metrics.snapshot()
    assert snapshot["requests_total"] == 50
    assert snapshot["failed_requests_total"] == 0
    assert snapshot["outbound_calls_total"] == 50
    # No state shared between requests: fifty answers, fifty distinct events.
    events = [json.loads(record.getMessage()) for record in caplog.records]
    assert len(events) == 50
    assert len({event["request_id"] for event in events}) == 50
    assert all(event["hops_count"] == 1 for event in events)


def test_fifty_concurrent_requests_against_a_limit_let_exactly_the_limit_through(tmp_path):
    metrics = Metrics()
    settings = Settings(rate_limit_requests=10, rate_limit_window_seconds=60)

    class SlowProvider(FakeProvider):
        async def complete(self, model, messages, timeout_ms=None, **kwargs):
            await asyncio.sleep(0.005)
            return await super().complete(model, messages, timeout_ms=timeout_ms, **kwargs)

    with client(configured(tmp_path), settings=settings, metrics=metrics,
                provider=SlowProvider()) as api:
        responses = asyncio.run(_concurrent(api, 50))

    codes = [response.status_code for response in responses]
    assert codes.count(200) == 10
    assert codes.count(429) == 40
    snapshot = metrics.snapshot()
    assert snapshot["requests_total"] == 10
    assert snapshot["rate_limited_total"] == 40


def test_concurrent_requests_with_different_keys_do_not_interfere(tmp_path):
    settings = Settings(
        client_keys=(("a", "key-a"), ("b", "key-b")),
        rate_limit_requests=5, rate_limit_window_seconds=60,
    )
    with client(configured(tmp_path), settings=settings) as api:
        first = asyncio.run(_concurrent(api, 20, key="key-a"))
        second = asyncio.run(_concurrent(api, 20, key="key-b"))

    assert [r.status_code for r in first].count(200) == 5
    assert [r.status_code for r in second].count(200) == 5


# ------------------------------------------------------------ edge coverage

def test_a_missing_declared_length_is_checked_after_reading():
    """No Content-Length header (chunked): the streamed size is still enforced."""

    class NoLengthRequest:
        headers: dict[str, str] = {}

        async def body(self):
            return b"x" * 500

    counters = Metrics()

    with pytest.raises(Exception) as caught:
        asyncio.run(_read_body(NoLengthRequest(), 200, counters))

    assert getattr(caught.value, "status_code", None) == 413
    assert counters.snapshot()["payload_too_large_total"] == 1


def test_a_body_within_the_limit_is_returned_unchanged():
    class SmallRequest:
        headers = {"content-length": "4"}

        async def body(self):
            return b"body"

    assert asyncio.run(_read_body(SmallRequest(), 200, Metrics())) == b"body"


def test_a_negative_payload_limit_fails_at_startup(monkeypatch):
    monkeypatch.setenv("MAX_PAYLOAD_BYTES", "-5")

    with pytest.raises(ValueError, match="MAX_PAYLOAD_BYTES must be at least 1"):
        Settings.from_env()


def test_a_non_integer_payload_limit_fails_at_startup(monkeypatch):
    monkeypatch.setenv("MAX_PAYLOAD_BYTES", "big")

    with pytest.raises(ValueError, match="MAX_PAYLOAD_BYTES must be an integer"):
        Settings.from_env()


# ------------------------------------------------------ payload validation

@pytest.mark.parametrize("body,needle", [
    ([1, 2], "JSON object"),
    ({"messages": [{"role": "user", "content": "q"}]}, "'model'"),
    ({"model": "   ", "messages": [{"role": "user", "content": "q"}]}, "'model'"),
    ({"model": "auto", "messages": []}, "'messages'"),
    ({"model": "auto", "messages": "q"}, "'messages'"),
    ({"model": "auto", "messages": ["q"]}, "string role"),
    ({"model": "auto", "messages": [{"role": "user"}]}, "missing content"),
    ({"model": "auto", "messages": [{"role": "user", "content": 5}]}, "string, array, or null"),
    ({"model": "auto", "messages": [{"role": "user", "content": "q"}], "stream": "yes"}, "'stream'"),
])
def test_a_malformed_payload_is_rejected_with_a_four_hundred(tmp_path, body, needle):
    with client(configured(tmp_path)) as api:
        response = api.post(CHAT, json=body)

    assert response.status_code == 400
    assert needle in response.json()["error"]["message"]


def test_invalid_json_is_rejected(tmp_path):
    with client(configured(tmp_path)) as api:
        response = api.post(CHAT, content=b"{not json",
                            headers={"Content-Type": "application/json"})

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_an_unexpected_error_does_not_leak_its_text(tmp_path):
    """A library exception can embed headers or URLs; the client gets a generic 500."""

    class ExplodingProvider(FakeProvider):
        async def complete(self, model, messages, timeout_ms=None, **kwargs):
            raise RuntimeError("secret-looking detail: api_key=abc123")

    with client(configured(tmp_path), provider=ExplodingProvider(),
                raise_server_exceptions=False) as api:
        # raise_server_exceptions=False: assert what a real client would receive,
        # not what the test harness would re-raise.
        response = api.post(CHAT, json=BODY)

    assert response.status_code == 500
    assert "secret-looking detail" not in response.text
    assert response.json()["error"]["message"] == "An internal server error occurred"


@pytest.mark.parametrize("stream", [False, True])
def test_an_unknown_model_is_rejected_on_both_paths(tmp_path, stream):
    """A bad ``model`` is a client error before anything leaves the proxy."""
    body = {**BODY, "model": "no-such-model", "stream": stream}
    with client(configured(tmp_path)) as api:
        response = api.post(CHAT, json=body)

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["param"] == "model"
    assert "Unsupported model" in error["message"]
