import os

import httpx
import pytest

from src.provider import ModelProvider, ProviderError
from src.settings import Settings


@pytest.mark.asyncio
async def test_openai_compatible_provider_request():
    async def handler(request):
        assert request.url == "https://provider.test/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer provider-key"
        assert request.read() == b'{"model":"model-x","messages":[{"role":"user","content":"hi"}],"temperature":0.1}'
        return httpx.Response(200, json={"choices": [{"message": {"content": "hello"}}]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    settings = Settings(provider_base_urls={"custom": "https://provider.test/v1"})
    # Arbitrary provider credentials are read using the conventional <PROVIDER>_API_KEY env name.
    provider = ModelProvider(settings, client=client)
    provider.settings = settings
    import os
    os.environ["CUSTOM_API_KEY"] = "provider-key"
    try:
        result = await provider.complete(
            "custom/model-x", [{"role": "user", "content": "hi"}], temperature=0.1
        )
    finally:
        os.environ.pop("CUSTOM_API_KEY", None)
        await client.aclose()

    assert result["choices"][0]["message"]["content"] == "hello"


@pytest.mark.asyncio
async def test_anthropic_provider_uses_native_messages_api():
    async def handler(request):
        assert request.url == "https://api.anthropic.com/v1/messages"
        assert request.headers["x-api-key"] == "anthropic-key"
        assert request.headers["anthropic-version"] == "2023-06-01"
        import json
        body = json.loads(request.content)
        assert body["model"] == "claude-test"
        assert body["max_tokens"] == 1024
        assert body["system"] == "Be concise"
        assert body["messages"] == [{"role": "user", "content": "Hi"}]
        return httpx.Response(200, json={
            "id": "msg_123",
            "content": [{"type": "text", "text": "Hello!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 3, "output_tokens": 2},
        })

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = ModelProvider(Settings(
        anthropic_api_key="anthropic-key",
        provider_base_urls={"anthropic": "https://api.anthropic.com/v1"},
    ), client=client)
    try:
        result = await provider.complete("anthropic/claude-test", [
            {"role": "system", "content": "Be concise"},
            {"role": "user", "content": "Hi"},
        ])
    finally:
        await client.aclose()

    assert result["id"] == "msg_123"
    assert result["choices"][0]["message"] == {"role": "assistant", "content": "Hello!"}
    assert result["choices"][0]["finish_reason"] == "stop"
    assert result["usage"] == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}


@pytest.mark.asyncio
async def test_provider_error_does_not_leak_upstream_body():
    async def handler(request):
        return httpx.Response(500, text="private upstream diagnostics")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = ModelProvider(
        Settings(provider_base_urls={"custom": "https://provider.test/v1"}),
        client=client,
    )
    os.environ["CUSTOM_API_KEY"] = "provider-key"
    try:
        with pytest.raises(ProviderError, match="returned an error") as exc_info:
            await provider.complete("custom/model-x", [{"role": "user", "content": "hi"}])
    finally:
        os.environ.pop("CUSTOM_API_KEY", None)
        await client.aclose()

    assert "private upstream diagnostics" not in str(exc_info.value)
    assert exc_info.value.status_code == 502
    assert exc_info.value.__cause__ is None


@pytest.mark.asyncio
async def test_a_429_preserves_its_status_and_is_marked_retryable():
    async def handler(request):
        return httpx.Response(429, json={"error": "slow down"})

    os.environ["CUSTOM_API_KEY"] = "provider-key"
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = ModelProvider(Settings(provider_base_urls={"custom": "https://provider.test/v1"}),
                             client=client)
    try:
        with pytest.raises(ProviderError) as exc_info:
            await provider.complete("custom/model-x", [{"role": "user", "content": "hi"}])
    finally:
        await client.aclose()

    # Only 429 keeps its meaning upstream; 5xx is reported to the client as 502.
    assert exc_info.value.status_code == 429
    assert exc_info.value.upstream_status == 429
    assert exc_info.value.retryable is True


@pytest.mark.asyncio
@pytest.mark.parametrize("upstream_status", [500, 502, 503, 504])
async def test_a_5xx_is_reported_as_502_and_is_not_retryable(upstream_status):
    async def handler(request):
        return httpx.Response(upstream_status, json={"error": "boom"})

    os.environ["CUSTOM_API_KEY"] = "provider-key"
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = ModelProvider(Settings(provider_base_urls={"custom": "https://provider.test/v1"}),
                             client=client)
    try:
        with pytest.raises(ProviderError) as exc_info:
            await provider.complete("custom/model-x", [{"role": "user", "content": "hi"}])
    finally:
        await client.aclose()
        os.environ.pop("CUSTOM_API_KEY", None)

    assert exc_info.value.status_code == 502
    assert exc_info.value.upstream_status == upstream_status
    assert exc_info.value.retryable is False


@pytest.mark.asyncio
async def test_a_timeout_has_no_upstream_status_and_is_not_retryable():
    async def handler(request):
        raise httpx.ReadTimeout("too slow", request=request)

    os.environ["CUSTOM_API_KEY"] = "provider-key"
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = ModelProvider(Settings(provider_base_urls={"custom": "https://provider.test/v1"}),
                             client=client)
    try:
        with pytest.raises(ProviderError) as exc_info:
            await provider.complete("custom/model-x", [{"role": "user", "content": "hi"}])
    finally:
        await client.aclose()
        os.environ.pop("CUSTOM_API_KEY", None)

    assert exc_info.value.status_code == 502
    assert exc_info.value.upstream_status is None
    assert exc_info.value.retryable is False


@pytest.mark.asyncio
async def test_the_lazily_built_client_disables_hidden_http_retries(monkeypatch):
    """The cascade owns every retry, so the transport must never retry on its own."""
    captured: dict = {}
    real_client = httpx.AsyncClient

    def recording_client(**kwargs):
        captured.update(kwargs)
        return real_client(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={
                "choices": [{"message": {"role": "assistant", "content": "ok"}}]})),
            **{k: v for k, v in kwargs.items() if k != "transport"},
        )

    monkeypatch.setenv("CUSTOM_API_KEY", "provider-key")
    monkeypatch.setattr(httpx, "AsyncClient", recording_client)
    provider = ModelProvider(Settings(provider_base_urls={"custom": "https://provider.test/v1"}))

    await provider.complete("custom/model-x", [{"role": "user", "content": "hi"}])

    assert isinstance(captured["transport"], httpx.AsyncHTTPTransport)
    # httpx stores the retry budget on its connection pool, not on the transport.
    assert captured["transport"]._pool._retries == 0
