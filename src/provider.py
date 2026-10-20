"""OpenAI-compatible upstream model provider."""

from __future__ import annotations

import logging
import uuid
from urllib.parse import urlparse

import httpx

from src.settings import Settings

DEFAULT_TIMEOUT_MS = 120_000
MODEL_LIST_TIMEOUT_MS = 10_000

logger = logging.getLogger("ai_cascade_proxy.provider")


class ProviderError(Exception):
    """An upstream provider could not complete the request.

    ``status_code`` is what the *client* would see if this error propagated;
    ``upstream_status`` is what the provider actually returned (None for timeouts
    and transport failures). ``retryable`` is True only for 429: the cascade owns
    every retry, so the classification lives here rather than in the HTTP client.
    """

    def __init__(self, message: str, *, status_code: int = 502, provider: str = "unknown",
                 upstream_status: int | None = None, retryable: bool = False):
        super().__init__(message)
        self.status_code = status_code
        self.provider = provider
        self.upstream_status = upstream_status
        self.retryable = retryable


def _client_status(upstream_status: int) -> int:
    """Map an upstream status to the status this proxy returns.

    429 keeps its meaning (the caller may retry later) and 5xx becomes 502: the
    failure is an upstream one from the client's point of view.
    """
    if upstream_status == 429:
        return 429
    if upstream_status < 500:
        return upstream_status
    return 502


class ModelProvider:
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None):
        self.settings = settings
        self._client = client

    def _provider_details(self, model: str) -> tuple[str, str, str]:
        if "/" not in model:
            raise ProviderError(f"Model must use provider/model format: {model}", status_code=400)
        provider, upstream_model = model.split("/", 1)
        base_url = self.settings.provider_base_urls.get(provider)
        if not base_url:
            raise ProviderError(f"Unsupported model provider: {provider}", status_code=400, provider=provider)

        api_key = self.settings.api_key_for(provider)
        parsed = urlparse(base_url)
        local_host = (parsed.hostname or "").lower() in {"localhost", "127.0.0.1", "::1"}
        if not api_key and not local_host:
            # Prevent silently unauthenticated calls to remote providers.
            raise ProviderError(f"API key is not configured for provider: {provider}", provider=provider)
        if parsed.scheme != "https" and not local_host:
            raise ProviderError("Remote provider endpoints must use HTTPS", status_code=400, provider=provider)
        return provider, upstream_model, base_url

    @staticmethod
    def _requires_api_key(base_url: str) -> bool:
        host = (urlparse(base_url).hostname or "").lower()
        return host not in {"localhost", "127.0.0.1", "::1"}

    def _api_key(self, provider: str) -> str:
        return self.settings.api_key_for(provider)

    async def list_models(self, provider: str) -> set[str]:
        """Model ids the provider reports. Used only by the optional startup check."""
        base_url = self.settings.provider_base_urls.get(provider)
        if not base_url:
            raise ProviderError(f"Unsupported model provider: {provider}", status_code=400, provider=provider)
        headers = {"Content-Type": "application/json"}
        api_key = self._api_key(provider)
        if api_key:
            if provider == "anthropic":
                headers["x-api-key"] = api_key
                headers["anthropic-version"] = "2023-06-01"
            else:
                headers["Authorization"] = f"Bearer {api_key}"
        owns_client = self._client is None
        client = self._client or httpx.AsyncClient(timeout=httpx.Timeout(MODEL_LIST_TIMEOUT_MS / 1000))
        try:
            response = await client.get(f"{base_url.rstrip('/')}/models", headers=headers)
            if response.is_error:
                raise ProviderError(
                    f"Provider {provider} returned HTTP {response.status_code} when listing models",
                    provider=provider,
                ) from None
            data = response.json()
            entries = data.get("data") if isinstance(data, dict) else None
            if not isinstance(entries, list):
                raise ProviderError(f"Provider {provider} returned an invalid model list", provider=provider)
            return {
                str(entry["id"]) for entry in entries
                if isinstance(entry, dict) and entry.get("id")
            }
        except httpx.TimeoutException:
            raise ProviderError(f"Provider {provider} timed out when listing models", provider=provider) from None
        except httpx.HTTPError:
            raise ProviderError(f"Provider {provider} request failed when listing models", provider=provider) from None
        except ValueError:
            raise ProviderError(f"Provider {provider} returned invalid JSON when listing models", provider=provider) from None
        finally:
            if owns_client:
                await client.aclose()

    async def open_stream(self, model: str, messages: list[dict], *,
                          timeout_ms: float | None = None, **kwargs):
        """Open the upstream stream and return an async iterator of raw SSE text.

        The request is sent **eagerly**, so an upstream error (429, 5xx) is raised
        here, before the caller starts writing a response body: a streaming
        endpoint cannot report a failure once bytes are flowing.

        Only OpenAI-compatible endpoints are supported: the frames are forwarded
        verbatim, which requires the upstream to already speak the OpenAI chunk
        format. Anthropic's native Messages API does not, so it is refused rather
        than translated (a translator that is only half right corrupts answers).
        """
        provider, upstream_model, base_url = self._provider_details(model)
        if provider == "anthropic":
            raise ProviderError(
                f"Provider {provider} cannot be streamed in OpenAI format by this proxy",
                status_code=400, provider=provider,
            )
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        api_key = self._api_key(provider)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = {"model": upstream_model, "messages": messages, "stream": True, **kwargs}
        timeout = httpx.Timeout(
            DEFAULT_TIMEOUT_MS / 1000 if timeout_ms is None else timeout_ms / 1000
        )
        # retries=0 for the same reason as complete(): the cascade counts calls.
        client = self._client or httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(retries=0), timeout=timeout
        )
        owns_client = self._client is None
        request = client.build_request(
            "POST", f"{base_url.rstrip('/')}/chat/completions", headers=headers, json=payload
        )
        try:
            response = await client.send(request, stream=True)
        except httpx.TimeoutException:
            if owns_client:
                await client.aclose()
            raise ProviderError(f"Provider {provider} request timed out", provider=provider) from None
        except httpx.HTTPError:
            if owns_client:
                await client.aclose()
            raise ProviderError(f"Provider {provider} request failed", provider=provider) from None

        if response.is_error:
            # Still no upstream body in the error: it may disclose provider internals.
            logger.warning("Provider %s returned HTTP %s while streaming",
                           provider, response.status_code)
            status = response.status_code
            await response.aclose()
            if owns_client:
                await client.aclose()
            raise ProviderError(
                f"Upstream provider {provider} returned an error",
                status_code=_client_status(status),
                provider=provider,
                upstream_status=status,
                retryable=status == 429,
            )

        async def frames():
            try:
                async for chunk in response.aiter_text():
                    if chunk:
                        yield chunk
            finally:
                await response.aclose()
                if owns_client:
                    await client.aclose()

        return frames()

    async def complete(self, model: str, messages: list[dict], *,
                       timeout_ms: float | None = None, **kwargs) -> dict:
        provider, upstream_model, base_url = self._provider_details(model)
        headers = {"Content-Type": "application/json"}
        api_key = self._api_key(provider)
        if api_key:
            if provider == "anthropic":
                headers["x-api-key"] = api_key
                headers["anthropic-version"] = "2023-06-01"
            else:
                headers["Authorization"] = f"Bearer {api_key}"
        if provider == "anthropic":
            anthropic_messages = []
            system_parts = []
            for message in messages:
                if message.get("role") == "system":
                    system_parts.append(str(message.get("content", "")))
                else:
                    anthropic_messages.append({"role": message.get("role", "user"), "content": message.get("content", "")})
            payload = {"model": upstream_model, "messages": anthropic_messages}
            system_text = "\n".join(system_parts)
            if system_text:
                payload["system"] = system_text
            payload["max_tokens"] = kwargs.get("max_tokens", kwargs.get("max_completion_tokens", 1024))
            for key in ("temperature", "top_p", "top_k", "stop_sequences", "metadata"):
                if key in kwargs:
                    payload[key] = kwargs[key]
            if "stop" in kwargs:
                payload["stop_sequences"] = kwargs["stop"] if isinstance(kwargs["stop"], list) else [kwargs["stop"]]
            endpoint = f"{base_url.rstrip('/')}/messages"
        else:
            payload = {"model": upstream_model, "messages": messages, **kwargs}
            endpoint = f"{base_url.rstrip('/')}/chat/completions"
        owns_client = self._client is None
        # retries=0: httpx must never retry on its own. Every retry in this proxy is
        # counted by the cascade engine, and hidden retries would break that budget.
        client = self._client or httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(retries=0),
            timeout=httpx.Timeout(DEFAULT_TIMEOUT_MS / 1000 if timeout_ms is None else timeout_ms / 1000),
        )
        try:
            response = await client.post(
                endpoint, headers=headers, json=payload
            )
            if response.is_error:
                # Avoid forwarding upstream response bodies: they may disclose secrets or
                # provider internals. The status and provider are enough for client diagnosis.
                logger.warning("Provider %s returned HTTP %s", provider, response.status_code)
                # Don't chain HTTPStatusError: it can retain the upstream response,
                # request headers, and credential-bearing request URL in its traceback.
                raise ProviderError(
                    f"Upstream provider {provider} returned an error",
                    status_code=_client_status(response.status_code),
                    provider=provider,
                    upstream_status=response.status_code,
                    retryable=response.status_code == 429,
                ) from None
            data = response.json()
            if provider == "anthropic":
                if not isinstance(data, dict) or not isinstance(data.get("content"), list):
                    raise ProviderError(f"Provider {provider} returned an invalid completion payload", provider=provider)
                text = "".join(
                    block.get("text", "") for block in data["content"]
                    if isinstance(block, dict) and block.get("type") == "text"
                )
                input_tokens = data.get("usage", {}).get("input_tokens", 0)
                output_tokens = data.get("usage", {}).get("output_tokens", 0)
                stop_reason = data.get("stop_reason")
                return {
                    "id": data.get("id", f"anthropic-{uuid.uuid4().hex}"),
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "length" if stop_reason == "max_tokens" else "stop",
                    }],
                    "usage": {
                        "prompt_tokens": input_tokens,
                        "completion_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens,
                    },
                }
            if not isinstance(data, dict) or not isinstance(data.get("choices"), list):
                raise ProviderError(f"Provider {provider} returned an invalid completion payload", provider=provider)
            return data
        except httpx.TimeoutException:
            raise ProviderError(f"Provider {provider} request timed out", provider=provider) from None
        except httpx.HTTPError:
            # HTTP client exceptions can hold the authenticated request and URL.
            raise ProviderError(f"Provider {provider} request failed", provider=provider) from None
        except ValueError as exc:
            raise ProviderError(f"Provider {provider} returned invalid JSON", provider=provider) from exc
        finally:
            if owns_client:
                await client.aclose()
