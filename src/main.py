"""FastAPI application exposing an OpenAI-compatible chat completions API."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
import uuid
from collections.abc import Mapping
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from src.cascade_config import (
    CascadeConfig,
    CascadeConfigError,
    load_cascade_config_from_env,
    validate_models_against_providers,
)
from src.cascade_engine import CascadeEngine, CascadeError
from src.jev_evaluator import JevEvaluator, JevUnavailable
from src.metrics import Metrics
from src.payload import BYPASS_HEADER, bypass_header_value, detect_unsupported
from src.provider import ModelProvider, ProviderError
from src.ratelimit import RateLimiter
from src.settings import Settings, get_settings
from src.streaming import buffered_stream, with_terminator

logger = logging.getLogger("ai_cascade_proxy.api")


def _error(message: str, error_type: str, *, param: str | None = None,
           code: str | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {"message": message, "type": error_type}
    if param is not None:
        body["param"] = param
    if code is not None:
        body["code"] = code
    return {"error": body}


def _authenticate(request: Request, configured: Settings) -> str:
    """Return the caller's label, or raise 401.

    With no keys configured there is nothing to authenticate against, so every
    caller is ``anonymous`` and shares one rate-limit bucket.
    """
    if not configured.client_keys:
        return "anonymous"
    authorization = request.headers.get("authorization", "")
    scheme, _, supplied = authorization.partition(" ")
    label = configured.key_label(supplied) if scheme.lower() == "bearer" and supplied else None
    if label is None:
        raise HTTPException(
            status_code=401,
            detail=_error("Invalid or missing API key", "authentication_error"),
            headers={"WWW-Authenticate": "Bearer"},
        )
    return label


def _enforce_rate_limit(limiter: RateLimiter | None, label: str, counters: Metrics) -> None:
    """One unit of budget per request, per client label."""
    if limiter is None:
        return
    decision = limiter.check(label)
    if not decision.allowed:
        counters.record_rate_limited()
        # Round up: telling a client to retry in 0 seconds invites a hot loop.
        retry_after = max(1, math.ceil(decision.retry_after))
        raise HTTPException(
            status_code=429,
            detail=_error(
                f"Rate limit exceeded for this API key: {limiter.limit} requests per "
                f"{limiter.window_seconds:g}s. Retry in {retry_after}s",
                "rate_limit_error",
                code="rate_limit_exceeded",
            ),
            headers={"Retry-After": str(retry_after)},
        )


async def _read_body(request: Request, limit: int, counters: Metrics) -> bytes:
    """Read the body, refusing anything over ``limit`` before it is parsed.

    ``Content-Length`` is checked first so an oversized body is rejected without
    being buffered; the streamed read then catches a lying or absent header.
    """
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            if int(declared) > limit:
                counters.record_payload_too_large()
                raise _payload_too_large(limit)
        except ValueError:
            pass
    body = await request.body()
    if len(body) > limit:
        counters.record_payload_too_large()
        raise _payload_too_large(limit)
    return body


def _payload_too_large(limit: int) -> HTTPException:
    return HTTPException(
        status_code=413,
        detail=_error(
            f"Request body exceeds the {limit}-byte limit",
            "invalid_request_error",
            code="payload_too_large",
        ),
    )


def _validate_payload(payload: Any) -> tuple[list[dict[str, Any]], str, dict[str, Any], bool]:
    if not isinstance(payload, dict):
        raise HTTPException(400, detail=_error("Request body must be a JSON object", "invalid_request_error"))
    model = payload.get("model")
    if not isinstance(model, str) or not model.strip():
        raise HTTPException(400, detail=_error("'model' must be a non-empty string", "invalid_request_error", param="model"))
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise HTTPException(400, detail=_error("'messages' must be a non-empty array", "invalid_request_error", param="messages"))
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or not isinstance(message.get("role"), str):
            raise HTTPException(400, detail=_error(f"messages[{index}] must be an object with a string role", "invalid_request_error", param="messages"))
        if "content" not in message and message.get("role") not in {"assistant", "tool"}:
            raise HTTPException(400, detail=_error(f"messages[{index}] is missing content", "invalid_request_error", param="messages"))
        if "content" in message and message["content"] is not None and not isinstance(message["content"], (str, list)):
            raise HTTPException(400, detail=_error(f"messages[{index}].content must be a string, array, or null", "invalid_request_error", param="messages"))
    stream = payload.get("stream", False)
    if not isinstance(stream, bool):
        raise HTTPException(400, detail=_error("'stream' must be a boolean", "invalid_request_error", param="stream"))
    excluded = {"model", "messages", "stream"}
    kwargs = {key: value for key, value in payload.items() if key not in excluded}
    return messages, model.strip(), kwargs, stream


def _error_type_for(status_code: int) -> str:
    return {400: "invalid_request_error", 429: "rate_limit_error"}.get(status_code, "upstream_error")


def _openai_response(result, requested_model: str) -> dict[str, Any]:
    source = result.response
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": result.model_used or requested_model,
        "choices": source["choices"],
        "usage": source.get("usage", {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }),
    }


def create_app(settings: Settings | None = None, cascade_config: CascadeConfig | None = None,
               provider: ModelProvider | None = None,
               evaluator: JevEvaluator | None = None,
               metrics: Metrics | None = None,
               engine: CascadeEngine | None = None) -> FastAPI:
    configured = settings or get_settings()
    config = cascade_config if cascade_config is not None else load_cascade_config_from_env()
    counters = metrics if metrics is not None else (
        engine.metrics if engine is not None else Metrics()
    )
    if engine is None:
        engine = CascadeEngine(configured, config,
                               provider=provider or ModelProvider(configured),
                               evaluator=evaluator or JevEvaluator(configured, config),
                               metrics=counters)

    application = FastAPI(title="AI Cascade Proxy", version="0.1.0")
    limiter = (
        RateLimiter(limit=configured.rate_limit_requests,
                    window_seconds=configured.rate_limit_window_seconds)
        if configured.rate_limit_requests > 0 else None
    )
    if configured.cascade_mode == "shadow" and configured.stream_mode == "passthrough_l3":
        # passthrough_l3 never runs the cascade, so shadow mode has nothing to
        # compare and the collected sample would silently be empty. Say so once,
        # at startup, instead of letting the operator discover it in the store.
        logger.warning(
            "CASCADE_MODE=shadow with STREAM_MODE=passthrough_l3: streamed requests "
            "forward the final level directly, so no shadow comparison is recorded "
            "for them"
        )
    application.state.settings = configured
    application.state.cascade_config = config
    application.state.cascade_engine = engine
    application.state.metrics = counters
    application.state.rate_limiter = limiter

    @application.exception_handler(HTTPException)
    async def openai_http_error(request: Request, exc: HTTPException):
        detail = exc.detail
        if isinstance(detail, Mapping) and "error" in detail:
            content = dict(detail)
        else:
            content = _error(str(detail), "invalid_request_error")
        return JSONResponse(status_code=exc.status_code, content=content, headers=exc.headers)

    @application.exception_handler(JevUnavailable)
    async def jev_unavailable(request: Request, exc: JevUnavailable):
        # fail_closed: no score and no acceptable answer, so nothing is returned as
        # a completion. The body is OpenAI-shaped so clients see a normal error.
        logger.error("JEV unavailable and jev.failure_policy is fail_closed")
        return JSONResponse(
            status_code=503,
            content=_error(str(exc), "service_unavailable", code="jev_unavailable"),
        )

    @application.exception_handler(CascadeError)
    async def cascade_error(request: Request, exc: CascadeError):
        # Every level failed or the call budget ran out: the client gets an
        # OpenAI-shaped error, never an empty completion and never a 200.
        logger.error("Cascade failed (%s): %s", exc.code, type(exc).__name__)
        return JSONResponse(
            status_code=exc.status_code,
            content=_error(str(exc), _error_type_for(exc.status_code), code=exc.code),
        )

    @application.exception_handler(ProviderError)
    async def provider_error(request: Request, exc: ProviderError):
        return JSONResponse(
            status_code=exc.status_code,
            content=_error(str(exc), _error_type_for(exc.status_code), code="provider_error"),
        )

    @application.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception):
        # Do not log exception text or traceback: libraries may embed request headers,
        # connection URLs, or other credential-bearing details in them.
        logger.error("Unhandled proxy error (%s)", type(exc).__name__)
        return JSONResponse(
            status_code=500,
            content=_error("An internal server error occurred", "internal_error"),
        )

    @application.get("/health")
    async def health():
        return {"status": "ok"}

    @application.get("/metrics")
    async def metrics_endpoint(request: Request):
        _authenticate(request, configured)
        return counters.snapshot()

    @application.get("/v1/models")
    async def list_models(request: Request):
        _authenticate(request, configured)
        models = list(dict.fromkeys(("auto", *config.model_names)))
        return {
            "object": "list",
            "data": [
                {"id": model, "object": "model", "created": 0, "owned_by": "ai-cascade-proxy"}
                for model in models
            ],
        }

    @application.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        label = _authenticate(request, configured)
        _enforce_rate_limit(limiter, label, counters)
        body = await _read_body(request, configured.max_payload_bytes, counters)
        try:
            payload = json.loads(body)
        except (ValueError, TypeError):
            raise HTTPException(400, detail=_error("Request body must be valid JSON", "invalid_request_error"))
        messages, requested_model, kwargs, stream = _validate_payload(payload)

        # Payloads the evaluator cannot judge skip the cascade entirely: one level
        # answers them unchanged and the response says why.
        reasons = detect_unsupported(payload)
        bypass_headers = (
            {BYPASS_HEADER: bypass_header_value(reasons)} if reasons else {}
        )

        if stream:
            return await _stream_response(
                configured, engine, messages, requested_model, kwargs, stream, bypass_headers
            )

        try:
            if reasons:
                result = await engine.run_bypass(
                    messages, level_number=configured.unsupported_features_level, **kwargs
                )
            else:
                result = await engine.run(messages, requested_model=requested_model, **kwargs)
        except ValueError as exc:
            raise HTTPException(400, detail=_error(str(exc), "invalid_request_error", param="model")) from exc
        return JSONResponse(content=_openai_response(result, requested_model),
                            headers=bypass_headers)

    async def _stream_response(configured, engine, messages, requested_model, kwargs,
                               stream, bypass_headers):
        """Streaming branch. A bypass that cannot be streamed falls back to JSON chunks."""
        if configured.stream_mode == "passthrough_l3" and not bypass_headers:
            level = engine.final_level_for(requested_model)
            if not engine.can_stream_passthrough(level):
                raise HTTPException(400, detail=_error(
                    f"passthrough_l3 requires the final level ({level.full_name}) to be an "
                    "OpenAI-compatible endpoint; use STREAM_MODE=buffered_cascade for other providers",
                    "invalid_request_error", param="stream", code="streaming_unsupported"))
            try:
                upstream = await engine.open_stream(level, messages, kwargs)
            except ValueError as exc:
                raise HTTPException(400, detail=_error(str(exc), "invalid_request_error", param="model")) from exc
            return StreamingResponse(
                with_terminator(upstream),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", **bypass_headers},
            )

        # buffered_cascade (and every bypass): finish the work, then replay it as
        # chunks, so failures stay normal JSON errors instead of truncated streams.
        try:
            if bypass_headers:
                result = await engine.run_bypass(
                    messages, level_number=configured.unsupported_features_level, **kwargs
                )
            else:
                result = await engine.run(messages, requested_model=requested_model, **kwargs)
        except ValueError as exc:
            raise HTTPException(400, detail=_error(str(exc), "invalid_request_error", param="model")) from exc
        return StreamingResponse(
            buffered_stream(result, requested_model),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", **bypass_headers},
        )

    return application


def build_app() -> FastAPI:
    """Uvicorn factory entry point: ``uvicorn src.main:build_app --factory``."""
    settings = get_settings()
    config = load_cascade_config_from_env()
    if settings.validate_models_on_startup:
        model_provider = ModelProvider(settings)
        try:
            asyncio.run(validate_models_against_providers(config, model_provider))
        except CascadeConfigError as exc:
            raise RuntimeError(str(exc)) from None
    return create_app(settings=settings, cascade_config=config)


def main() -> None:
    """Validate the configuration, then serve. A bad config exits before the port is bound."""
    import sys

    try:
        load_cascade_config_from_env()
        settings = get_settings()
    except CascadeConfigError as exc:
        print(f"AI Cascade Proxy cannot start: {exc}", file=sys.stderr)
        raise SystemExit(1) from None

    import uvicorn

    uvicorn.run("src.main:build_app", factory=True, host=settings.host, port=settings.port, reload=False)


if __name__ == "__main__":
    main()
