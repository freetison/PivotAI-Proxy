"""Server-sent events for ``stream: true``.

Mode ``buffered_cascade`` (default) replays an already-approved answer as an
OpenAI chunk stream. ``passthrough_l3`` forwards the final level's own stream.

The buffered mode exists because evaluation needs the complete answer: there is
no partial answer to score, so nothing can be emitted before the cascade ends.
The cost is time-to-first-token (TTFT ≈ full cascade latency); the benefit is
that an unevaluated answer never reaches the client.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator

from src.cascade_engine import CascadeResult


def sse_event(payload: dict) -> str:
    """One SSE frame. Compact JSON: chunk streams are dominated by framing overhead."""
    return f"data: {json.dumps(payload, separators=(',', ':'), ensure_ascii=False)}\n\n"


def sse_done() -> str:
    """The OpenAI end-of-stream marker, byte for byte."""
    return "data: [DONE]\n\n"


def _chunk(chunk_id: str, created: int, model: str, delta: dict,
           finish_reason: str | None = None) -> dict:
    return {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def _finish_reason(result: CascadeResult) -> str:
    choices = result.response.get("choices") or []
    if choices and isinstance(choices[0], dict):
        reason = choices[0].get("finish_reason")
        if isinstance(reason, str) and reason:
            return reason
    return "stop"


async def buffered_stream(result: CascadeResult, requested_model: str) -> AsyncIterator[str]:
    """Replay a finished cascade as OpenAI chunks, then the ``[DONE]`` marker.

    One content chunk, not one per token: the answer is already complete, so
    splitting it would only add framing cost and no streaming benefit.
    """
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    model = result.model_used or requested_model
    choices = result.response.get("choices") or []
    content = ""
    if choices and isinstance(choices[0], dict):
        message = choices[0].get("message") or {}
        raw = message.get("content")
        content = raw if isinstance(raw, str) else ""

    yield sse_event(_chunk(chunk_id, created, model, {"role": "assistant"}))
    if content:
        yield sse_event(_chunk(chunk_id, created, model, {"content": content}))
    yield sse_event(_chunk(chunk_id, created, model, {}, finish_reason=_finish_reason(result)))
    yield sse_done()


async def with_terminator(upstream: AsyncIterator[str]) -> AsyncIterator[str]:
    """Forward upstream frames, guaranteeing a single ``[DONE]`` at the end.

    Frames are forwarded byte for byte: the client parses what the provider
    produced. Only the terminator is added, because some compatible endpoints
    close the connection without it and OpenAI SDKs then hang or raise.
    """
    tail = ""
    async for chunk in upstream:
        tail = (tail + chunk)[-64:]
        yield chunk
    if not tail.rstrip().endswith("[DONE]"):
        yield sse_done()
