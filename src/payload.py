"""Detection of payload features the cascade cannot evaluate.

The cascade's evaluator scores *one text answer*. Four kinds of request break
that assumption:

* ``tools``/``tool_choice`` — the answer is a tool call, not prose.
* ``response_format`` — the answer is judged against a schema, not by JEV.
* image content — the input is multimodal; a text score says nothing about it.
* ``n > 1`` — there is no single answer to approve or reject.

Rather than guess, these requests bypass the cascade entirely: one configured
level answers them, unchanged, and the response says why with ``X-Cascade-Bypass``.
"""

from __future__ import annotations

from typing import Any

#: Response header naming the reasons a request skipped the cascade.
BYPASS_HEADER = "X-Cascade-Bypass"

#: Stable detection order, so the header is reproducible.
BYPASS_REASONS = ("tools", "response_format", "image_content", "choices")

#: Content-block types that carry something other than text.
_NON_TEXT_BLOCK_TYPES = frozenset({
    "image_url", "input_image", "image", "file", "input_file", "audio", "input_audio",
})


def _has_tools(payload: dict[str, Any]) -> bool:
    tools = payload.get("tools")
    if isinstance(tools, list) and tools:
        return True
    # `tool_choice` without `tools` is still a tool-shaped request: honour it.
    return "tool_choice" in payload and payload.get("tool_choice") not in (None, "none")


def _has_images(payload: dict[str, Any]) -> bool:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return False
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            block_type = block.get("type") if isinstance(block, dict) else None
            if block_type in _NON_TEXT_BLOCK_TYPES:
                return True
    return False


def detect_unsupported(payload: dict[str, Any]) -> tuple[str, ...]:
    """Reasons this payload cannot be evaluated, in :data:`BYPASS_REASONS` order."""
    reasons = []
    if _has_tools(payload):
        reasons.append("tools")
    if payload.get("response_format") is not None:
        reasons.append("response_format")
    if _has_images(payload):
        reasons.append("image_content")
    n = payload.get("n")
    if isinstance(n, int) and not isinstance(n, bool) and n > 1:
        reasons.append("choices")
    return tuple(reasons)


def bypass_header_value(reasons: tuple[str, ...]) -> str:
    """Header value: comma-joined reasons, or empty when nothing was bypassed."""
    return ",".join(reasons)
