"""Structured, privacy-preserving request telemetry with real (config-driven) costs."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field

from src.cascade_config import CascadeConfig, CascadeLevel

logger = logging.getLogger("ai_cascade_proxy.telemetry")

#: Characters kept per side when content logging is explicitly enabled. A log line
#: must stay greppable, not become the transcript nobody dares to rotate.
MAX_LOGGED_CONTENT_CHARS = 500


@dataclass(frozen=True)
class TelemetryEvent:
    model_used: str
    hops_count: int
    jev_score: float | None
    total_latency_ms: float
    cost_usd: float
    baseline_cost_usd: float
    net_savings_usd: float
    #: ``approved``/``rejected``/``unavailable``; None when the request ended before JEV ran.
    jev_status: str | None = None
    #: Every outbound provider call made for this request, retries included.
    outbound_calls: int = 0
    #: Levels whose provider call failed (any reason), in the order they were tried.
    failed_levels: tuple[str, ...] = ()
    #: Wall-clock spent per level, failures included, keyed ``provider/model``.
    stage_latencies: dict[str, float] = field(default_factory=dict)
    #: Present only when content logging was explicitly requested.
    prompt_text: str | None = None
    response_text: str | None = None
    log_content: bool = False
    #: ``active`` or ``shadow``: which mode produced the answer that was returned.
    cascade_mode: str = "active"
    #: The model that answered in shadow mode, so the pair can be correlated.
    shadow_model: str | None = None
    #: Spend added by the shadow call. Zero in active mode, always.
    shadow_cost_usd: float = 0.0
    #: What the JEV verifications cost. Already included in ``cost_usd``.
    jev_cost_usd: float = 0.0
    #: Correlates a log line with a stored shadow pair. Never sent to the client.
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    @property
    def escalated(self) -> bool:
        """Did the request leave the first tier? A bypass is not an escalation."""
        return self.hops_count > 1

    @property
    def shadow_failed_levels(self) -> tuple[str, ...]:
        """Levels that failed while shadowing; the request itself still succeeded."""
        return self.failed_levels if self.cascade_mode == "shadow" else ()

    def emit(self) -> None:
        payload = {
            "request_id": self.request_id,
            "cascade_mode": self.cascade_mode,
            "model_used": self.model_used,
            "hops_count": self.hops_count,
            "escalated": self.escalated,
            "jev_score": self.jev_score,
            "jev_status": self.jev_status,
            "total_latency_ms": round(self.total_latency_ms, 3),
            "latency_ms_by_stage": {
                stage: round(value, 3) for stage, value in self.stage_latencies.items()
            },
            "cost_usd": round(self.cost_usd, 8),
            "jev_cost_usd": round(self.jev_cost_usd, 8),
            "baseline_cost_usd": round(self.baseline_cost_usd, 8),
            "net_savings_usd": round(self.net_savings_usd, 8),
            "outbound_calls": self.outbound_calls,
            "failed_levels": list(self.failed_levels),
            "shadow_model": self.shadow_model,
            "shadow_cost_usd": round(self.shadow_cost_usd, 8),
        }
        if self.cascade_mode == "shadow":
            # Only shadow records carry this: in active mode a failed level is
            # already visible in failed_levels and did not cost an extra answer.
            payload["shadow_failed_levels"] = list(self.shadow_failed_levels)
        if self.log_content:
            payload["prompt"] = _truncate(self.prompt_text or "")
            payload["response"] = _truncate(self.response_text or "")
        logger.info(json.dumps(payload, separators=(",", ":"), sort_keys=True))


def estimate_cost(level: CascadeLevel, usage: dict) -> float:
    """USD cost of one provider call, from the level's configured per-million-token prices."""
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    if not isinstance(prompt_tokens, (int, float)) or not isinstance(completion_tokens, (int, float)):
        return 0.0
    pricing = level.pricing
    return (
        max(0.0, float(prompt_tokens)) * pricing.input_per_mtok_usd / 1_000_000
        + max(0.0, float(completion_tokens)) * pricing.output_per_mtok_usd / 1_000_000
    )


def create_event(config: CascadeConfig, model_used: str, hops_count: int,
                 jev_score: float | None, total_latency_ms: float, usage: dict,
                 candidate_usages: list[tuple[CascadeLevel, dict]] | None = None,
                 jev_status: str | None = None,
                 jev_cost_usd: float = 0.0,
                 outbound_calls: int = 0,
                 failed_levels: tuple[str, ...] = (),
                 stage_latencies: dict[str, float] | None = None,
                 prompt_text: str | None = None,
                 response_text: str | None = None,
                 log_content: bool = False,
                 cascade_mode: str = "active",
                 shadow_model: str | None = None,
                 shadow_cost_usd: float = 0.0,
                 request_id: str | None = None) -> TelemetryEvent:
    # Include rejected candidates: their provider calls are real incurred costs too.
    calls = candidate_usages if candidate_usages else []
    # The verifications are paid calls as well; leaving them out would flatter the
    # savings figure by hiding the cost of deciding to escalate.
    cost = sum(estimate_cost(level, call_usage) for level, call_usage in calls) + jev_cost_usd
    # Baseline: the same request answered directly by the final tier.
    baseline = estimate_cost(config.final_level, usage)
    return TelemetryEvent(
        model_used=model_used,
        hops_count=hops_count,
        jev_score=jev_score,
        total_latency_ms=total_latency_ms,
        cost_usd=cost,
        baseline_cost_usd=baseline,
        net_savings_usd=baseline - cost,
        jev_status=jev_status,
        outbound_calls=outbound_calls,
        failed_levels=tuple(failed_levels),
        # No breakdown given means one model answered: attribute it its whole latency.
        stage_latencies=dict(stage_latencies or {model_used: total_latency_ms}),
        prompt_text=prompt_text,
        response_text=response_text,
        log_content=log_content,
        cascade_mode=cascade_mode,
        shadow_model=shadow_model,
        shadow_cost_usd=shadow_cost_usd,
        jev_cost_usd=jev_cost_usd,
        request_id=request_id or uuid.uuid4().hex,
    )


def _truncate(text: str) -> str:
    if len(text) <= MAX_LOGGED_CONTENT_CHARS:
        return text
    return text[:MAX_LOGGED_CONTENT_CHARS] + "...[truncated]"
