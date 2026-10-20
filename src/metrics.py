"""In-process counters exposed by `/metrics`.

Deliberately tiny and dependency-free: the proxy needs a handful of counters
that stay correct under concurrent requests, not a metrics framework. Counters
are guarded by a lock so no snapshot can observe a half-updated value.
"""

from __future__ import annotations

import threading

from src.jev_evaluator import JevStatus

#: JEV statuses that mean "the evaluator produced no score".
UNAVAILABLE_STATUSES = frozenset({"unavailable"})


class Metrics:
    """Counters shared by the engine and the API. One instance per application."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jev_evaluations = 0
        self._jev_unavailable = 0
        self._requests = 0
        self._escalations = 0
        self._failed_requests = 0
        self._outbound_calls = 0
        self._provider_failures = 0
        self._failures_by_model: dict[str, int] = {}
        self._rate_limited = 0
        self._payload_too_large = 0

    def record_jev_evaluation(self, status: JevStatus | str) -> None:
        """Count one evaluator decision; ``unavailable`` means no score was produced."""
        unavailable = str(status) in UNAVAILABLE_STATUSES
        with self._lock:
            self._jev_evaluations += 1
            if unavailable:
                self._jev_unavailable += 1

    def record_request(self, *, hops: int, failed: bool = False) -> None:
        """Count one cascade; ``hops > 1`` means it escalated."""
        with self._lock:
            self._requests += 1
            if hops > 1:
                self._escalations += 1
            if failed:
                self._failed_requests += 1

    def record_outbound_call(self) -> None:
        """Count one provider call, retries included (the cost-bearing unit)."""
        with self._lock:
            self._outbound_calls += 1

    def record_provider_failure(self, model: str) -> None:
        """Count one failed provider call, by model."""
        with self._lock:
            self._provider_failures += 1
            self._failures_by_model[model] = self._failures_by_model.get(model, 0) + 1

    def record_rate_limited(self) -> None:
        """Count one request refused by the per-key rate limiter."""
        with self._lock:
            self._rate_limited += 1

    def record_payload_too_large(self) -> None:
        """Count one request refused for exceeding the body size ceiling."""
        with self._lock:
            self._payload_too_large += 1

    def snapshot(self) -> dict[str, float | int]:
        """A consistent copy of every counter, with the derived fallback rate."""
        with self._lock:
            evaluations = self._jev_evaluations
            unavailable = self._jev_unavailable
            requests = self._requests
            escalations = self._escalations
            failed_requests = self._failed_requests
            outbound_calls = self._outbound_calls
            provider_failures = self._provider_failures
            failures_by_model = dict(self._failures_by_model)
            rate_limited = self._rate_limited
            payload_too_large = self._payload_too_large
        return {
            "requests_total": requests,
            "escalations_total": escalations,
            "failed_requests_total": failed_requests,
            "outbound_calls_total": outbound_calls,
            "provider_failures_total": provider_failures,
            "provider_failures_by_model": failures_by_model,
            "rate_limited_total": rate_limited,
            "payload_too_large_total": payload_too_large,
            "jev_evaluations_total": evaluations,
            "jev_unavailable_total": unavailable,
            # Undefined rate is reported as 0.0 so dashboards and alerting rules
            # have a number to compare; the totals stay authoritative.
            "jev_fallback_rate": (unavailable / evaluations) if evaluations else 0.0,
        }
