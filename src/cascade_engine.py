"""Bounded three-tier model cascade with per-level JEV evaluation.

Failure handling, in one place:

* A provider failure (429, 5xx, timeout, transport error) **consumes the hop** and
  escalates to the next level. The candidate never reaches JEV: there is nothing
  to evaluate.
* Only 429 is retried, at most once per level, with a short backoff.
* Every outbound provider call is counted, and a hard cap aborts the request
  before the (cap + 1)-th call. This is what bounds cost: hop count alone does
  not, because retries are calls too.
* If the last level fails, ``UpstreamCascadeFailed`` is raised: the client never
  receives an empty answer or a 200.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from src.cascade_config import CascadeConfig, CascadeLevel
from src.jev_evaluator import EvaluationResult, JevEvaluator, JevStatus, JevUnavailable
from src.metrics import Metrics
from src.provider import ModelProvider, ProviderError
from src.settings import Settings
from src.shadow import record_for, store_from_settings
from src.telemetry import create_event, estimate_cost

logger = logging.getLogger("ai_cascade_proxy.cascade")

#: Retries allowed per level, on a 429 only. The plan caps this at one.
MAX_RETRIES_PER_LEVEL = 1
#: Delay before the single 429 retry. Overridable for tests.
DEFAULT_RETRY_BACKOFF_MS = 250


def failure_context(result: EvaluationResult) -> str:
    """Why a candidate was rejected, as a note the next level can act on.

    Module level and public because ``eval/escalation_context.py`` measures the
    escalation prompt: an experiment measuring a copy of this text would answer a
    different question than the proxy does.
    """
    if result.score is None:
        reason = "the evaluator could not score it"
    else:
        reason = f"its score {result.score} is below the approval threshold"
    return (
        "The previous candidate response was not approved because "
        f"{reason}. Re-answer the original request accurately and "
        "address the prior response's shortcomings. Treat the user's original "
        "messages as authoritative."
    )


class CascadeError(Exception):
    """Base class for failures the API turns into an OpenAI-shaped error."""

    status_code = 502
    code = "cascade_error"


class UpstreamCascadeFailed(CascadeError):
    """Every level that was tried failed: no answer exists for this request."""

    status_code = 502
    code = "cascade_failed"

    def __init__(self, failures: dict[str, int | None]):
        self.failures = dict(failures)
        tried = ", ".join(
            f"{model} ({'timeout' if status is None else f'HTTP {status}'})"
            for model, status in self.failures.items()
        )
        super().__init__(f"Every cascade level failed for this request: {tried}")


class OutboundCallLimitExceeded(CascadeError):
    """The request would exceed the hard cap on outbound provider calls."""

    status_code = 503
    code = "outbound_call_limit"

    def __init__(self, limit: int, outbound_calls: int):
        self.limit = limit
        self.outbound_calls = outbound_calls
        super().__init__(
            f"Refusing to make outbound provider call {outbound_calls + 1}: "
            f"the hard limit is {limit} calls per request"
        )


def _prompt_text(messages: list[dict]) -> str:
    """The full conversation, for opt-in content logging only."""
    return "\n".join(
        f"{message.get('role', '?')}: {_candidate_text(message.get('content'))}"
        for message in messages
        if isinstance(message, dict)
    )


def _response_text(response: dict) -> str:
    """The answer text, for opt-in content logging only."""
    try:
        return _candidate_text(response["choices"][0]["message"].get("content"))
    except (KeyError, IndexError, TypeError):
        return ""


def _candidate_text(value) -> str:
    """Extract readable text from OpenAI text or multimodal content blocks."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(
            str(block["text"])
            for block in value
            if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)
        )
    return ""


@dataclass(frozen=True)
class CascadeResult:
    response: dict
    model_used: str
    hops_count: int
    jev_score: float | None
    total_latency_ms: float
    cost_usd: float
    baseline_cost_usd: float
    net_savings_usd: float
    #: ``approved``/``rejected`` when JEV scored the answer; ``unavailable`` when it did not.
    jev_status: JevStatus | None = None
    #: Every outbound provider call made for this request, successful or not.
    outbound_calls: int = 0
    #: Levels whose provider call failed, in the order they were tried.
    failed_levels: tuple[str, ...] = field(default_factory=tuple)
    #: Wall-clock per level, failures included, keyed ``provider/model``.
    latency_ms_by_stage: dict[str, float] = field(default_factory=dict)
    #: ``active`` or ``shadow``; says which mode produced the answer above.
    cascade_mode: str = "active"
    #: The cascade's own answer, kept only in shadow mode, for comparison.
    cascade_response: dict | None = None
    #: The model that produced :attr:`cascade_response`, in shadow mode.
    cascade_model_used: str | None = None
    #: The spend the direct final-level call added on top of the cascade.
    shadow_cost_usd: float = 0.0
    #: What the JEV verifications cost. Already included in :attr:`cost_usd`.
    jev_cost_usd: float = 0.0

    @property
    def escalated(self) -> bool:
        return self.hops_count > 1


class CascadeEngine:
    def __init__(self, settings: Settings, config: CascadeConfig,
                 provider: ModelProvider | None = None,
                 evaluator: JevEvaluator | None = None,
                 metrics: Metrics | None = None,
                 retry_backoff_ms: int = DEFAULT_RETRY_BACKOFF_MS,
                 max_outbound_calls: int | None = None):
        self.settings = settings
        self.config = config
        self.provider = provider or ModelProvider(settings)
        self.evaluator = evaluator or JevEvaluator(settings, config)
        self.metrics = metrics or Metrics()
        self.retry_backoff_ms = retry_backoff_ms
        self.max_outbound_calls = self._resolve_call_cap(max_outbound_calls)

    def _resolve_call_cap(self, override: int | None) -> int:
        """Effective cap: explicit argument, then settings, then 2 x max_hops.

        The default is derived from the configuration rather than hardcoded, so
        raising ``max_hops`` cannot silently raise the number of calls allowed.
        """
        configured = override if override is not None else self.settings.max_outbound_calls
        if configured is None:
            return 2 * self.config.max_hops
        return max(1, int(configured))

    def _levels_for(self, requested_model: str) -> list[CascadeLevel]:
        if requested_model in ("", "auto"):
            return self.config.levels[: self.config.max_hops]
        level = self.config.level_for_model(requested_model)
        if level is None:
            raise ValueError(f"Unsupported model: {requested_model}")
        return [level]

    def final_level_for(self, requested_model: str) -> CascadeLevel:
        """The level whose stream would be forwarded in passthrough mode."""
        return self._levels_for(requested_model)[-1]

    def level_by_number(self, number: int) -> CascadeLevel:
        """Look up a configured level by its 1-based number."""
        level = self.config.level_for_number(number)
        if level is None:
            raise ValueError(f"Unsupported cascade level: {number}")
        return level

    def can_stream_passthrough(self, level: CascadeLevel) -> bool:
        """Only OpenAI-compatible providers produce frames this proxy can forward as-is."""
        return level.provider != "anthropic"

    async def run_bypass(self, messages: list[dict], *, level_number: int,
                         **kwargs) -> CascadeResult:
        """Answer with one level, no cascade and no evaluation.

        Used for payloads JEV cannot judge (tools, schemas, images, ``n > 1``).
        Deliberately no retry and no escalation: the caller asked for one specific
        shape of answer, so silently trying a different model is not an option.
        """
        level = self.level_by_number(level_number)
        started = time.perf_counter()
        response = await self.provider.complete(
            level.full_name, messages,
            timeout_ms=level.params.timeout_ms,
            **self._call_params(level, kwargs),
        )
        self.metrics.record_outbound_call()
        self.metrics.record_request(hops=1)
        latency = (time.perf_counter() - started) * 1000
        stage_latencies = {level.full_name: latency}
        log_content = bool(getattr(self.settings, "log_content", False))
        usage = response.get("usage", {})
        event = create_event(
            self.config,
            model_used=level.full_name,
            hops_count=1,
            jev_score=None,
            total_latency_ms=latency,
            usage=usage if isinstance(usage, dict) else {},
            candidate_usages=[(level, usage if isinstance(usage, dict) else {})],
            jev_status=None,
            jev_cost_usd=0.0,
            outbound_calls=1,
            failed_levels=(),
            stage_latencies=stage_latencies,
            prompt_text=_prompt_text(messages),
            response_text=_response_text(response),
            log_content=log_content,
        )
        event.emit()
        return CascadeResult(
            response=response,
            model_used=level.full_name,
            hops_count=1,
            jev_score=None,
            total_latency_ms=latency,
            cost_usd=event.cost_usd,
            baseline_cost_usd=event.baseline_cost_usd,
            net_savings_usd=event.net_savings_usd,
            jev_status=None,
            jev_cost_usd=0.0,
            outbound_calls=1,
            failed_levels=(),
            latency_ms_by_stage=stage_latencies,
        )

    async def open_stream(self, level: CascadeLevel, messages: list[dict],
                          kwargs: dict) -> AsyncIterator[str]:
        """Start a passthrough stream, bypassing the cascade and JEV entirely.

        No retry and no escalation: bytes are already flowing, so a failure is
        reported as an error rather than by quietly starting a second answer.
        """
        return await self.provider.open_stream(
            level.full_name,
            messages,
            timeout_ms=level.params.timeout_ms,
            **self._call_params(level, kwargs),
        )

    async def _call_level(self, level: CascadeLevel, messages: list[dict],
                          kwargs: dict, state: "_CallState") -> dict:
        """Call one level, retrying a 429 at most once. Counts and caps every call."""
        for attempt in range(MAX_RETRIES_PER_LEVEL + 1):
            if state.outbound_calls >= self.max_outbound_calls:
                raise OutboundCallLimitExceeded(self.max_outbound_calls, state.outbound_calls)
            state.outbound_calls += 1
            self.metrics.record_outbound_call()
            if attempt:
                # A retry is a deliberate second call, not a hidden one: wait first.
                await asyncio.sleep(self.retry_backoff_ms / 1000)
            try:
                return await self.provider.complete(
                    level.full_name,
                    messages,
                    timeout_ms=level.params.timeout_ms,
                    **self._call_params(level, kwargs),
                )
            except ProviderError as exc:
                state.failures.append((level.full_name, exc.upstream_status))
                self.metrics.record_provider_failure(level.full_name)
                logger.warning(
                    "Level %s failed (%s); %s",
                    level.full_name,
                    f"HTTP {exc.upstream_status}" if exc.upstream_status else "timeout/transport",
                    "retrying once" if exc.retryable and attempt < MAX_RETRIES_PER_LEVEL
                    else "consuming the hop",
                )
                if exc.retryable and attempt < MAX_RETRIES_PER_LEVEL:
                    continue
                raise
        raise AssertionError("unreachable: the retry loop always returns or raises")

    async def run(self, messages: list[dict], requested_model: str = "auto", **kwargs) -> CascadeResult:
        """Answer one request.

        In ``active`` mode that is the cascade's own answer. In ``shadow`` mode the
        cascade still runs and still costs money, but the client receives a direct
        final-level answer and the cascade's answer is kept for comparison.
        """
        if self._shadow_applies(requested_model):
            return await self._run_shadow(messages, requested_model, **kwargs)
        return await self._run_cascade(messages, requested_model, **kwargs)

    def _shadow_applies(self, requested_model: str) -> bool:
        """Shadow only makes sense for a cascade request, not for a named model."""
        if getattr(self.settings, "cascade_mode", "active") != "shadow":
            return False
        return requested_model in ("", "auto")

    async def _run_shadow(self, messages: list[dict], requested_model: str,
                          state: "_CallState | None" = None, **kwargs) -> CascadeResult:
        """Run the cascade for evidence, then answer from the final level.

        A cascade failure is not a request failure here: the client's answer does
        not come from the cascade, so a broken cascade must not break the request.
        The failure is reported in telemetry (``shadow_failed_levels``) instead.
        """
        levels = self._levels_for(requested_model)
        final_level = levels[-1]
        started = time.perf_counter()
        state = state if state is not None else _CallState()
        cascade_result: CascadeResult | None = None

        try:
            cascade_result = await self._run_cascade(
                messages, requested_model, state=state, **kwargs
            )
        except (CascadeError, ProviderError, JevUnavailable) as exc:
            # Expected in shadow mode; the direct call below still has to succeed.
            logger.info(
                "Shadow cascade failed (%s); answering with the final level anyway",
                type(exc).__name__,
            )

        # The shadow call still obeys the per-request cap: shadowing must not
        # widen the budget it exists to observe. If the cascade already spent it,
        # degrade to the cascade's answer and say so in the log.
        if state.outbound_calls >= self.max_outbound_calls:
            logger.warning(
                "Shadow call skipped: the cascade already used %s outbound calls",
                state.outbound_calls,
            )
            if cascade_result is None:
                self.metrics.record_request(
                    hops=len(state.failures) or 1, failed=True
                )
                raise UpstreamCascadeFailed({model: status for model, status in state.failures})
            return cascade_result

        state.outbound_calls += 1
        self.metrics.record_outbound_call()
        direct_response = await self.provider.complete(
            final_level.full_name,
            messages,
            timeout_ms=final_level.params.timeout_ms,
            **self._call_params(final_level, kwargs),
        )
        direct_usage = direct_response.get("usage", {})
        direct_usage = direct_usage if isinstance(direct_usage, dict) else {}
        state.candidate_usages.append((final_level, direct_usage))

        latency = (time.perf_counter() - started) * 1000
        failed_levels = cascade_result.failed_levels if cascade_result else tuple(
            model for model, _ in state.failures
        )
        event = create_event(
            self.config,
            model_used=final_level.full_name,
            hops_count=cascade_result.hops_count if cascade_result else 1,
            jev_score=cascade_result.jev_score if cascade_result else None,
            total_latency_ms=latency,
            usage=direct_usage,
            candidate_usages=state.candidate_usages,
            jev_status=cascade_result.jev_status if cascade_result else None,
            jev_cost_usd=state.jev_cost_usd,
            outbound_calls=state.outbound_calls,
            failed_levels=failed_levels,
            stage_latencies=(cascade_result.latency_ms_by_stage if cascade_result else {}),
            prompt_text=_prompt_text(messages),
            response_text=_response_text(direct_response),
            log_content=bool(getattr(self.settings, "log_content", False)),
            cascade_mode="shadow",
            shadow_model=final_level.full_name,
            shadow_cost_usd=sum(
                _usage_cost(level, usage) for level, usage in state.candidate_usages[-1:]
            ),
        )
        event.emit()

        store = store_from_settings(self.settings)
        if store is not None:
            store.append(record_for(
                event.request_id,
                cascade_model=cascade_result.model_used if cascade_result else "",
                cascade_response=cascade_result.response if cascade_result else None,
                shadow_model=final_level.full_name,
                shadow_response=direct_response,
                failed_levels=failed_levels,
            ))

        return CascadeResult(
            response=direct_response,
            model_used=final_level.full_name,
            hops_count=cascade_result.hops_count if cascade_result else 1,
            jev_score=cascade_result.jev_score if cascade_result else None,
            total_latency_ms=latency,
            cost_usd=event.cost_usd,
            baseline_cost_usd=event.baseline_cost_usd,
            net_savings_usd=event.net_savings_usd,
            jev_status=cascade_result.jev_status if cascade_result else None,
            jev_cost_usd=state.jev_cost_usd,
            outbound_calls=state.outbound_calls,
            failed_levels=failed_levels,
            latency_ms_by_stage=dict(cascade_result.latency_ms_by_stage) if cascade_result else {},
            cascade_mode="shadow",
            cascade_response=cascade_result.response if cascade_result else None,
            cascade_model_used=cascade_result.model_used if cascade_result else None,
            shadow_cost_usd=event.shadow_cost_usd,
        )

    async def _run_cascade(self, messages: list[dict], requested_model: str = "auto", *,
                           state: "_CallState | None" = None, **kwargs) -> CascadeResult:
        levels = self._levels_for(requested_model)

        started = time.perf_counter()
        current_messages = [dict(message) for message in messages]
        final_response: dict | None = None
        final_model = ""
        final_score: float | None = None
        final_status: JevStatus | None = None
        # Usage accumulates on the shared state, so shadow mode can price the whole
        # request (cascade calls + the direct call) instead of only its last call.
        state = state if state is not None else _CallState()
        candidate_usages = state.candidate_usages
        stage_latencies: dict[str, float] = {}
        hops = 0

        # Explicit bounded iteration: the final tier is returned without JEV, even if all
        # previous tiers were rejected, so fallback cannot loop past the last level.
        for index, level in enumerate(levels):
            hops = index + 1
            # Time the level, failures included: a failed call still took time, and
            # hiding it would make the per-stage breakdown lie about total latency.
            level_started = time.perf_counter()
            try:
                response = await self._call_level(level, current_messages, kwargs, state)
            except OutboundCallLimitExceeded:
                self.metrics.record_request(hops=hops, failed=True)
                raise
            except ProviderError:
                stage_latencies[level.full_name] = (time.perf_counter() - level_started) * 1000
                if index == len(levels) - 1:
                    failed = {model: status for model, status in state.failures}
                    self.metrics.record_request(hops=hops, failed=True)
                    raise UpstreamCascadeFailed(failed) from None
                continue

            stage_latencies[level.full_name] = (time.perf_counter() - level_started) * 1000
            final_response = response
            final_model = level.full_name
            usage = response.get("usage", {})
            if isinstance(usage, dict):
                candidate_usages.append((level, usage))
            choices = response.get("choices", [])
            if not choices or not isinstance(choices[0], dict):
                raise ValueError("Provider returned no completion choices")
            message = choices[0].get("message")
            if not isinstance(message, dict):
                raise ValueError("Provider returned an invalid completion message")
            content = _candidate_text(message.get("content"))

            if not level.is_evaluated or index == len(levels) - 1:
                break

            try:
                evaluation = await self.evaluator.evaluate(
                    self._user_prompt(messages), content, threshold=level.jev_threshold,
                    system=self._system_prompt(messages),
                )
            except JevUnavailable:
                # fail_closed: count the failed evaluation before aborting, so the
                # outage is visible in /metrics even though no request completes.
                self.metrics.record_jev_evaluation("unavailable")
                raise
            final_score = evaluation.score
            final_status = evaluation.status
            state.jev_cost_usd += evaluation.cost_usd
            self.metrics.record_jev_evaluation(evaluation.status)
            if evaluation.approved:
                break

            current_messages.extend([
                {"role": "assistant", "content": content},
                {"role": "system", "content": failure_context(evaluation)},
            ])

        if final_response is None:  # Defensive guard; the loop either answers or raises.
            failed = {model: status for model, status in state.failures}
            self.metrics.record_request(hops=hops, failed=True)
            raise UpstreamCascadeFailed(failed)

        failed_levels = tuple(model for model, _ in state.failures)
        latency = (time.perf_counter() - started) * 1000
        self.metrics.record_request(hops=hops)
        final_usage = final_response.get("usage", {})
        event = create_event(
            self.config,
            model_used=final_model,
            hops_count=hops,
            jev_score=final_score,
            total_latency_ms=latency,
            usage=final_usage if isinstance(final_usage, dict) else {},
            candidate_usages=candidate_usages,
            jev_status=final_status,
            jev_cost_usd=state.jev_cost_usd,
            outbound_calls=state.outbound_calls,
            failed_levels=failed_levels,
            stage_latencies=stage_latencies,
            prompt_text=_prompt_text(messages),
            response_text=_response_text(final_response),
            log_content=bool(getattr(self.settings, "log_content", False)),
        )
        event.emit()
        return CascadeResult(
            response=final_response,
            model_used=final_model,
            hops_count=hops,
            jev_score=final_score,
            total_latency_ms=latency,
            cost_usd=event.cost_usd,
            baseline_cost_usd=event.baseline_cost_usd,
            net_savings_usd=event.net_savings_usd,
            jev_status=final_status,
            outbound_calls=state.outbound_calls,
            failed_levels=failed_levels,
            latency_ms_by_stage=stage_latencies,
        )

    @staticmethod
    def _call_params(level: CascadeLevel, request_kwargs: dict) -> dict:
        """Level configuration wins over client-supplied values; everything else passes through."""
        return {**request_kwargs, **level.provider_params}

    @staticmethod
    def _system_prompt(messages: list[dict]) -> str:
        """The client's system turns, joined in order.

        ``_user_prompt`` alone loses them, which let JEV approve an answer that
        ignored the instructions it was given. Dialects that can express the
        system prompt receive it; the legacy one ignores it, as it always has.
        """
        return "\n".join(
            str(message.get("content", ""))
            for message in messages
            if message.get("role") == "system"
        )

    @staticmethod
    def _user_prompt(messages: list[dict]) -> str:
        """Kept for JEV; content logging uses :func:`_prompt_text` instead."""
        return "\n".join(
            str(message.get("content", ""))
            for message in messages
            if message.get("role") == "user"
        )


@dataclass
class _CallState:
    """Per-request accounting: calls made, which levels failed, and what they cost."""

    outbound_calls: int = 0
    failures: list[tuple[str, int | None]] = field(default_factory=list)
    #: Every paid call with its reported usage, across modes; used for cost sums.
    candidate_usages: list[tuple[CascadeLevel, dict]] = field(default_factory=list)
    #: What the JEV verifications cost, as reported by the evaluator. The
    #: decision to escalate is itself a paid call, so it belongs in the total.
    jev_cost_usd: float = 0.0


def _usage_cost(level: CascadeLevel, usage: dict) -> float:
    """One call's USD cost, from the level's configured prices. Never estimated."""
    return estimate_cost(level, usage)
