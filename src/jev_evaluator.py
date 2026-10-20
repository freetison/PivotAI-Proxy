"""JEV HTTP client, failure policy and per-level approval threshold.

Two transports exist behind ``JevEvaluator``:

* ``simple`` — a plain ``httpx`` POST carrying ``{"prompt","response"}``.
* ``typesafe_decisions`` — the official ``typesafe_sdk`` pointed at OpenRouter,
  speaking System One (typed questions about ``state``).

Both must yield the same contract: a float in [0, 1] or ``None``. The failure
policy, the per-level threshold and the calibration harness never learn which
one produced the number.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any, Callable, Literal

import httpx

from src.cascade_config import CascadeConfig, FailurePolicy
from src.jev_dialects import (
    DEFAULT_QUESTION,
    DEFAULT_QUESTION_NAME,
    SIMPLE,
    TYPESAFE_DECISIONS,
    cost_from_answers,
    dialect_payload,
    score_from_answers,
    score_from_simple,
)
from src.settings import Settings

logger = logging.getLogger("ai_cascade_proxy.jev")

#: ``approved``/``rejected`` come from a real score; ``unavailable`` means none was produced.
JevStatus = Literal["approved", "rejected", "unavailable"]

#: OpenRouter's System One base URL; the SDK appends ``/v1/systemone``.
OPENROUTER_BASE_URL = "https://openrouter.ai/api"


class JevUnavailable(RuntimeError):
    """JEV produced no score and ``fail_closed`` forbids answering without one."""


@dataclass(frozen=True)
class EvaluationResult:
    approved: bool
    score: float | None = None
    fallback: bool = False
    status: JevStatus = "approved"
    #: What this verification cost, as reported by the provider. 0.0 when unknown.
    cost_usd: float = 0.0

    @property
    def available(self) -> bool:
        return self.status != "unavailable"


@dataclass(frozen=True)
class Measurement:
    """One evaluation attempt: what JEV said, and what it charged.

    ``cost_usd`` comes from the provider's own usage report and is 0.0 when it
    reports nothing — never a price invented from a table.
    """

    score: float | None
    cost_usd: float = 0.0

    @property
    def available(self) -> bool:
        return self.score is not None


class JevEvaluator:
    def __init__(self, settings: Settings, config: CascadeConfig | None = None,
                 client: httpx.AsyncClient | None = None,
                 transport: Any | None = None):
        self.settings = settings
        self.config = config
        self._client = client
        # Only the typesafe dialect uses it; tests inject an httpx2 transport here.
        self._transport = transport

    @property
    def timeout_ms(self) -> int:
        return self.config.jev_timeout_ms if self.config is not None else 1500

    @property
    def failure_policy(self) -> FailurePolicy:
        return self.config.failure_policy if self.config is not None else "escalate"

    @property
    def dialect(self) -> str:
        return self.config.jev_dialect if self.config is not None else self.settings.jev_dialect

    @property
    def model(self) -> str:
        return self.config.jev_model if self.config is not None else self.settings.jev_model

    @property
    def questions(self) -> dict[str, str]:
        if self.config is not None:
            return dict(self.config.jev_questions)
        return {DEFAULT_QUESTION_NAME: DEFAULT_QUESTION}

    @property
    def configured(self) -> bool:
        """Is there enough configured to attempt a call at all?"""
        if not self.settings.jev_api_key:
            return False
        if self.dialect == TYPESAFE_DECISIONS:
            # The SDK takes a base URL; the model is required by the API itself.
            return bool(self.model)
        return bool(self.settings.jev_api_url)

    async def measure(self, prompt: str, response: str, *, system: str = "",
                      context: str | None = None) -> Measurement:
        """Score ``response`` and report what the evaluation cost.

        ``context`` carries the client's system prompt for the dialects that can
        express it; the legacy dialect drops it, as it always has.
        """
        if self.dialect == TYPESAFE_DECISIONS:
            return await self._measure_system_one(prompt, response, system=system)
        return await self._measure_simple(prompt, response)

    async def score(self, prompt: str, response: str, *, system: str = "") -> float | None:
        """Return the raw JEV score in [0, 1], or None when JEV is unavailable.

        ``None`` means "no measurement", never "score zero": the caller's failure
        policy decides what to do, and the calibration harness counts it as
        unavailable.
        """
        return (await self.measure(prompt, response, system=system)).score

    # -- the policy, shared by both transports ----------------------------- #

    async def evaluate(self, prompt: str, response: str, *, threshold: float | None = None,
                       system: str = "") -> EvaluationResult:
        if threshold is None:
            raise ValueError("JEV evaluation requires the approval threshold of the current level")

        measurement = await self.measure(prompt, response, system=system)
        if measurement.score is None:
            result = self._on_unavailable()
        else:
            approved = measurement.score >= threshold
            result = EvaluationResult(approved=approved, score=measurement.score,
                                      status="approved" if approved else "rejected")
        # The verification is billed whether or not it approved anything, so the
        # cost travels with the verdict instead of being re-derived by the caller.
        return replace(result, cost_usd=measurement.cost_usd)

    def _on_unavailable(self) -> EvaluationResult:
        """Apply ``jev.failure_policy``. The score stays ``None`` in every branch."""
        policy = self.failure_policy
        if policy == "fail_open":
            logger.warning("JEV unavailable (fail_open): returning the candidate answer")
            return EvaluationResult(approved=True, score=None, fallback=True,
                                    status="unavailable")
        if policy == "fail_closed":
            logger.error("JEV unavailable (fail_closed): refusing to answer")
            raise JevUnavailable(
                "JEV evaluation is unavailable and jev.failure_policy is 'fail_closed'; "
                "the response was not delivered"
            )
        logger.warning("JEV unavailable (escalate): escalating to the next level")
        return EvaluationResult(approved=False, score=None, fallback=True,
                                status="unavailable")

    # -- simple ------------------------------------------------------------ #

    async def _measure_simple(self, prompt: str, response: str) -> Measurement:
        if not self.configured:
            logger.warning("JEV is not configured; no score is available")
            return Measurement(score=None)

        owns_client = self._client is None
        client = self._client or httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout_ms / 1000)
        )
        try:
            result = await client.post(
                self.settings.jev_api_url,
                headers={"Authorization": f"Bearer {self.settings.jev_api_key}"},
                json=dialect_payload(
                    SIMPLE, model="", system="", prompt=prompt, response=response,
                    questions=self.questions,
                ),
            )
            result.raise_for_status()
            return Measurement(score=score_from_simple(result.json()))
        except Exception as exc:
            # Treat every client-side failure (timeout, 5xx, bad JSON, bad endpoint)
            # as "no score" and let the policy decide. Do not catch BaseException,
            # so cancellation and process interrupts propagate correctly. Omit the
            # exception message to avoid logging credentials embedded in URLs.
            logger.warning("JEV evaluation failed (%s); no score is available", type(exc).__name__)
            return Measurement(score=None)
        finally:
            if owns_client:
                await client.aclose()

    # -- typesafe_decisions ------------------------------------------------ #

    async def _measure_system_one(self, prompt: str, response: str, *,
                                  system: str) -> Measurement:
        if not self.configured:
            logger.warning("JEV is not configured; no score is available")
            return Measurement(score=None)

        body = dialect_payload(
            TYPESAFE_DECISIONS, model=self.model, system=system, prompt=prompt,
            response=response, questions=self.questions,
        )
        try:
            data = await self._call_system_one(body)
        except Exception as exc:
            logger.warning("JEV evaluation failed (%s); no score is available", type(exc).__name__)
            return Measurement(score=None)
        return Measurement(score=score_from_answers(data), cost_usd=cost_from_answers(data))

    async def _call_system_one(self, body: dict[str, Any]) -> Any:
        """Send one System One request and return the response body as a dict.

        Retries are disabled explicitly. The cascade counts outbound calls against
        a hard budget, so a client that retries on its own would spend money the
        budget never approved — the same reason the provider client sets
        ``retries=0``.
        """
        import httpx2
        from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

        client_kwargs: dict[str, Any] = {
            "api_key": self.settings.jev_api_key,
            "base_url": self.settings.jev_api_url or OPENROUTER_BASE_URL,
            "retry": RetryPolicy(max_retries=0, backoff_initial=0.0),
            "timeout": self.timeout_ms / 1000,
        }
        if self._transport is not None:
            client_kwargs["transport"] = self._transport

        async with AsyncTypeSafeClient(**client_kwargs) as client:
            reply = await client.system_one(
                state=body["state"],
                questions=body["questions"],
                model=body["model"],
                response_model=_SystemOneReply,
            )
        return reply.model_dump()


_SystemOneReply: Any = None


def _build_reply_model():
    """A Pydantic model that keeps ``usage.cost``, which the SDK's own type drops.

    ``typesafe_sdk.Usage`` declares only the token counts with ``extra="ignore"``,
    so OpenRouter's billed ``cost`` is discarded before we ever see it. Passing our
    own ``response_model`` to ``system_one`` is the supported way to keep it.
    """
    from pydantic import BaseModel, ConfigDict, Field

    class Usage(BaseModel):
        model_config = ConfigDict(extra="ignore", frozen=True)
        input_tokens: int | None = None
        output_tokens: int | None = None
        cost: float | None = None

    class NoulAnswer(BaseModel):
        model_config = ConfigDict(extra="ignore", frozen=True)
        type: str = "noul"
        noul: float | None = None

    class Reply(BaseModel):
        model_config = ConfigDict(extra="ignore", frozen=True)
        model: str = ""
        answers: dict[str, NoulAnswer] = Field(default_factory=dict)
        usage: Usage | None = None

    return Reply


_SystemOneReply = _build_reply_model()
