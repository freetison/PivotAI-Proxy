"""Typed cascade configuration: the single source of truth for models, prices and thresholds.

`config/cascade.yaml` describes the model levels, per-level JEV approval thresholds,
per-level parameters and prices, and the JEV failure policy. Environment variables are
reserved for deployment overrides and secrets:

* secrets: ``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY`` (never in the YAML)
* path: ``CASCADE_CONFIG_PATH``
* per-level overrides: ``MODEL_LEVEL_N`` / ``JEV_THRESHOLD_LN`` (N = 1..3, env wins)
* ``JEV_TIMEOUT_MS`` / ``JEV_FAILURE_POLICY`` / ``JEV_DIALECT`` / ``JEV_MODEL``

There is intentionally no hot reload: changing the configuration requires a restart.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.jev_dialects import (
    DEFAULT_JEV_DIALECT,
    DEFAULT_QUESTION,
    DEFAULT_QUESTION_NAME,
    JEV_DIALECTS,
    TYPESAFE_DECISIONS,
)

logger = logging.getLogger("ai_cascade_proxy.config")

CASCADE_CONFIG_PATH_ENV = "CASCADE_CONFIG_PATH"
DEFAULT_CASCADE_CONFIG_PATH = Path("config") / "cascade.yaml"

MAX_HOPS_HARD_LIMIT = 3
REQUIRED_LEVELS = 3
EVALUATED_LEVELS = 2
FINAL_LEVEL = 3
TODO_MARKER = "TODO"

KNOWN_PROVIDERS = ("anthropic", "openai")
PROVIDER_API_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}

FailurePolicy = Literal["fail_open", "escalate", "fail_closed"]
FAILURE_POLICIES: tuple[FailurePolicy, ...] = ("fail_open", "escalate", "fail_closed")

class CascadeConfigError(RuntimeError):
    """The cascade configuration is missing, unreadable or invalid."""

class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

class LevelParams(_Frozen):
    temperature: float
    max_tokens: int
    timeout_ms: int

class LevelPricing(_Frozen):
    input_per_mtok_usd: float
    output_per_mtok_usd: float

class CascadeLevel(_Frozen):
    level: int
    provider: str
    model: str
    params: LevelParams
    pricing: LevelPricing
    jev_threshold: float | None = None

    @property
    def full_name(self) -> str:
        """Provider-qualified model id, as accepted by ``ModelProvider``."""
        return f"{self.provider}/{self.model}"

    @property
    def provider_params(self) -> dict[str, Any]:
        """Parameters always forwarded to the provider for this level."""
        return {"temperature": self.params.temperature, "max_tokens": self.params.max_tokens}

    @property
    def is_evaluated(self) -> bool:
        return self.level != FINAL_LEVEL

class JevSettings(_Frozen):
    timeout_ms: int = 1500
    failure_policy: FailurePolicy = "escalate"
    #: Wire dialect of the evaluator. ``simple`` is the legacy two-string contract.
    dialect: str = DEFAULT_JEV_DIALECT
    #: System One model id (e.g. "jev-1.13"); required by ``typesafe_decisions``.
    model: str = ""
    #: Named yes/no questions asked in the ``typesafe_decisions`` dialect. The score
    #: is the minimum answer, so every question is a requirement, not a vote.
    questions: dict[str, str] = Field(default_factory=lambda: {DEFAULT_QUESTION_NAME: DEFAULT_QUESTION})

    @field_validator("dialect")
    @classmethod
    def _known_dialect(cls, value: str) -> str:
        if value not in JEV_DIALECTS:
            raise ValueError(
                f"jev.dialect must be one of: {', '.join(JEV_DIALECTS)}; got {value!r}"
            )
        return value

    @field_validator("questions")
    @classmethod
    def _non_empty_questions(cls, value: dict[str, str]) -> dict[str, str]:
        if not value:
            raise ValueError(
                "jev.questions must define at least one question; the proxy cannot "
                "approve an answer it never asked about"
            )
        blank = [name for name, text in value.items() if not str(text).strip()]
        if blank:
            raise ValueError(
                f"jev.questions.{blank[0]} is empty; every question needs instructions"
            )
        return value

    @model_validator(mode="after")
    def _model_required_for_system_one(self):
        if self.dialect == TYPESAFE_DECISIONS and not self.model.strip():
            raise ValueError(
                "jev.model is required when jev.dialect is 'typesafe_decisions' "
                "(for example: jev-1.13); the API rejects a request without it"
            )
        return self

class _CascadeSettings(_Frozen):
    max_hops: int = MAX_HOPS_HARD_LIMIT
    levels: list[CascadeLevel]

class CascadeConfig(_Frozen):
    cascade: _CascadeSettings
    jev: JevSettings

    # --- convenience views used by the engine, the API and telemetry ---------------
    @property
    def levels(self) -> list[CascadeLevel]:
        return self.cascade.levels

    @property
    def max_hops(self) -> int:
        return self.cascade.max_hops

    @property
    def final_level(self) -> CascadeLevel:
        return self.cascade.levels[-1]

    @property
    def model_names(self) -> tuple[str, ...]:
        return tuple(level.full_name for level in self.cascade.levels)

    @property
    def failure_policy(self) -> FailurePolicy:
        return self.jev.failure_policy

    @property
    def jev_timeout_ms(self) -> int:
        return self.jev.timeout_ms

    @property
    def jev_dialect(self) -> str:
        return self.jev.dialect

    @property
    def jev_model(self) -> str:
        return self.jev.model

    @property
    def jev_questions(self) -> dict[str, str]:
        """The named yes/no questions the evaluator asks, in configuration order."""
        return dict(self.jev.questions)

    def level_for_model(self, model: str) -> CascadeLevel | None:
        for level in self.cascade.levels:
            if level.full_name == model:
                return level
        return None

    def level_for_number(self, number: int) -> CascadeLevel | None:
        """The level with this 1-based number, or None when it is not configured."""
        for level in self.cascade.levels:
            if level.level == number:
                return level
        return None

    def threshold_for_level(self, level: int) -> float | None:
        """JEV approval threshold of an evaluated level, or ``None`` for the final level."""
        entry = self.cascade.levels[level - 1]
        return entry.jev_threshold

# ---------------------------------------------------------------------------------
# Environment overrides
# ---------------------------------------------------------------------------------
def _split_model_value(value: str) -> tuple[str, str]:
    if "/" in value:
        provider, model = value.split("/", 1)
        return provider.strip(), model.strip()
    return "", value.strip()

def _apply_env_overrides(raw: dict[str, Any], env: dict[str, str]) -> None:
    levels: list[dict[str, Any]] = raw["cascade"]["levels"]

    for index, entry in enumerate(levels):
        level_number = index + 1

        model_override = env.get(f"MODEL_LEVEL_{level_number}", "").strip()
        if model_override:
            provider, model = _split_model_value(model_override)
            if provider:
                entry["provider"] = provider
            entry["model"] = model

        threshold_override = env.get(f"JEV_THRESHOLD_L{level_number}", "").strip()
        if threshold_override:
            try:
                entry["jev_threshold"] = float(threshold_override)
            except ValueError:
                raise CascadeConfigError(
                    f"JEV_THRESHOLD_L{level_number} must be a number, got {threshold_override!r}"
                ) from None

    timeout_override = env.get("JEV_TIMEOUT_MS", "").strip()
    if timeout_override:
        try:
            raw.setdefault("jev", {})["timeout_ms"] = int(timeout_override)
        except ValueError:
            raise CascadeConfigError(
                f"JEV_TIMEOUT_MS must be an integer number of milliseconds, got {timeout_override!r}"
            ) from None

    policy_override = env.get("JEV_FAILURE_POLICY", "").strip()
    if policy_override:
        if policy_override not in FAILURE_POLICIES:
            raise CascadeConfigError(
                f"JEV_FAILURE_POLICY must be one of: {', '.join(FAILURE_POLICIES)}; "
                f"got {policy_override!r}"
            )
        raw.setdefault("jev", {})["failure_policy"] = policy_override

    dialect_override = env.get("JEV_DIALECT", "").strip()
    if dialect_override:
        if dialect_override not in JEV_DIALECTS:
            raise CascadeConfigError(
                f"JEV_DIALECT must be one of: {', '.join(JEV_DIALECTS)}; "
                f"got {dialect_override!r}"
            )
        raw.setdefault("jev", {})["dialect"] = dialect_override

    model_override = env.get("JEV_MODEL", "").strip()
    if model_override:
        raw.setdefault("jev", {})["model"] = model_override

# ---------------------------------------------------------------------------------
# Validation rules (each one fails with a specific, actionable message)
# ---------------------------------------------------------------------------------
def _reject_todo(value: Any, location: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_todo(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_todo(item, f"{location}[{index}]")
    elif isinstance(value, str) and TODO_MARKER in value.upper():
        raise CascadeConfigError(
            f"{location.lstrip('.')} is still the placeholder {value!r}; fill in every TODO before starting "
            "the proxy (see config/cascade.example.yaml)"
        )

def _validate_level_structure(raw: dict[str, Any]) -> None:
    cascade = raw.get("cascade")
    if not isinstance(cascade, dict):
        raise CascadeConfigError("Missing 'cascade' section in the cascade configuration")
    levels = cascade.get("levels")
    if not isinstance(levels, list):
        raise CascadeConfigError("'cascade.levels' must be a list of three model levels")
    if len(levels) != REQUIRED_LEVELS:
        raise CascadeConfigError(
            f"Cascade configuration must define exactly 3 levels, found {len(levels)}"
        )
    numbers = [entry.get("level") if isinstance(entry, dict) else None for entry in levels]
    if numbers != [1, 2, 3]:
        raise CascadeConfigError(
            f"Cascade levels must be numbered 1, 2, 3 without gaps or reordering; found {numbers}"
        )

    max_hops = cascade.get("max_hops", MAX_HOPS_HARD_LIMIT)
    if not isinstance(max_hops, int) or isinstance(max_hops, bool) or not 1 <= max_hops <= MAX_HOPS_HARD_LIMIT:
        raise CascadeConfigError(
            f"cascade.max_hops must be an integer between 1 and {MAX_HOPS_HARD_LIMIT} "
            f"(the hard limit cannot be raised), found {max_hops!r}"
        )

def _validate_prices(levels: list[dict[str, Any]]) -> None:
    for entry in levels:
        pricing = entry.get("pricing")
        if not isinstance(pricing, dict):
            raise CascadeConfigError(f"Level {entry['level']} is missing the 'pricing' section")
        for key in ("input_per_mtok_usd", "output_per_mtok_usd"):
            value = pricing.get(key)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise CascadeConfigError(
                    f"Level {entry['level']} pricing.{key} must be a number (USD per million tokens)"
                )
            if value < 0:
                raise CascadeConfigError(
                    f"Level {entry['level']} pricing.{key} cannot be negative, found {value!r}"
                )

def _validate_thresholds(levels: list[dict[str, Any]]) -> None:
    for entry in levels:
        level_number = entry["level"]
        threshold = entry.get("jev_threshold")
        if level_number == FINAL_LEVEL:
            if threshold is not None:
                raise CascadeConfigError(
                    "The final level 3 must not define jev_threshold: it is returned without "
                    "evaluation. Remove 'jev_threshold' from cascade.levels[2]."
                )
            continue
        if threshold is None:
            raise CascadeConfigError(
                f"Cascade level {level_number} requires jev_threshold (levels 1 and 2 are "
                "evaluated; the final level is not)"
            )
        if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
            raise CascadeConfigError(
                f"Level {level_number} jev_threshold must be a number, found {threshold!r}"
            )
        if not 0 <= threshold <= 1:
            raise CascadeConfigError(
                f"Level {level_number} jev_threshold must be between 0 and 1, found {threshold!r}"
            )

def _validate_providers(levels: list[dict[str, Any]], api_keys: dict[str, str]) -> None:
    for entry in levels:
        provider = entry.get("provider")
        if provider not in KNOWN_PROVIDERS:
            raise CascadeConfigError(
                f"Cascade level {entry['level']} uses unknown provider {provider!r}; "
                f"expected one of: {', '.join(KNOWN_PROVIDERS)}"
            )
        if not api_keys.get(provider):
            raise CascadeConfigError(
                f"API key is not configured for provider: {provider} "
                f"(level {entry['level']} uses it). Set {PROVIDER_API_KEY_ENV[provider]}."
            )

def _build_config(raw: dict[str, Any]) -> CascadeConfig:
    try:
        return CascadeConfig.model_validate(raw)
    except Exception as exc:  # pydantic.ValidationError, but keep the dependency surface small
        raise CascadeConfigError(f"Invalid cascade configuration: {_describe_validation_error(exc)}") from None

def _describe_validation_error(exc: Exception) -> str:
    errors = getattr(exc, "errors", None)
    if callable(errors):
        details = []
        for error in errors():
            location = ".".join(str(part) for part in error.get("loc", ()))
            details.append(f"{location}: {error.get('msg', 'invalid value')}")
        if details:
            return "; ".join(details)
    return type(exc).__name__

def _resolve_api_keys(env: dict[str, str], api_keys: dict[str, str] | None) -> dict[str, str]:
    resolved = dict(api_keys or {})
    for provider, env_name in PROVIDER_API_KEY_ENV.items():
        if not resolved.get(provider):
            value = env.get(env_name, "").strip()
            if value:
                resolved[provider] = value
    return resolved

# ---------------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------------
def configuration_warnings(config: CascadeConfig) -> list[str]:
    """Non-fatal configuration smells. These never block startup."""
    warnings: list[str] = []

    for previous, current in zip(config.levels, config.levels[1:]):
        if previous.full_name == current.full_name:
            warnings.append(
                f"Cascade level {current.level} reuses the model of level {previous.level} "
                f"({current.full_name}); escalation will silently retry the same model"
            )

    first, second = config.levels[0], config.levels[1]
    if (
        first.jev_threshold is not None
        and second.jev_threshold is not None
        and second.jev_threshold > first.jev_threshold
    ):
        warnings.append(
            f"Level 2 jev_threshold ({second.jev_threshold}) is higher than level 1 "
            f"({first.jev_threshold}); escalation should be easier, not harder"
        )

    if config.failure_policy == "fail_closed":
        warnings.append(
            "jev.failure_policy='fail_closed' means every request fails with 503 whenever "
            "JEV is unreachable; make sure that outage is acceptable before deploying"
        )

    return warnings

def load_cascade_config(
    path: str | os.PathLike[str],
    *,
    env: dict[str, str] | None = None,
    api_keys: dict[str, str] | None = None,
    log_warnings: bool = True,
) -> CascadeConfig:
    """Load, override and validate ``path``. Raises :class:`CascadeConfigError` on any problem."""
    environment = dict(os.environ) if env is None else dict(env)
    config_path = Path(path)

    if not config_path.is_file():
        raise CascadeConfigError(
            f"Cascade configuration file not found: {config_path}. "
            f"Copy config/cascade.example.yaml and set {CASCADE_CONFIG_PATH_ENV} if it lives elsewhere."
        )

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise CascadeConfigError(f"Cascade configuration {config_path} is not valid YAML: {exc}") from None

    if not isinstance(raw, dict):
        raise CascadeConfigError(f"Cascade configuration {config_path} must be a YAML mapping")

    try:
        _validate_level_structure(raw)
        _apply_env_overrides(raw, environment)
        _reject_todo(raw, "")
        _validate_prices(raw["cascade"]["levels"])
        _validate_thresholds(raw["cascade"]["levels"])
        resolved_keys = _resolve_api_keys(environment, api_keys)
        _validate_providers(raw["cascade"]["levels"], resolved_keys)
        config = _build_config(raw)
    except CascadeConfigError as exc:
        raise CascadeConfigError(f"{exc} [{config_path}]") from None

    if log_warnings:
        for warning in configuration_warnings(config):
            logger.warning("%s", warning)

    return config

async def validate_models_against_providers(config: CascadeConfig, provider) -> None:
    """Optional startup check: every configured model id must exist at its provider.

    Disabled by default so the proxy can boot without network access. Any provider
    failure is reported as a configuration error, never silently ignored.
    """
    available: dict[str, set[str]] = {}
    for level in config.levels:
        if level.provider not in available:
            try:
                available[level.provider] = await provider.list_models(level.provider)
            except Exception as exc:
                detail = str(exc) or type(exc).__name__
                raise CascadeConfigError(
                    f"Could not list models for provider {level.provider}: {detail}. "
                    "Set VALIDATE_MODELS_ON_STARTUP=false to start without this check."
                ) from None
        models = available[level.provider]
        if models and level.model not in models:
            raise CascadeConfigError(
                f"Configured model {level.full_name} (level {level.level}) does not exist at "
                f"provider {level.provider}. Available ids matching '{level.model.split('-')[0]}': "
                f"{_similar_models(level.model, models)}"
            )


def _similar_models(model: str, models: set[str]) -> str:
    prefix = model.split("-")[0]
    matches = sorted(name for name in models if prefix in name)[:5]
    return ", ".join(matches) if matches else "none"


def load_cascade_config_from_env(
    *,
    env: dict[str, str] | None = None,
    api_keys: dict[str, str] | None = None,
) -> CascadeConfig:
    environment = dict(os.environ) if env is None else dict(env)
    path = environment.get(CASCADE_CONFIG_PATH_ENV, "").strip() or str(DEFAULT_CASCADE_CONFIG_PATH)
    config = load_cascade_config(path, env=environment, api_keys=api_keys)
    # Imported here: `pricing` depends on this module, so a top-level import would cycle.
    from src.pricing import apply_pricing_overrides

    return apply_pricing_overrides(config, env=environment)
