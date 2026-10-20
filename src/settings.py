"""Environment-backed application settings: secrets, endpoints and runtime flags only.

Model selection, per-level JEV thresholds and prices live in ``config/cascade.yaml``
(see :mod:`src.cascade_config`). This module deliberately holds only what a deployment
must keep out of version control: credentials and endpoint URLs.
"""

from __future__ import annotations

import json
import hmac
import os
from dataclasses import dataclass, field
from functools import lru_cache

from src.jev_dialects import DEFAULT_JEV_DIALECT, JEV_DIALECTS
from src.shadow import (
    DEFAULT_SHADOW_MODE,
    SHADOW_MODES,
)
from src.shadow import DEFAULT_SHADOW_STORE_PATH as DEFAULT_SHADOW_STORE_PATH

SECRET_FIELDS = ("api_key", "jev_api_key", "openai_api_key", "anthropic_api_key")

#: Body size ceiling. One mebibyte is generous for a chat completion (a page of
#: text is ~2 KB) and small enough that a hostile body cannot exhaust memory.
DEFAULT_MAX_PAYLOAD_BYTES = 1_048_576

#: Absolute ceiling for MAX_OUTBOUND_CALLS: 3 levels with at most one retry each.
OUTBOUND_CALLS_HARD_LIMIT = 6

#: Streaming modes. ``buffered_cascade`` is the only one that evaluates every
#: answer; ``passthrough_l3`` trades evaluation for time-to-first-token.
STREAM_MODES = ("buffered_cascade", "passthrough_l3")
DEFAULT_STREAM_MODE = "buffered_cascade"

#: Shadow mode lives in :mod:`src.shadow`; the constants are re-exported here so a
#: deployment only has to learn one module for its runtime flags.
CASCADE_MODES = SHADOW_MODES
DEFAULT_CASCADE_MODE = DEFAULT_SHADOW_MODE
DEFAULT_SHADOW_STORE_PATH = DEFAULT_SHADOW_STORE_PATH


def _json_object(name: str, default: str) -> dict[str, object]:
    raw = os.getenv(name, default)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must contain valid JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _optional_positive_int(name: str, *, ceiling: int) -> int | None:
    """Read an optional positive integer env var, refusing values above ``ceiling``.

    An unbounded call budget is not a tuning knob: it is a way to spend an
    unlimited amount of provider quota by mistake, so it fails at startup.
    """
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None
    if value < 1:
        raise ValueError(f"{name} must be at least 1, got {value}")
    if value > ceiling:
        raise ValueError(
            f"{name} cannot exceed {ceiling} (three levels with at most one retry each); "
            f"got {value}"
        )
    return value


def _stream_mode() -> str:
    """STREAM_MODE, validated at startup rather than failing on the first stream."""
    raw = os.getenv("STREAM_MODE", "").strip() or DEFAULT_STREAM_MODE
    if raw not in STREAM_MODES:
        raise ValueError(
            f"STREAM_MODE must be one of: {', '.join(STREAM_MODES)}; got {raw!r}"
        )
    return raw


def _cascade_mode() -> str:
    """CASCADE_MODE, validated at startup so a typo cannot silently disable shadowing."""
    raw = (os.getenv("CASCADE_MODE", "").strip() or DEFAULT_CASCADE_MODE).lower()
    if raw not in CASCADE_MODES:
        raise ValueError(
            f"CASCADE_MODE must be one of: {', '.join(CASCADE_MODES)}; got {raw!r}"
        )
    return raw


def _jev_dialect() -> str:
    """JEV_DIALECT, validated at startup: an unknown dialect must not reach a request."""
    raw = (os.getenv("JEV_DIALECT", "").strip() or DEFAULT_JEV_DIALECT).lower()
    if raw not in JEV_DIALECTS:
        raise ValueError(
            f"JEV_DIALECT must be one of: {', '.join(JEV_DIALECTS)}; got {raw!r}"
        )
    return raw


def _flag(name: str) -> bool:
    return os.getenv(name, "false").strip().lower() in {"1", "true", "yes"}


def _client_keys(legacy_key: str) -> tuple[tuple[str, str], ...]:
    """Client keys as ``(label, key)`` pairs.

    ``API_KEYS`` (``label:key,label:key``) wins over ``API_KEY``; a bare entry is
    given a generated label so a single-key list still works. Labels are what the
    rate limiter counts against, so duplicates would silently share a budget.
    """
    raw = os.getenv("API_KEYS", "").strip()
    if not raw:
        return (("default", legacy_key),) if legacy_key else ()
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for position, entry in enumerate(raw.split(","), start=1):
        entry = entry.strip()
        if not entry:
            raise ValueError("API_KEYS must not contain empty entries")
        label, separator, key = entry.partition(":")
        if not separator:
            label, key = f"key-{position}", entry
        label, key = label.strip(), key.strip()
        if not label or not key:
            raise ValueError(
                f"API_KEYS entry {position} must be 'label:key' or a bare key, got {entry!r}"
            )
        if label in seen:
            raise ValueError(
                f"API_KEYS has a duplicate label {label!r}; labels identify a client "
                "budget and must be unique"
            )
        seen.add(label)
        pairs.append((label, key))
    return tuple(pairs)


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None
    if value < 1:
        raise ValueError(f"{name} must be at least 1, got {value}")
    return value


def _rate_limit_requests() -> int:
    """0 (the default) disables limiting entirely."""
    raw = os.getenv("RATE_LIMIT_REQUESTS", "").strip()
    if not raw:
        return 0
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"RATE_LIMIT_REQUESTS must be an integer, got {raw!r}") from None
    if value < 0:
        raise ValueError(f"RATE_LIMIT_REQUESTS must be 0 (disabled) or positive, got {value}")
    return value


def _rate_limit_window_seconds() -> float:
    raw = os.getenv("RATE_LIMIT_WINDOW_SECONDS", "").strip()
    if not raw:
        return 60.0
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(
            f"RATE_LIMIT_WINDOW_SECONDS must be a number of seconds, got {raw!r}"
        ) from None
    if value <= 0:
        raise ValueError(f"RATE_LIMIT_WINDOW_SECONDS must be positive, got {value}")
    return value


def _unsupported_features_level() -> int:
    """Level used for payloads the cascade cannot evaluate (tools, schemas, images, n>1)."""
    raw = os.getenv("UNSUPPORTED_FEATURES_LEVEL", "").strip() or "3"
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(
            f"UNSUPPORTED_FEATURES_LEVEL must be a cascade level number (1-3), got {raw!r}"
        ) from None
    if not 1 <= value <= 3:
        raise ValueError(
            f"UNSUPPORTED_FEATURES_LEVEL must be a cascade level number (1-3), got {value}"
        )
    return value


def _redacted(value: object) -> object:
    if isinstance(value, str) and value:
        return "***redacted***"
    return value


@dataclass(frozen=True)
class Settings:
    host: str = "0.0.0.0"
    port: int = 8000
    api_key: str = ""
    jev_api_key: str = ""
    jev_api_url: str = ""
    #: Wire dialect: "simple" (legacy) or "typesafe_decisions" (TypeSafe/OpenRouter).
    jev_dialect: str = DEFAULT_JEV_DIALECT
    #: System One model id, required by typesafe_decisions (e.g. "jev-1.13").
    jev_model: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    provider_base_urls: dict[str, str] = field(default_factory=lambda: {
        "openai": "https://api.openai.com/v1",
        "anthropic": "https://api.anthropic.com/v1",
    })
    log_content: bool = False
    validate_models_on_startup: bool = False
    #: Hard ceiling on outbound provider calls per request. None = derive it from
    #: the cascade (2 x max_hops), which is the documented default.
    max_outbound_calls: int | None = None
    #: ``buffered_cascade`` (evaluate then replay) or ``passthrough_l3`` (no evaluation).
    stream_mode: str = DEFAULT_STREAM_MODE
    #: Level (1-3) that answers payloads the cascade cannot evaluate, unchanged.
    unsupported_features_level: int = 3
    #: ``active`` or ``shadow``; shadow always answers with the final level.
    cascade_mode: str = DEFAULT_CASCADE_MODE
    #: Persist ``(cascade, direct)`` pairs locally. Contains prompts: opt-in only.
    shadow_store: bool = False
    shadow_store_path: str = DEFAULT_SHADOW_STORE_PATH
    #: ``(label, key)`` pairs. Empty means no client authentication is required.
    client_keys: tuple[tuple[str, str], ...] = ()
    #: Requests allowed per key per window. 0 (the default) disables limiting.
    rate_limit_requests: int = 0
    rate_limit_window_seconds: float = 60.0
    #: Body size ceiling in bytes; larger bodies are refused before parsing.
    max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES

    def key_label(self, supplied: str) -> str | None:
        """The label for a presented key, or None when it is not valid."""
        for label, key in self.client_keys:
            if hmac.compare_digest(supplied, key):
                return label
        return None

    def __post_init__(self) -> None:
        # ``api_key`` is the pre-Phase-8 single key. Treat it as a one-entry key
        # list so every existing deployment keeps working (and gains the label).
        if not self.client_keys and self.api_key:
            object.__setattr__(self, "client_keys", (("default", self.api_key),))

    def api_key_for(self, provider: str) -> str:
        """Credential for ``provider``: settings attribute first, then ``<PROVIDER>_API_KEY``."""
        key_name = f"{provider.upper().replace('-', '_')}_API_KEY".lower()
        value = getattr(self, key_name, "") or os.getenv(key_name.upper(), "")
        if not value and provider == "openai":
            value = self.openai_api_key
        if not value and provider == "anthropic":
            value = self.anthropic_api_key
        return value

    def __repr__(self) -> str:
        fields = ", ".join(
            f"{name}={self._shown(name)!r}" for name in self.__dataclass_fields__
        )
        return f"Settings({fields})"

    def _shown(self, name: str):
        """Mask secrets wherever they live, including inside the key list."""
        value = getattr(self, name)
        if name == "client_keys":
            # Labels are safe to print; the keys they name are not.
            return tuple((label, "***redacted***") for label, _ in value)
        return _redacted(value)
    __str__ = __repr__

    @classmethod
    def from_env(cls) -> "Settings":
        port = int(os.getenv("PORT", "8000"))
        if not 1 <= port <= 65535:
            raise ValueError("PORT must be between 1 and 65535")

        configured_urls = _json_object("PROVIDER_BASE_URLS_JSON", "{}")
        urls: dict[str, str] = {}
        for provider, url in configured_urls.items():
            if not isinstance(provider, str) or not isinstance(url, str):
                raise ValueError("PROVIDER_BASE_URLS_JSON must map provider names to URL strings")
            urls[provider] = url.rstrip("/")
        openai_base_url = os.getenv("OPENAI_BASE_URL", "").strip().rstrip("/") or "https://api.openai.com/v1"
        urls.setdefault("openai", openai_base_url)
        anthropic_base_url = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1").strip().rstrip("/")
        urls.setdefault("anthropic", anthropic_base_url)

        max_outbound_calls = _optional_positive_int("MAX_OUTBOUND_CALLS", 
                                                    ceiling=OUTBOUND_CALLS_HARD_LIMIT)
        stream_mode = _stream_mode()
        unsupported_features_level = _unsupported_features_level()
        cascade_mode = _cascade_mode()
        legacy_key = os.getenv("API_KEY", "")
        client_keys = _client_keys(legacy_key)

        return cls(
            host=os.getenv("HOST", "0.0.0.0"),
            port=port,
            api_key=legacy_key,
            jev_api_key=os.getenv("JEV_API_KEY", ""),
            jev_api_url=os.getenv("JEV_API_URL", ""),
            jev_dialect=_jev_dialect(),
            jev_model=os.getenv("JEV_MODEL", "").strip(),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            openai_base_url=openai_base_url,
            provider_base_urls=urls,
            log_content=os.getenv("LOG_CONTENT", "false").strip().lower() in {"1", "true", "yes"},
            validate_models_on_startup=os.getenv("VALIDATE_MODELS_ON_STARTUP", "false").strip().lower()
            in {"1", "true", "yes"},
            max_outbound_calls=max_outbound_calls,
            stream_mode=stream_mode,
            unsupported_features_level=unsupported_features_level,
            cascade_mode=cascade_mode,
            shadow_store=_flag("SHADOW_STORE"),
            shadow_store_path=os.getenv("SHADOW_STORE_PATH", "").strip()
            or DEFAULT_SHADOW_STORE_PATH,
            client_keys=client_keys,
            rate_limit_requests=_rate_limit_requests(),
            rate_limit_window_seconds=_rate_limit_window_seconds(),
            max_payload_bytes=_positive_int("MAX_PAYLOAD_BYTES", DEFAULT_MAX_PAYLOAD_BYTES),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
