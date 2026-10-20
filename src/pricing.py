"""Optional `pricing.yaml`: a finance-owned price table, keyed by model.

`cascade.yaml` already prices each level, which keeps the cascade self-contained.
This module adds the other arrangement: prices live in their own file, keyed by
``provider/model`` rather than by level number, so changing which model sits at a
level can never silently pair a model with another model's prices.

Two deliberate rules:

* The file is **all or nothing**. A partial table is refused instead of silently
  keeping some YAML prices and replacing others: a mixed source is exactly the
  kind of number nobody can audit later.
* Prices are never assumed. A ``TODO`` or an unknown model is an error, because a
  fabricated price produces a confident, wrong savings figure.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.cascade_config import (
    CascadeConfig,
    CascadeConfigError,
    CascadeLevel,
    LevelPricing,
)

#: Environment variable pointing at the price table.
PRICING_PATH_ENV = "PRICING_PATH"

#: Default location, next to `cascade.yaml`.
DEFAULT_PRICING_PATH = Path("config") / "pricing.yaml"

_PRICE_KEYS = ("input_per_mtok_usd", "output_per_mtok_usd")


class PricingError(CascadeConfigError):
    """A price table that cannot be trusted.

    Subclasses :class:`CascadeConfigError` so startup fails the same way for a bad
    price file as for a bad cascade file, with the same message and exit code.
    """


def _parse_pricing_entry(key: str, raw: Any, path: Path) -> LevelPricing:
    if not isinstance(raw, Mapping):
        raise PricingError(
            f"{key!r} in {path} must be a mapping with "
            f"{' and '.join(_PRICE_KEYS)}, found {type(raw).__name__}"
        )
    values: dict[str, float] = {}
    for price_key in _PRICE_KEYS:
        if price_key not in raw:
            raise PricingError(f"{key!r} in {path} is missing {price_key}")
        value = raw[price_key]
        if isinstance(value, str) and value.strip().upper() == "TODO":
            raise PricingError(
                f"{key!r} in {path} still has the placeholder 'TODO' for {price_key}; "
                "look up the real price before starting the proxy"
            )
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PricingError(
                f"{key!r} in {path} has a non-numeric {price_key}: {value!r}"
            )
        if value < 0:
            raise PricingError(
                f"{key!r} in {path} has a negative {price_key}: {value!r}; "
                "prices cannot be negative"
            )
        values[price_key] = float(value)
    return LevelPricing(**values)


def load_pricing(path: Path | str) -> dict[str, LevelPricing]:
    """Read a price table, keyed ``provider/model``.

    Raises :class:`PricingError` for a missing file, bad YAML, a TODO, a negative
    price, or a non-mapping entry. Never returns a partially-filled table.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PricingError(f"Cannot read the pricing file {path}: {exc}") from None
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise PricingError(f"Pricing file {path} is not valid YAML: {exc}") from None
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise PricingError(f"Pricing file {path} must be a YAML mapping of provider/model to prices")

    prices: dict[str, LevelPricing] = {}
    for key, entry in raw.items():
        if not isinstance(key, str) or "/" not in key:
            raise PricingError(
                f"Pricing key {key!r} in {path} must be 'provider/model' "
                "(for example 'openai/gpt-4o-mini')"
            )
        prices[key] = _parse_pricing_entry(key, entry, path)
    return prices


def _with_pricing(level: CascadeLevel, pricing: LevelPricing) -> CascadeLevel:
    return CascadeLevel(
        level=level.level,
        provider=level.provider,
        model=level.model,
        params=level.params,
        pricing=pricing,
        jev_threshold=level.jev_threshold,
    )


def apply_pricing_overrides(config: CascadeConfig, *, path: Path | str | None = None,
                            env: Mapping[str, str] | None = None) -> CascadeConfig:
    """Return ``config`` with prices taken from `pricing.yaml` when one is configured.

    With no file (no explicit ``path`` and no ``PRICING_PATH``) the configuration is
    returned unchanged: `cascade.yaml` remains the default source of prices.
    """
    environment = os.environ if env is None else env
    selected = path if path is not None else environment.get(PRICING_PATH_ENV)
    if not selected:
        return config

    resolved = Path(selected)
    prices = load_pricing(resolved)
    levels = config.levels
    missing = [level.full_name for level in levels if level.full_name not in prices]
    if missing:
        raise PricingError(
            f"Pricing file {resolved} does not price every configured level; missing: "
            f"{', '.join(missing)}. The table must cover all levels or none "
            "(a partial table mixes two sources of truth)."
        )
    unknown = sorted(set(prices) - {level.full_name for level in levels})
    if unknown:
        raise PricingError(
            f"Pricing file {resolved} prices models that no cascade level uses: "
            f"{', '.join(unknown)}. Remove them, or fix the model names in the cascade."
        )
    return config.model_copy(update={
        "cascade": config.cascade.model_copy(update={
            "levels": tuple(
                _with_pricing(level, prices[level.full_name]) for level in levels
            ),
        }),
    })
