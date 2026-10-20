"""Shared test fixtures: a valid cascade configuration and matching settings.

The numbers below are test fixtures only. They are NOT real provider prices and
must never be copied into a deployment configuration.
"""

from __future__ import annotations

import copy
import textwrap
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.cascade_config import CascadeConfig, load_cascade_config
from src.settings import Settings

TEST_API_KEYS = {"openai": "test-openai-key", "anthropic": "test-anthropic-key"}

_VALID_YAML = textwrap.dedent(
    """\
    cascade:
      max_hops: 3
      levels:
        - level: 1
          provider: openai
          model: model-one
          params: {temperature: 0.2, max_tokens: 1024, timeout_ms: 30000}
          pricing: {input_per_mtok_usd: 1.0, output_per_mtok_usd: 2.0}
          jev_threshold: 0.80
        - level: 2
          provider: anthropic
          model: model-two
          params: {temperature: 0.2, max_tokens: 2048, timeout_ms: 45000}
          pricing: {input_per_mtok_usd: 3.0, output_per_mtok_usd: 6.0}
          jev_threshold: 0.70
        - level: 3
          provider: anthropic
          model: model-three
          params: {temperature: 0.2, max_tokens: 4096, timeout_ms: 90000}
          pricing: {input_per_mtok_usd: 9.0, output_per_mtok_usd: 12.0}
    jev:
      timeout_ms: 1500
      failure_policy: escalate
    """
)


def base_config_dict() -> dict[str, Any]:
    """Return a fresh, valid configuration as a plain dictionary."""
    return copy.deepcopy(yaml.safe_load(_VALID_YAML))


def write_config_data(tmp_path: Path, data: dict[str, Any]) -> Path:
    path = tmp_path / "cascade.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def load_config(path: Path, *, env: dict[str, str] | None = None,
                api_keys: dict[str, str] | None = None) -> CascadeConfig:
    return load_cascade_config(
        path,
        env={} if env is None else env,
        api_keys=TEST_API_KEYS if api_keys is None else api_keys,
    )


@pytest.fixture
def config_data() -> dict[str, Any]:
    return base_config_dict()


@pytest.fixture
def cascade_config(tmp_path: Path, config_data: dict[str, Any]) -> CascadeConfig:
    return load_config(write_config_data(tmp_path, config_data))


@pytest.fixture
def settings() -> Settings:
    return Settings(
        jev_api_key="test-jev-key",
        jev_api_url="https://jev.test/evaluate",
        openai_api_key=TEST_API_KEYS["openai"],
        anthropic_api_key=TEST_API_KEYS["anthropic"],
    )
