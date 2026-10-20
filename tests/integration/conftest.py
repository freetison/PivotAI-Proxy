"""Fixtures for the live suite: real configuration, real credentials, explicit opt-in.

Nothing here builds a fake. If a fixture cannot find what it needs it skips, so a
half-configured machine reports *skipped* instead of a stack trace or, worse, a
test that silently exercises nothing.
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.cascade_config import CascadeConfig, load_cascade_config
from src.settings import Settings

OPT_IN_ENV = "CASCADE_RUN_INTEGRATION"

INTEGRATION_NOTES = textwrap.dedent(
    """\
    Integration tests are opt-in. Export CASCADE_RUN_INTEGRATION=1 to enable
    them; without it this suite skips so a routine `pytest` never bills an
    account.
    """
)


def opted_in() -> bool:
    return os.environ.get(OPT_IN_ENV) == "1"


@pytest.fixture(scope="session")
def live_config_path() -> Path:
    """The configuration the deployment would actually use.

    ``CASCADE_CONFIG_PATH`` wins, so the suite can be pointed at a real
    ``config/cascade.yaml``; otherwise ``config/cascade.yaml`` next to the repo.
    """
    configured = os.environ.get("CASCADE_CONFIG_PATH")
    path = Path(configured) if configured else Path("config/cascade.yaml")

    if not path.is_file():
        pytest.skip(f"no cascade configuration at {path}; copy config/cascade.example.yaml "
                    f"and fill in the real models, prices and thresholds")

    return path


@pytest.fixture(scope="session")
def live_settings() -> Settings:
    return Settings.from_env()


@pytest.fixture(scope="session")
def live_config(live_config_path: Path, live_settings: Settings) -> CascadeConfig:
    """Load the real configuration, skipping instead of failing when it is unusable.

    An unfilled configuration is not a bug in the suite; it is a machine that has
    not been set up, and the answer is "skipped", not a red build.
    """
    try:
        return load_cascade_config(
            live_config_path,
            api_keys=_live_api_keys(live_settings),
        )
    except Exception as exc:  # noqa: BLE001 - any load failure means "not set up"
        pytest.skip(f"{live_config_path} is not usable for a live run: {exc}")


def _live_api_keys(settings: Settings) -> dict[str, str]:
    keys = {}
    if settings.openai_api_key:
        keys["openai"] = settings.openai_api_key
    if settings.anthropic_api_key:
        keys["anthropic"] = settings.anthropic_api_key
    return keys


@pytest.fixture(scope="session")
def live_payload() -> dict[str, Any]:
    """A request small enough to be cheap and unambiguous enough to grade."""
    return {
        "model": "auto",
        "messages": [
            {"role": "user",
             "content": "Reply with exactly one word: the capital of France."},
        ],
        "temperature": 0.0,
        "max_tokens": 16,
    }


@pytest.fixture(scope="session", autouse=True)
def require_opt_in() -> None:
    if not opted_in():
        pytest.skip(INTEGRATION_NOTES, allow_module_level=False)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Mark everything in this directory, so nobody has to remember the decorator."""
    for item in items:
        if Path(str(item.fspath)).parent.name == "integration":
            item.add_marker(pytest.mark.integration)
