"""The suite's own contract: what runs by default, and what must not.

The plan requires that anything touching a real provider or JEV lives in
``tests/integration/`` behind ``@pytest.mark.integration`` and is excluded from
the default run. That promise is only worth something if something checks it —
otherwise a future integration test would quietly start making live calls in CI.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def pytest_config() -> dict:
    """The effective pytest configuration, as pytest itself resolved it."""
    import tomllib

    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["tool"]["pytest"]["ini_options"]


def registered_markers() -> set[str]:
    markers = pytest_config().get("markers", [])
    return {marker.split(":", 1)[0].strip() for marker in markers}


def test_the_integration_marker_is_registered():
    """An unregistered marker warns; a registered one documents itself with `--markers`."""
    assert "integration" in registered_markers()


def test_each_marker_carries_a_description():
    markers = pytest_config()["markers"]

    assert all(":" in marker for marker in markers), markers


def test_integration_tests_are_excluded_by_default():
    addopts = pytest_config().get("addopts", "")

    assert "-m" in addopts
    assert "not integration" in addopts


def test_the_integration_suite_is_excluded_from_the_default_run():
    """Ask pytest what it would actually run, not what the config claims."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "tests/integration"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )

    assert result.returncode in {0, 5}, result.stdout + result.stderr
    assert "deselected" in result.stdout or "no tests collected" in result.stdout


def test_the_integration_suite_is_reachable_on_demand():
    """The exclusion must be a marker filter, not an empty directory."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-m", "integration",
         "tests/integration", "--override-ini=addopts="],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "test_live_cascade.py" in result.stdout


def test_integration_tests_are_skipped_rather_than_failed_without_credentials():
    """Without opt-in the suite must report *skipped*, never a live call or a crash."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-m", "integration", "tests/integration",
         "--override-ini=addopts=", "--no-header", "-p", "no:cacheprovider"],
        cwd=ROOT, capture_output=True, text=True, check=False,
        env=_without_credentials(),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "skipped" in result.stdout
    assert "passed" not in result.stdout


def _without_credentials() -> dict[str, str]:
    import os

    env = {key: value for key, value in os.environ.items()
           if not key.endswith("_API_KEY") and key != "CASCADE_CONFIG_PATH"}
    # Make the "no credentials" case explicit even if the caller's shell is dirty.
    env["CASCADE_RUN_INTEGRATION"] = "0"
    return env


@pytest.mark.parametrize("name", ["__init__.py", "conftest.py"])
def test_the_integration_package_is_well_formed(name):
    assert (ROOT / "tests" / "integration" / name).is_file()
