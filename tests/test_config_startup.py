"""Phase 0: startup validation must fail fast, with a clear message, before serving."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from src.cascade_config import CascadeConfigError, load_cascade_config, load_cascade_config_from_env
from src.main import create_app
from src.settings import Settings
from tests.conftest import TEST_API_KEYS, base_config_dict, write_config_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_PRICES = textwrap.dedent(
    """\
    "openai/model-one":
      input_per_mtok_usd: {one}
      output_per_mtok_usd: {one}
    "anthropic/model-two":
      input_per_mtok_usd: {two}
      output_per_mtok_usd: {two}
    "anthropic/model-three":
      input_per_mtok_usd: {three}
      output_per_mtok_usd: {three}
    """
)


def write_pricing(path: Path, *, one="0.5", two="1.5", three="2.5") -> Path:
    path.write_text(_PRICES.format(one=one, two=two, three=three), encoding="utf-8")
    return path


def test_startup_fails_fast_on_invalid_config(tmp_path: Path):
    data = base_config_dict()
    data["cascade"]["levels"][2]["jev_threshold"] = 0.5
    path = write_config_data(tmp_path, data)

    env = {
        **os.environ,
        "CASCADE_CONFIG_PATH": str(path),
        "OPENAI_API_KEY": TEST_API_KEYS["openai"],
        "ANTHROPIC_API_KEY": TEST_API_KEYS["anthropic"],
    }
    completed = subprocess.run(
        [sys.executable, "-c", "import src.main; src.main.main()"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert completed.returncode != 0
    combined = completed.stdout + completed.stderr
    assert "final level 3 must not define jev_threshold" in combined
    assert "Traceback" not in combined


def test_load_cascade_config_uses_environment_path(tmp_path: Path, monkeypatch):
    path = write_config_data(tmp_path, base_config_dict())
    monkeypatch.setenv("CASCADE_CONFIG_PATH", str(path))
    monkeypatch.setenv("OPENAI_API_KEY", TEST_API_KEYS["openai"])
    monkeypatch.setenv("ANTHROPIC_API_KEY", TEST_API_KEYS["anthropic"])

    config = load_cascade_config_from_env()
    assert config.model_names[2] == "anthropic/model-three"


def test_missing_config_path_is_reported(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CASCADE_CONFIG_PATH", str(tmp_path / "absent.yaml"))

    with pytest.raises(CascadeConfigError, match="absent.yaml"):
        load_cascade_config_from_env()


def test_threshold_out_of_range_is_rejected(tmp_path: Path):
    data = base_config_dict()
    data["cascade"]["levels"][0]["jev_threshold"] = 2.0
    path = write_config_data(tmp_path, data)

    with pytest.raises(CascadeConfigError, match="between 0 and 1"):
        load_cascade_config(path, env={}, api_keys=TEST_API_KEYS)


# ------------------------------------------------- pricing table at startup

def test_startup_fails_fast_on_a_bad_pricing_table(tmp_path: Path):
    """A bad price file must stop the proxy before it binds the port, like a bad cascade."""
    config_path = write_config_data(tmp_path, base_config_dict())
    pricing_path = write_pricing(tmp_path / "pricing.yaml", one="TODO")

    env = {
        **os.environ,
        "CASCADE_CONFIG_PATH": str(config_path),
        "PRICING_PATH": str(pricing_path),
        "OPENAI_API_KEY": TEST_API_KEYS["openai"],
        "ANTHROPIC_API_KEY": TEST_API_KEYS["anthropic"],
    }
    completed = subprocess.run(
        [sys.executable, "-c", "import src.main; src.main.main()"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert completed.returncode != 0
    combined = completed.stdout + completed.stderr
    assert "placeholder 'TODO'" in combined
    assert "Traceback" not in combined


def test_the_pricing_table_is_applied_by_the_env_loader(tmp_path: Path, monkeypatch):
    config_path = write_config_data(tmp_path, base_config_dict())
    pricing_path = write_pricing(tmp_path / "pricing.yaml")
    monkeypatch.setenv("CASCADE_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("PRICING_PATH", str(pricing_path))
    monkeypatch.setenv("OPENAI_API_KEY", TEST_API_KEYS["openai"])
    monkeypatch.setenv("ANTHROPIC_API_KEY", TEST_API_KEYS["anthropic"])

    config = load_cascade_config_from_env()

    assert [level.pricing.input_per_mtok_usd for level in config.levels] == [0.5, 1.5, 2.5]


def test_without_a_pricing_table_the_cascade_prices_win(tmp_path: Path, monkeypatch):
    config_path = write_config_data(tmp_path, base_config_dict())
    monkeypatch.setenv("CASCADE_CONFIG_PATH", str(config_path))
    monkeypatch.delenv("PRICING_PATH", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", TEST_API_KEYS["openai"])
    monkeypatch.setenv("ANTHROPIC_API_KEY", TEST_API_KEYS["anthropic"])

    config = load_cascade_config_from_env()

    assert [level.pricing.input_per_mtok_usd for level in config.levels] == [1.0, 3.0, 9.0]


def test_create_app_uses_injected_config(cascade_config):
    app = create_app(
        settings=Settings(openai_api_key=TEST_API_KEYS["openai"],
                          anthropic_api_key=TEST_API_KEYS["anthropic"]),
        cascade_config=cascade_config,
    )
    assert app.state.cascade_config is cascade_config


# --------------------------------------------------- optional runtime knobs (env)

def test_outbound_call_cap_defaults_to_twice_the_hop_limit(monkeypatch):
    monkeypatch.delenv("MAX_OUTBOUND_CALLS", raising=False)

    assert Settings.from_env().max_outbound_calls is None


def test_outbound_call_cap_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("MAX_OUTBOUND_CALLS", "4")

    assert Settings.from_env().max_outbound_calls == 4


@pytest.mark.parametrize("raw", ["0", "-2"])
def test_outbound_call_cap_refuses_a_value_below_one(monkeypatch, raw):
    monkeypatch.setenv("MAX_OUTBOUND_CALLS", raw)

    with pytest.raises(ValueError, match="must be at least 1"):
        Settings.from_env()


def test_outbound_call_cap_refuses_a_non_integer(monkeypatch):
    monkeypatch.setenv("MAX_OUTBOUND_CALLS", "six")

    with pytest.raises(ValueError, match="must be an integer"):
        Settings.from_env()


def test_an_invalid_port_is_refused(monkeypatch):
    monkeypatch.setenv("PORT", "70000")

    with pytest.raises(ValueError, match="PORT must be between 1 and 65535"):
        Settings.from_env()


def test_provider_base_urls_must_be_a_json_object(monkeypatch):
    monkeypatch.setenv("PROVIDER_BASE_URLS_JSON", '["not-an-object"]')

    with pytest.raises(ValueError, match="must be a JSON object"):
        Settings.from_env()


def test_provider_base_urls_reject_non_string_values(monkeypatch):
    monkeypatch.setenv("PROVIDER_BASE_URLS_JSON", '{"openai": 42}')

    with pytest.raises(ValueError, match="must map provider names to URL strings"):
        Settings.from_env()
