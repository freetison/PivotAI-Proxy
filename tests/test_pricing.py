"""Phase 6: prices can live in `pricing.yaml`, and they must never be invented.

`cascade.yaml` already carries per-level prices (that is what Phase 0 settled).
`pricing.yaml` is an optional, explicit override: a separate file for the finance
owner to edit, keyed by `provider/model` rather than by level number, so changing
a level's model cannot silently pair it with another model's prices.

Rules under test:
* No file -> the prices in `cascade.yaml` are used, unchanged.
* File present -> it must cover every configured level, with no TODO left.
* A partial table is an error, not a silent mix of two sources.
"""

from __future__ import annotations

import textwrap

import pytest

from src.cascade_config import CascadeConfigError
from src.pricing import (
    DEFAULT_PRICING_PATH,
    PRICING_PATH_ENV,
    PricingError,
    apply_pricing_overrides,
    load_pricing,
)

from tests.conftest import base_config_dict, load_config, write_config_data

VALID_PRICING = textwrap.dedent(
    """\
    "openai/model-one":
      input_per_mtok_usd: 0.15
      output_per_mtok_usd: 0.6
    "anthropic/model-two":
      input_per_mtok_usd: 1.0
      output_per_mtok_usd: 5.0
    "anthropic/model-three":
      input_per_mtok_usd: 3.0
      output_per_mtok_usd: 15.0
    """
)


def write_pricing(tmp_path, text: str):
    path = tmp_path / "pricing.yaml"
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


def configured(tmp_path):
    return load_config(write_config_data(tmp_path, base_config_dict()))


# ------------------------------------------------------------------- the file

def test_prices_are_keyed_by_provider_and_model(tmp_path):
    prices = load_pricing(write_pricing(tmp_path, VALID_PRICING))

    assert set(prices) == {"openai/model-one", "anthropic/model-two", "anthropic/model-three"}
    assert prices["anthropic/model-two"].input_per_mtok_usd == 1.0
    assert prices["anthropic/model-two"].output_per_mtok_usd == 5.0


def test_a_todo_price_is_refused(tmp_path):
    path = write_pricing(tmp_path, VALID_PRICING.replace("0.15", "TODO"))

    with pytest.raises(PricingError, match="output_per_mtok_usd|input_per_mtok_usd"):
        load_pricing(path)


def test_a_negative_price_is_refused(tmp_path):
    path = write_pricing(tmp_path, VALID_PRICING.replace("0.15", "-1"))

    with pytest.raises(PricingError, match="cannot be negative"):
        load_pricing(path)


def test_a_missing_file_names_the_path(tmp_path):
    with pytest.raises(PricingError, match="absent\\.yaml"):
        load_pricing(tmp_path / "absent.yaml")


def test_a_non_mapping_entry_is_refused(tmp_path):
    path = write_pricing(tmp_path, '"openai/model-one": 0.5\n')

    with pytest.raises(PricingError, match="must be a mapping"):
        load_pricing(path)


def test_a_missing_price_key_is_named(tmp_path):
    path = write_pricing(tmp_path, '"openai/model-one":\n  input_per_mtok_usd: 0.15\n')

    with pytest.raises(PricingError, match="missing output_per_mtok_usd"):
        load_pricing(path)


def test_a_non_numeric_price_is_refused(tmp_path):
    path = write_pricing(tmp_path, VALID_PRICING.replace("0.15", '"cheap"'))

    with pytest.raises(PricingError, match="non-numeric input_per_mtok_usd"):
        load_pricing(path)


def test_invalid_yaml_is_reported_as_such(tmp_path):
    path = write_pricing(tmp_path, '"openai/model-one": [unclosed\n')

    with pytest.raises(PricingError, match="not valid YAML"):
        load_pricing(path)


def test_an_empty_file_prices_nothing(tmp_path):
    assert load_pricing(write_pricing(tmp_path, "")) == {}


def test_a_non_mapping_file_is_refused(tmp_path):
    path = write_pricing(tmp_path, "- a list, not a mapping\n")

    with pytest.raises(PricingError, match="must be a YAML mapping"):
        load_pricing(path)


def test_a_key_without_a_provider_prefix_is_refused(tmp_path):
    path = write_pricing(tmp_path, '"model-one":\n  input_per_mtok_usd: 1\n  output_per_mtok_usd: 1\n')

    with pytest.raises(PricingError, match="must be 'provider/model'"):
        load_pricing(path)


def test_the_example_file_ships_with_todos_so_nothing_is_invented():
    from pathlib import Path

    example = Path("config/pricing.example.yaml")
    assert example.is_file()
    text = example.read_text(encoding="utf-8")
    # The template must not contain a single real-looking price.
    assert "TODO" in text
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.endswith(("_usd: TODO", "_usd: TODO")):
            continue
        if "_usd:" in stripped:
            assert "TODO" in stripped, f"invented price in the template: {striped}"


# ------------------------------------------------------------------- applying

def test_without_a_pricing_file_the_yaml_prices_are_kept(tmp_path):
    config = configured(tmp_path)

    assert apply_pricing_overrides(config, path=None, env={}) == config


def test_a_pricing_file_replaces_the_yaml_prices(tmp_path):
    config = configured(tmp_path)

    updated = apply_pricing_overrides(config, path=write_pricing(tmp_path, VALID_PRICING))

    assert updated.levels[0].pricing.output_per_mtok_usd == 0.6
    assert updated.levels[2].pricing.input_per_mtok_usd == 3.0
    # Everything else about the levels is untouched.
    assert updated.levels[0].model == config.levels[0].model
    assert updated.levels[0].jev_threshold == config.levels[0].jev_threshold
    assert updated.failure_policy == config.failure_policy


def test_the_env_variable_selects_the_file(tmp_path):
    config = configured(tmp_path)
    path = write_pricing(tmp_path, VALID_PRICING)

    updated = apply_pricing_overrides(config, env={PRICING_PATH_ENV: str(path)})

    assert updated.levels[1].pricing.input_per_mtok_usd == 1.0


def test_the_env_variable_pointing_nowhere_is_an_error(tmp_path):
    config = configured(tmp_path)

    with pytest.raises(PricingError, match="absent.yaml"):
        apply_pricing_overrides(config, env={PRICING_PATH_ENV: str(tmp_path / "absent.yaml")})


def test_a_partial_table_is_refused_instead_of_mixing_sources(tmp_path):
    config = configured(tmp_path)
    path = write_pricing(tmp_path, VALID_PRICING.split('"anthropic/model-two"')[0])

    with pytest.raises(PricingError, match="anthropic/model-two"):
        apply_pricing_overrides(config, path=path)


def test_an_entry_for_an_unconfigured_model_is_reported(tmp_path):
    config = configured(tmp_path)
    path = write_pricing(tmp_path, VALID_PRICING + '"openai/unknown-model":\n'
                         "  input_per_mtok_usd: 1.0\n  output_per_mtok_usd: 1.0\n")

    with pytest.raises(PricingError, match="unknown-model"):
        apply_pricing_overrides(config, path=path)


def test_the_default_path_is_config_pricing_yaml():
    assert DEFAULT_PRICING_PATH.name == "pricing.yaml"
    assert DEFAULT_PRICING_PATH.parent.name == "config"
    assert PRICING_PATH_ENV == "PRICING_PATH"


def test_pricing_errors_are_configuration_errors():
    # Startup must fail the same way for a bad price file as for a bad cascade file.
    assert issubclass(PricingError, CascadeConfigError)
