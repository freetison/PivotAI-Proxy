"""Phase 0: cascade.yaml is the single source of truth for models, thresholds and prices."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.cascade_config import (
    CascadeConfigError,
    configuration_warnings,
    load_cascade_config,
)
from src.settings import Settings
from tests.conftest import (
    TEST_API_KEYS,
    base_config_dict,
    load_config,
    write_config_data,
)


def test_loads_valid_configuration(cascade_config):
    assert [level.level for level in cascade_config.levels] == [1, 2, 3]
    assert cascade_config.model_names == (
        "openai/model-one",
        "anthropic/model-two",
        "anthropic/model-three",
    )
    assert cascade_config.levels[0].jev_threshold == 0.80
    assert cascade_config.levels[1].jev_threshold == 0.70
    assert cascade_config.levels[2].jev_threshold is None
    assert cascade_config.levels[1].params.max_tokens == 2048
    assert cascade_config.levels[1].pricing.input_per_mtok_usd == 3.0
    assert cascade_config.levels[1].pricing.output_per_mtok_usd == 6.0
    assert cascade_config.jev.timeout_ms == 1500
    assert cascade_config.jev.failure_policy == "escalate"
    assert cascade_config.max_hops == 3
    assert cascade_config.final_level.level == 3


def test_config_is_immutable(cascade_config):
    with pytest.raises(Exception):
        cascade_config.max_hops = 1  # type: ignore[misc]


def test_missing_config_file_reports_the_path(tmp_path: Path):
    missing = tmp_path / "nope.yaml"
    with pytest.raises(CascadeConfigError, match="nope.yaml"):
        load_config(missing)


def test_rejects_wrong_number_of_levels(tmp_path: Path, config_data: dict[str, Any]):
    config_data["cascade"]["levels"].pop()
    with pytest.raises(CascadeConfigError, match="exactly 3 levels"):
        load_config(write_config_data(tmp_path, config_data))


def test_rejects_level_numbering_gap(tmp_path: Path, config_data: dict[str, Any]):
    config_data["cascade"]["levels"][2]["level"] = 4
    with pytest.raises(CascadeConfigError, match="1, 2, 3"):
        load_config(write_config_data(tmp_path, config_data))


def test_rejects_missing_threshold_on_level_1(tmp_path: Path, config_data: dict[str, Any]):
    del config_data["cascade"]["levels"][0]["jev_threshold"]
    with pytest.raises(CascadeConfigError, match="level 1 requires jev_threshold"):
        load_config(write_config_data(tmp_path, config_data))


def test_rejects_threshold_on_final_level(tmp_path: Path, config_data: dict[str, Any]):
    config_data["cascade"]["levels"][2]["jev_threshold"] = 0.5
    with pytest.raises(CascadeConfigError, match="final level 3 must not define jev_threshold"):
        load_config(write_config_data(tmp_path, config_data))


@pytest.mark.parametrize("threshold", [1.5, -0.1])
def test_rejects_threshold_out_of_range(tmp_path: Path, config_data: dict[str, Any], threshold: float):
    config_data["cascade"]["levels"][0]["jev_threshold"] = threshold
    with pytest.raises(CascadeConfigError, match="between 0 and 1"):
        load_config(write_config_data(tmp_path, config_data))


def test_rejects_todo_model(tmp_path: Path, config_data: dict[str, Any]):
    config_data["cascade"]["levels"][1]["model"] = "TODO"
    with pytest.raises(CascadeConfigError, match="TODO"):
        load_config(write_config_data(tmp_path, config_data))


def test_rejects_todo_price(tmp_path: Path, config_data: dict[str, Any]):
    config_data["cascade"]["levels"][0]["pricing"]["output_per_mtok_usd"] = "TODO"
    with pytest.raises(CascadeConfigError, match="TODO"):
        load_config(write_config_data(tmp_path, config_data))


def test_rejects_negative_price(tmp_path: Path, config_data: dict[str, Any]):
    config_data["cascade"]["levels"][2]["pricing"]["input_per_mtok_usd"] = -1.0
    with pytest.raises(CascadeConfigError, match="negative"):
        load_config(write_config_data(tmp_path, config_data))


def test_rejects_unknown_provider(tmp_path: Path, config_data: dict[str, Any]):
    config_data["cascade"]["levels"][0]["provider"] = "acme"
    with pytest.raises(CascadeConfigError, match="unknown provider"):
        load_config(write_config_data(tmp_path, config_data))


def test_rejects_missing_api_key_for_used_provider(tmp_path: Path, config_data: dict[str, Any]):
    path = write_config_data(tmp_path, config_data)
    with pytest.raises(CascadeConfigError, match="API key is not configured for provider: anthropic"):
        load_cascade_config(path, env={}, api_keys={"openai": TEST_API_KEYS["openai"]})


def test_accepts_api_key_from_environment(tmp_path: Path, config_data: dict[str, Any]):
    path = write_config_data(tmp_path, config_data)
    config = load_cascade_config(
        path,
        env={"OPENAI_API_KEY": "env-openai", "ANTHROPIC_API_KEY": "env-anthropic"},
        api_keys={},
    )
    assert config.model_names[0] == "openai/model-one"


def test_rejects_max_hops_above_hard_limit(tmp_path: Path, config_data: dict[str, Any]):
    config_data["cascade"]["max_hops"] = 4
    with pytest.raises(CascadeConfigError, match="max_hops"):
        load_config(write_config_data(tmp_path, config_data))


def test_environment_overrides_model_and_threshold(tmp_path: Path, config_data: dict[str, Any]):
    path = write_config_data(tmp_path, config_data)
    config = load_config(
        path,
        env={"MODEL_LEVEL_2": "openai/model-x", "JEV_THRESHOLD_L1": "0.65"},
    )
    assert config.levels[1].provider == "openai"
    assert config.levels[1].model == "model-x"
    assert config.levels[1].full_name == "openai/model-x"
    assert config.levels[0].jev_threshold == 0.65
    # Untouched values keep coming from YAML.
    assert config.levels[1].jev_threshold == 0.70


def test_environment_threshold_override_must_be_numeric(tmp_path: Path, config_data: dict[str, Any]):
    path = write_config_data(tmp_path, config_data)
    with pytest.raises(CascadeConfigError, match="JEV_THRESHOLD_L1"):
        load_config(path, env={"JEV_THRESHOLD_L1": "high"})


def test_environment_cannot_set_a_threshold_on_the_final_level(tmp_path: Path,
                                                                 config_data: dict[str, Any]):
    path = write_config_data(tmp_path, config_data)
    with pytest.raises(CascadeConfigError, match="final level 3 must not define jev_threshold"):
        load_config(path, env={"JEV_THRESHOLD_L3": "0.5"})


def test_warns_when_a_level_reuses_the_previous_model(tmp_path: Path,
                                                       config_data: dict[str, Any]):
    config_data["cascade"]["levels"][1]["provider"] = "openai"
    config_data["cascade"]["levels"][1]["model"] = "model-one"
    config = load_config(write_config_data(tmp_path, config_data))
    warnings = configuration_warnings(config)
    assert any("level 2 reuses the model of level 1" in warning for warning in warnings)


def test_warns_when_level_2_threshold_exceeds_level_1(tmp_path: Path,
                                                       config_data: dict[str, Any]):
    config_data["cascade"]["levels"][0]["jev_threshold"] = 0.60
    config_data["cascade"]["levels"][1]["jev_threshold"] = 0.75
    config = load_config(write_config_data(tmp_path, config_data))
    warnings = configuration_warnings(config)
    assert any("higher than level 1" in warning for warning in warnings)


def test_no_warnings_for_the_reference_configuration(cascade_config):
    warnings = configuration_warnings(cascade_config)
    assert not [w for w in warnings if "final level" in w]
    # 'escalate' is the default policy now, so the reference config warns about nothing
    # policy-related; only fail_closed is worth a warning (see test_jev_failure_policy).
    assert not [w for w in warnings if "failure_policy" in w]


def test_repr_contains_no_secrets(cascade_config: Any):
    settings = Settings(
        api_key="client-secret-value",
        jev_api_key="jev-secret-value",
        openai_api_key="openai-secret-value",
        anthropic_api_key="anthropic-secret-value",
    )
    for rendered in (repr(settings), str(settings), repr(cascade_config), str(cascade_config)):
        assert "secret-value" not in rendered
    assert "input_per_mtok_usd" in repr(cascade_config)


def test_base_config_fixture_is_not_shared_between_tests(config_data: dict[str, Any]):
    config_data["cascade"]["levels"][0]["model"] = "mutated"
    assert base_config_dict()["cascade"]["levels"][0]["model"] == "model-one"
