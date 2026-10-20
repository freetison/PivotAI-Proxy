"""Phase 6: the harness that says whether the cascade pays for itself.

The arithmetic is checked against numbers computed by hand, not against the
implementation's own output. Costs are always supplied by the caller.
"""

from __future__ import annotations

import pytest

from eval.breakeven import (
    BreakevenAnalysis,
    costs_from_config,
    evaluate,
    main,
    render_report,
)

from tests.conftest import base_config_dict, load_config, write_config_data

# Fixture prices per Mtok: L1 1.0/2.0, L2 3.0/6.0, L3 9.0/12.0.
# A 1000-prompt / 500-completion call therefore costs 0.002 / 0.006 / 0.015.


def configured(tmp_path):
    return load_config(write_config_data(tmp_path, base_config_dict()))


# ------------------------------------------------------------------- the maths

def test_expected_cost_at_a_given_acceptance_is_hand_computable():
    # p=0.5, q=0.8: 0.002 + 0.5*0.006 + 0.5*0.2*0.015 = 0.002+0.003+0.0015 = 0.0065
    analysis = evaluate([0.002, 0.006, 0.015], secondary_acceptance=0.8)

    assert analysis.expected_cost(0.5) == pytest.approx(0.0065)


def test_full_acceptance_pays_only_for_the_first_level():
    analysis = evaluate([0.002, 0.006, 0.015])

    assert analysis.expected_cost(1.0) == pytest.approx(0.002)
    assert analysis.cost_at_full_acceptance == pytest.approx(0.002)


def test_zero_acceptance_pays_every_level():
    # 0.002 + 0.006 + 0.2*0.015 ... no: with q=0.8, L3 is always reached when L1 rejects.
    # 0.002 + 0.006 + (1-0.8)*... -> careful: L2 is reached, then L3 is always paid.
    # cost(0) = c1 + c2 + (1-q)*c3 ... the formula below is the implementation's contract:
    # c1 + 1.0*c2 + 1.0*c3*(1-q) is WRONG; L3 is paid whenever L2 rejects, and with
    # p rejected L2 is always called and rejected with probability 1-q.
    analysis = evaluate([0.002, 0.006, 0.015], secondary_acceptance=0.8)

    assert analysis.expected_cost(0.0) == pytest.approx(0.002 + 0.006 + 0.2 * 0.015)


def test_a_cascade_that_is_cheaper_even_without_approvals_has_no_breakeven():
    analysis = evaluate([0.002, 0.006, 0.015])

    assert analysis.always_cheaper is True
    assert analysis.breakeven_acceptance is None
    assert "never has an answer approved" in render_report(analysis)


def test_a_cascade_that_never_pays_off_says_so():
    # Level 1 costs MORE than the final level: even approving everything costs more.
    analysis = evaluate([0.016, 0.02, 0.015])

    assert analysis.never_cheaper is True
    assert analysis.breakeven_acceptance is None
    assert "never pays for itself" in render_report(analysis)


def test_a_breakeven_acceptance_is_found_when_one_exists():
    # c1 = 0.004, c2 = 0.006, c3 = 0.015, q = 0.5
    # cost(p) = 0.004 + (1-p)*0.006 + (1-p)*0.5*0.015
    #         = 0.004 + (1-p)*0.0135 = 0.015  ->  (1-p) = 0.011/0.0135 = 0.8148...
    # p = 0.185185...
    analysis = evaluate([0.004, 0.006, 0.015], secondary_acceptance=0.5)

    assert analysis.breakeven_acceptance == pytest.approx(0.185185, abs=1e-5)
    assert analysis.saving_at(0.5) > 0
    assert analysis.saving_at(0.05) < 0
    assert "18.5%" in render_report(analysis)


def test_the_saving_is_the_baseline_minus_the_expected_cost():
    analysis = evaluate([0.002, 0.006, 0.015])

    assert analysis.saving_at(0.5) == pytest.approx(
        analysis.baseline_cost - analysis.expected_cost(0.5)
    )


def test_two_levels_are_enough():
    analysis = evaluate([0.002, 0.015])

    assert analysis.expected_cost(0.5) == pytest.approx(0.002 + 0.5 * 0.015)


def test_one_level_is_refused():
    with pytest.raises(ValueError, match="at least two"):
        evaluate([0.002])


def test_a_negative_cost_is_refused():
    with pytest.raises(ValueError, match="cannot be negative"):
        evaluate([0.002, -1.0, 0.015])


def test_an_impossible_acceptance_assumption_is_refused():
    with pytest.raises(ValueError, match="between 0 and 1"):
        evaluate([0.002, 0.006, 0.015], secondary_acceptance=1.5)


# --------------------------------------------------------------- from config

def test_stage_costs_come_from_the_configured_prices(tmp_path):
    config = configured(tmp_path)

    costs = costs_from_config(config, prompt_tokens=1000, completion_tokens=500)

    assert costs == pytest.approx((0.002, 0.006, 0.015))


def test_a_negative_token_count_is_refused(tmp_path):
    config = configured(tmp_path)

    with pytest.raises(ValueError, match="cannot be negative"):
        costs_from_config(config, prompt_tokens=-1, completion_tokens=0)


def test_no_prices_are_invented_for_a_zero_sized_call(tmp_path):
    config = configured(tmp_path)

    assert costs_from_config(config, prompt_tokens=0, completion_tokens=0) == (0.0, 0.0, 0.0)


# ------------------------------------------------------------------------ CLI

def test_the_cli_prints_a_report(capsys):
    exit_code = main(["0.002", "0.006", "0.015"])

    assert exit_code == 0
    assert "Break-even analysis" in capsys.readouterr().out


def test_the_cli_can_read_costs_from_a_config(tmp_path, capsys, monkeypatch):
    path = write_config_data(tmp_path, base_config_dict())
    # The cascade loader validates that the configured providers have credentials,
    # even for a read-only cost analysis.
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")

    exit_code = main(["--from-config", str(path)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "L1=$0.002000" in output


def test_the_cli_refuses_an_unusable_invocation(capsys):
    assert main(["0.002"]) == 2
    assert "at least two stage costs" in capsys.readouterr().err


def test_the_cli_reports_a_missing_config_file(tmp_path, capsys):
    assert main(["--from-config", str(tmp_path / "absent.yaml")]) == 2
    assert "breakeven failed" in capsys.readouterr().err


def test_the_cli_reports_bad_costs_instead_of_guessing(capsys):
    assert main(["0.002", "-1", "0.015"]) == 2
    assert "cannot be negative" in capsys.readouterr().err


def test_an_analysis_exposes_the_assumption_it_used():
    analysis = evaluate([0.002, 0.006, 0.015], secondary_acceptance=0.6)

    assert isinstance(analysis, BreakevenAnalysis)
    assert analysis.secondary_acceptance == 0.6
    assert "60%" in render_report(analysis)
