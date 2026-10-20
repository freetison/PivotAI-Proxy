"""Phase 9: the escalation-context experiment.

Production sends the rejected candidate back to the next level together with a
short explanation of *why* it was rejected. This experiment asks whether that
actually helps, by re-running the same escalations twice: once with the failure
context ("failure_context") and once with the original prompt only ("clean").

The experiment owns no production behaviour. The only thing it shares with the
proxy is the text of the failure note itself — imported, never copied — so the
two cannot drift apart.
"""

from __future__ import annotations

import asyncio
import inspect
import json

import pytest

from eval.escalation_context import (
    CONTEXT_MODES,
    ExperimentError,
    ModeOutcome,
    build_escalation_messages,
    render_report,
    run_experiment,
)
from src.cascade_engine import failure_context
from src.jev_evaluator import EvaluationResult

from tests.conftest import base_config_dict, load_config, write_config_data

PROMPT = "What is the capital of Australia?"
CANDIDATE = "The capital of Australia is Sydney."


def completion(text: str, *, prompt_tokens: int = 10, completion_tokens: int = 5) -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "anthropic/model-two",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


class Recorder:
    """Stands in for the provider: records every request, replies from a script."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests: list[dict] = []

    async def __call__(self, *, level, messages, **kwargs):
        self.requests.append({"level": level, "messages": messages, **kwargs})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


# ------------------------------------------------------------ prompt shaping

def test_clean_context_sends_only_the_original_prompt():
    messages = build_escalation_messages(PROMPT, CANDIDATE, "clean")

    assert messages == [{"role": "user", "content": PROMPT}]


def test_failure_context_sends_the_candidate_and_the_reason():
    messages = build_escalation_messages(
        PROMPT, CANDIDATE, "failure_context",
        evaluation=EvaluationResult(approved=False, score=0.41),
    )

    assert [message["role"] for message in messages] == ["user", "assistant", "system"]
    assert messages[0]["content"] == PROMPT
    assert messages[1]["content"] == CANDIDATE
    assert messages[2]["content"] == failure_context(EvaluationResult(approved=False, score=0.41))


def test_the_failure_note_is_the_production_one_not_a_copy():
    """The experiment must measure the real prompt, so it imports the engine's text."""
    messages = build_escalation_messages(
        PROMPT, CANDIDATE, "failure_context",
        evaluation=EvaluationResult(approved=False, score=None, status="unavailable"),
    )

    assert "the evaluator could not score it" in messages[-1]["content"]
    assert "0.41" not in messages[-1]["content"]


def test_failure_context_without_an_evaluation_is_refused():
    """Guessing a reason would make the two arms incomparable."""

    with pytest.raises(ValueError, match="requires the evaluation"):
        build_escalation_messages(PROMPT, CANDIDATE, "failure_context")


def test_only_the_two_documented_modes_exist():
    assert CONTEXT_MODES == ("clean", "failure_context")

    with pytest.raises(ValueError, match="unknown context mode"):
        build_escalation_messages(PROMPT, CANDIDATE, "verbose")


# ---------------------------------------------------------------- the run

@pytest.fixture
def config(tmp_path):
    return load_config(write_config_data(tmp_path, base_config_dict()))


@pytest.fixture
def dataset_row():
    from eval.calibrate import DatasetRow

    return DatasetRow(
        id="row-1",
        prompt=PROMPT,
        candidate_response=CANDIDATE,
        human_label="reject",
        source="unit-test",
        notes="",
    )


async def run(config, rows, *, scores, replies, threshold=0.8):
    """Run the experiment with a scripted candidate score and scripted replies."""
    # One candidate score per row, in row order. The first time a row's prompt is
    # scored it is the candidate; every later scoring of that prompt is the answer
    # the second level produced (which may repeat the candidate word for word, and
    # must still be judged as an answer).
    candidates = {row.id: score for row, score in zip(rows, scores)}
    pending = set(candidates)

    async def scorer(prompt, response):
        for row in rows:
            if row.id in pending and prompt == row.prompt:
                pending.discard(row.id)
                return candidates[row.id]
        return 0.95

    recorder = Recorder(replies)
    analysis = await run_experiment(
        rows, threshold=threshold, completer=recorder, scorer=scorer, config=config,
    )
    return analysis, recorder


@pytest.mark.asyncio
async def test_both_arms_ask_the_second_level_once(config, dataset_row):
    analysis, recorder = await run(
        config, [dataset_row],
        scores=[0.2],  # the candidate was rejected -> this row escalates
        replies=[completion("Canberra."), completion("Canberra is the capital.")],
    )

    assert analysis.escalated_rows == 1
    assert len(recorder.requests) == 2
    assert [request["level"].level for request in recorder.requests] == [2, 2]
    assert analysis.outcomes["clean"].attempted == 1
    assert analysis.outcomes["failure_context"].attempted == 1


@pytest.mark.asyncio
async def test_the_two_arms_differ_only_in_the_context(config, dataset_row):
    _, recorder = await run(
        config, [dataset_row], scores=[0.2],
        replies=[completion("a"), completion("b")],
    )

    clean, with_context = recorder.requests
    # clean: prompt only. failure_context: prompt + candidate + note.
    assert len(clean["messages"]) == 1
    assert len(with_context["messages"]) == 3
    assert clean["messages"][0] == with_context["messages"][0]
    assert with_context["messages"][1]["content"] == CANDIDATE


@pytest.mark.asyncio
async def test_an_approved_candidate_does_not_escalate_and_costs_nothing(config, dataset_row):
    analysis, recorder = await run(config, [dataset_row], scores=[0.95], replies=[])

    assert analysis.escalated_rows == 0
    assert recorder.requests == []
    assert analysis.outcomes["clean"].attempted == 0


@pytest.mark.asyncio
async def test_an_unscorable_candidate_escalates_like_the_escalate_policy(config, dataset_row):
    """``escalate`` is the default failure policy, so an unavailable score escalates."""

    analysis, recorder = await run(
        config, [dataset_row], scores=[None],
        replies=[completion("a"), completion("b")],
    )

    assert analysis.escalated_rows == 1
    assert len(recorder.requests) == 2


@pytest.mark.asyncio
async def test_a_failing_arm_is_recorded_and_never_counted_as_approved(config, dataset_row):
    analysis, _ = await run(
        config, [dataset_row], scores=[0.2],
        replies=[RuntimeError("upstream exploded"), completion("b")],
    )

    clean = analysis.outcomes["clean"]
    assert clean.attempted == 1
    assert clean.answered == 0
    assert clean.approved == 0
    assert "upstream exploded" in clean.errors[0]
    assert analysis.outcomes["failure_context"].approved == 1


@pytest.mark.asyncio
async def test_an_answer_identical_to_the_candidate_is_flagged(config, dataset_row):
    """If the second level repeats the rejected text, the context changed nothing."""

    analysis, _ = await run(
        config, [dataset_row], scores=[0.2],
        replies=[completion(CANDIDATE), completion("Canberra is the capital.")],
    )

    assert analysis.outcomes["clean"].distinct == 0
    assert analysis.outcomes["failure_context"].distinct == 1


@pytest.mark.asyncio
async def test_the_cost_of_each_arm_is_priced_from_the_second_level(config, dataset_row):
    analysis, _ = await run(
        config, [dataset_row], scores=[0.2],
        replies=[completion("a"), completion("b")],
    )

    level_two = config.level_for_model("anthropic/model-two")
    expected = (10 * level_two.pricing.input_per_mtok_usd
                + 5 * level_two.pricing.output_per_mtok_usd) / 1_000_000
    assert analysis.outcomes["clean"].cost_usd == pytest.approx(expected)
    assert analysis.total_cost_usd == pytest.approx(2 * expected)


@pytest.mark.asyncio
async def test_an_empty_escalation_set_produces_no_numbers(config, dataset_row):
    analysis, _ = await run(config, [dataset_row], scores=[0.99], replies=[])

    report = render_report(analysis)

    assert "no row would escalate" in report
    assert "%" not in report
    # Undefined rates must be None, never 0.0.
    assert analysis.outcomes["clean"].approval_rate is None
    assert analysis.outcomes["failure_context"].approval_rate is None


@pytest.mark.asyncio
async def test_the_report_compares_the_two_arms(config, dataset_row):
    analysis, _ = await run(
        config, [dataset_row], scores=[0.2],
        replies=[completion("Canberra."), completion("Canberra.")],
    )

    report = render_report(analysis)

    assert "clean" in report and "failure_context" in report
    assert "threshold 0.80" in report
    assert "escalated rows: 1" in report


# ---------------------------------------------------------------- the CLI

def test_the_cli_refuses_an_unreadable_dataset(tmp_path, capsys):
    from eval.escalation_context import main

    code = main(["--dataset", str(tmp_path / "missing.jsonl")])

    assert code == 2
    assert "dataset error" in capsys.readouterr().err


def test_the_cli_uses_the_level_one_threshold_by_default(tmp_path, config):
    from eval.escalation_context import default_threshold

    assert default_threshold(config) == config.threshold_for_level(1)


def test_dry_run_prints_the_requests_and_makes_no_call(tmp_path, capsys, config):
    from eval.escalation_context import main

    dataset = tmp_path / "rows.jsonl"
    dataset.write_text(json.dumps({
        "id": "row-1", "prompt": PROMPT, "candidate_response": CANDIDATE,
        "human_label": "reject", "source": "unit-test", "notes": "",
    }) + "\n", encoding="utf-8")

    def forbidden(*args, **kwargs):
        raise AssertionError("dry run must not call the provider")

    code = main(["--dataset", str(dataset), "--dry-run"], config=config, completer=forbidden,
                scorer=forbidden)
    out = capsys.readouterr().out

    assert code == 0
    assert "clean" in out and "failure_context" in out
    assert CANDIDATE in out


def test_a_missing_cascade_config_is_reported_not_raised(tmp_path, capsys):
    from eval.escalation_context import main

    dataset = tmp_path / "rows.jsonl"
    dataset.write_text(json.dumps({
        "id": "row-1", "prompt": PROMPT, "candidate_response": CANDIDATE,
        "human_label": "reject", "source": "unit-test", "notes": "",
    }) + "\n", encoding="utf-8")

    code = main(["--dataset", str(dataset), "--cascade-config", str(tmp_path / "nope.yaml")],
                completer=None, scorer=None)
    out = capsys.readouterr()

    assert code == 4
    assert "cascade configuration" in out.err


def test_mode_outcome_rates_are_none_when_undefined():
    outcome = ModeOutcome(mode="clean")

    assert outcome.approval_rate is None
    assert outcome.distinct_rate is None

    outcome = ModeOutcome(mode="clean", attempted=4, answered=4, approved=1, distinct=2)
    assert outcome.approval_rate == pytest.approx(0.25)
    assert outcome.distinct_rate == pytest.approx(0.5)


# ------------------------------------------------- shapes the model can return

@pytest.mark.parametrize("payload", [
    {},
    {"choices": []},
    {"choices": "nope"},
    {"choices": [None]},
    {"choices": [{}]},
    {"choices": [{"message": None}]},
    {"choices": [{"message": {"content": 5}}]},
])
def test_an_answer_without_usable_text_is_an_error_not_an_approval(config, dataset_row, payload):
    analysis, _ = asyncio.run(_run_with_payload(config, dataset_row, payload))

    outcome = analysis.outcomes["clean"]
    assert outcome.answered == 0
    assert outcome.approved == 0
    assert outcome.approval_rate is None
    assert "no completion text" in outcome.errors[0]


async def _run_with_payload(config, row, payload):
    async def scorer(prompt, response):
        return 0.2 if response == CANDIDATE else 0.95

    class Fixed:
        async def __call__(self, *, level, messages, **kwargs):
            return payload

    analysis = await run_experiment(
        [row], threshold=0.8, completer=Fixed(), scorer=scorer, config=config,
    )
    return analysis, None


@pytest.mark.asyncio
async def test_a_content_block_list_is_read_as_text(config, dataset_row):
    async def scorer(prompt, response):
        return 0.2 if response == CANDIDATE else 0.95

    payload = completion([{"type": "text", "text": "Canberra."}])

    class Single:
        def __init__(self):
            self.replies = [payload, payload]

        async def __call__(self, *, level, messages, **kwargs):
            return self.replies.pop(0)

    analysis = await run_experiment(
        [dataset_row], threshold=0.8, completer=Single(), scorer=scorer, config=config,
    )

    assert analysis.outcomes["clean"].answered == 1
    assert analysis.outcomes["clean"].approved == 1


# ------------------------------------------------------------- CLI plumbing

def test_limit_cuts_the_dataset_short(tmp_path, config):
    from eval.escalation_context import main

    dataset = tmp_path / "rows.jsonl"
    dataset.write_text("\n".join(json.dumps({
        "id": f"row-{index}", "prompt": f"{PROMPT} {index}", "candidate_response": CANDIDATE,
        "human_label": "reject", "source": "unit-test", "notes": "",
    }) for index in range(5)) + "\n", encoding="utf-8")

    def forbidden(*args, **kwargs):
        raise AssertionError("dry run must not call anything")

    assert main(["--dataset", str(dataset), "--dry-run", "--limit", "2"],
                config=config, completer=forbidden, scorer=forbidden) == 0


def test_an_empty_dataset_is_refused_by_the_loader(tmp_path, capsys, config):
    """An empty dataset cannot answer anything, so it is a dataset error, not a report."""
    from eval.escalation_context import main

    dataset = tmp_path / "rows.jsonl"
    dataset.write_text("", encoding="utf-8")

    def forbidden(*args, **kwargs):
        raise AssertionError("an empty dataset must not call anything")

    assert main(["--dataset", str(dataset), "--dry-run"],
                config=config, completer=forbidden, scorer=forbidden) == 2
    assert "dataset is empty" in capsys.readouterr().err


def test_the_dry_run_says_so_when_there_is_nothing_to_send(config):
    from eval.escalation_context import _render_dry_run

    assert "nothing to send" in _render_dry_run([], config, 2)


def test_errors_are_listed_in_the_report(config, dataset_row):
    analysis, _ = asyncio.run(run(
        config, [dataset_row], scores=[0.2],
        replies=[RuntimeError("boom"), completion("fine")],
    ))

    assert "clean: row-1: RuntimeError: boom" in render_report(analysis)


def test_the_default_threshold_is_the_level_one_threshold(config):
    from eval.escalation_context import default_threshold

    assert default_threshold(config) == config.threshold_for_level(1)


def test_the_cli_runs_the_full_experiment_when_handed_a_config(tmp_path, capsys, config):
    """The only path that spends money: driven here with a scripted completer."""
    from eval.escalation_context import main

    dataset = tmp_path / "rows.jsonl"
    dataset.write_text(json.dumps({
        "id": "row-1", "prompt": PROMPT, "candidate_response": CANDIDATE,
        "human_label": "reject", "source": "unit-test", "notes": "",
    }) + "\n", encoding="utf-8")

    async def completer(*, level, messages):
        return completion("Canberra is the capital.")

    async def scorer(prompt, response):
        return 0.2 if response == CANDIDATE else 0.95

    code = main(["--dataset", str(dataset)], config=config,
                completer=completer, scorer=scorer)
    out = capsys.readouterr().out

    assert code == 0
    assert "escalated rows: 1" in out
    assert "clean" in out and "failure_context" in out


def test_the_cascade_config_path_option_is_used(tmp_path, config, monkeypatch):
    from eval import escalation_context

    dataset = tmp_path / "rows.jsonl"
    dataset.write_text(json.dumps({
        "id": "row-1", "prompt": PROMPT, "candidate_response": CANDIDATE,
        "human_label": "reject", "source": "unit-test", "notes": "",
    }) + "\n", encoding="utf-8")
    seen = {}

    def fake_load(path):
        seen["path"] = path
        return config

    monkeypatch.setattr("src.cascade_config.load_cascade_config", fake_load)

    assert escalation_context.main(["--dataset", str(dataset), "--dry-run",
                                   "--cascade-config", "custom.yaml"]) == 0
    assert seen["path"] == "custom.yaml"


def test_the_real_wiring_builds_callables_without_calling_them(monkeypatch):
    """The production adapters must not touch the network until a row is scored."""
    from eval import escalation_context
    from src.settings import Settings

    monkeypatch.setattr(escalation_context, "Settings", Settings, raising=False)

    completer = escalation_context._real_completer()
    scorer = escalation_context._real_scorer()

    assert callable(completer) and callable(scorer)
    assert inspect.iscoroutinefunction(completer)
    assert inspect.iscoroutinefunction(scorer)


def test_the_default_config_path_comes_from_the_environment(tmp_path, monkeypatch):
    from eval import escalation_context
    from tests.conftest import base_config_dict, write_config_data

    path = write_config_data(tmp_path, base_config_dict())
    monkeypatch.setenv("CASCADE_CONFIG_PATH", str(path))
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")

    config = escalation_context._load_config(None)

    assert [level.model for level in config.levels] == ["model-one", "model-two", "model-three"]
