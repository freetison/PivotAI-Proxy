"""Phase 7: shadow mode — run the cascade, but always answer with the final level.

The point of shadow mode is to collect comparison data without changing what the
client gets. Two consequences drive every test here:

* the answer the client receives is always the final level's, never the cascade's;
* the cascade still ran, so the extra spend is real and must be visible.

Nothing is written to disk unless ``SHADOW_STORE`` is enabled, because those
records contain prompts and answers.
"""

from __future__ import annotations

import json
import logging

import pytest

from src.cascade_engine import CascadeEngine, UpstreamCascadeFailed
from src.jev_evaluator import EvaluationResult, JevUnavailable
from src.provider import ProviderError
from src.settings import Settings
from src.shadow import ShadowRecord, ShadowStore

from tests.conftest import base_config_dict, load_config, write_config_data

L1 = "openai/model-one"
L2 = "anthropic/model-two"
L3 = "anthropic/model-three"


def configured(tmp_path):
    return load_config(write_config_data(tmp_path, base_config_dict()))


class RecordingProvider:
    """Answers per model in order; an Exception item is raised instead of returned."""

    def __init__(self, responses: dict[str, list]):
        self.responses = {model: list(items) for model, items in responses.items()}
        self.calls: list[str] = []

    async def complete(self, model, messages, timeout_ms=None, **kwargs):
        self.calls.append(model)
        queued = self.responses.get(model) or []
        item = queued.pop(0) if len(queued) > 1 else queued[0]
        if isinstance(item, Exception):
            raise item
        return item


class ApprovingEvaluator:
    def __init__(self, scores=(0.95,)):
        self.scores = list(scores)
        self.calls = 0

    async def evaluate(self, prompt, response, *, threshold=None, system=""):
        self.calls += 1
        score = self.scores.pop(0) if len(self.scores) > 1 else self.scores[0]
        return EvaluationResult(approved=score >= threshold, score=score)


def completion(text: str, *, prompt_tokens: int = 1000, completion_tokens: int = 500):
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "provider-report-model",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


def shadow_engine(config, provider, evaluator, *, settings=None, **kwargs):
    return CascadeEngine(
        settings if settings is not None else Settings(cascade_mode="shadow"),
        config,
        provider=provider,
        evaluator=evaluator,
        **kwargs,
    )


def answer_text(result) -> str:
    return result.response["choices"][0]["message"]["content"]


# ------------------------------------------------------------- configuration

def test_active_is_the_default_mode():
    assert Settings().cascade_mode == "active"


def test_shadow_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("CASCADE_MODE", "shadow")

    assert Settings.from_env().cascade_mode == "shadow"


def test_an_unknown_mode_fails_at_startup(monkeypatch):
    monkeypatch.setenv("CASCADE_MODE", "paranoid")

    with pytest.raises(ValueError, match="CASCADE_MODE must be one of: active, shadow"):
        Settings.from_env()


def test_the_mode_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("CASCADE_MODE", "  SHADOW  ")

    assert Settings.from_env().cascade_mode == "shadow"


def test_the_store_is_off_by_default():
    assert Settings().shadow_store is False


def test_the_store_is_opt_in(monkeypatch):
    monkeypatch.setenv("SHADOW_STORE", "true")

    assert Settings.from_env().shadow_store is True


def test_the_store_path_is_configurable(monkeypatch, tmp_path):
    monkeypatch.setenv("SHADOW_STORE_PATH", str(tmp_path / "pairs.jsonl"))

    assert Settings.from_env().shadow_store_path == str(tmp_path / "pairs.jsonl")


# ---------------------------------------------------------------- the answer

async def test_the_answer_is_the_final_level_even_when_the_cascade_approved(tmp_path):
    config = configured(tmp_path)
    provider = RecordingProvider({
        L1: [completion("cheap answer")],
        L3: [completion("gold answer")],
    })
    engine = shadow_engine(config, provider, ApprovingEvaluator())

    result = await engine.run([{"role": "user", "content": "q"}])

    assert answer_text(result) == "gold answer"
    assert result.model_used == L3


async def test_the_cascade_still_runs(tmp_path):
    config = configured(tmp_path)
    provider = RecordingProvider({
        L1: [completion("cheap answer")],
        L3: [completion("gold answer")],
    })
    evaluator = ApprovingEvaluator()
    engine = shadow_engine(config, provider, evaluator)

    await engine.run([{"role": "user", "content": "q"}])

    assert provider.calls == [L1, L3]
    assert evaluator.calls == 1


async def test_the_cascade_answer_is_kept_for_comparison(tmp_path):
    config = configured(tmp_path)
    provider = RecordingProvider({
        L1: [completion("cheap answer")],
        L3: [completion("gold answer")],
    })
    engine = shadow_engine(config, provider, ApprovingEvaluator())

    result = await engine.run([{"role": "user", "content": "q"}])

    # The comparison target travels with the result; the body is the L3 answer.
    assert result.cascade_response is not None
    assert result.cascade_response["choices"][0]["message"]["content"] == "cheap answer"
    assert result.cascade_model_used == L1


async def test_an_escalating_cascade_still_answers_with_a_direct_final_call(tmp_path):
    config = configured(tmp_path)
    provider = RecordingProvider({
        L1: [completion("rejected")],
        L2: [completion("also rejected")],
        # Once inside the cascade, once directly.
        L3: [completion("cascade's final answer"), completion("cascade's final answer")],
    })
    engine = shadow_engine(config, provider, ApprovingEvaluator((0.1, 0.1)))

    result = await engine.run([{"role": "user", "content": "q"}])

    assert provider.calls == [L1, L2, L3, L3]
    assert result.model_used == L3
    assert result.cascade_model_used == L3


async def test_an_explicit_model_request_is_not_overridden(tmp_path):
    """Asking for a specific model is a targeted request, not a cascade run."""
    config = configured(tmp_path)
    provider = RecordingProvider({L1: [completion("explicit")]})
    engine = shadow_engine(config, provider, ApprovingEvaluator())

    result = await engine.run([{"role": "user", "content": "q"}], requested_model=L1)

    assert provider.calls == [L1]
    assert result.model_used == L1
    assert result.cascade_response is None


# --------------------------------------------------------------------- cost

async def test_the_extra_cost_is_charged_and_visible(tmp_path, caplog):
    config = configured(tmp_path)
    provider = RecordingProvider({
        L1: [completion("cheap", prompt_tokens=1_000_000, completion_tokens=0)],
        L3: [completion("gold", prompt_tokens=1_000_000, completion_tokens=0)],
    })
    engine = shadow_engine(config, provider, ApprovingEvaluator())

    with caplog.at_level(logging.INFO, logger="ai_cascade_proxy.telemetry"):
        result = await engine.run([{"role": "user", "content": "q"}])
    payload = json.loads(caplog.records[-1].getMessage())

    # 1.0 for level 1 + 9.0 for the direct final-level call.
    assert payload["shadow_cost_usd"] == pytest.approx(9.0)
    assert payload["cost_usd"] == pytest.approx(10.0)
    assert payload["baseline_cost_usd"] == pytest.approx(9.0)
    assert payload["net_savings_usd"] == pytest.approx(-1.0)
    assert result.cost_usd == pytest.approx(10.0)


async def test_the_telemetry_says_which_mode_produced_the_request(tmp_path, caplog):
    config = configured(tmp_path)
    provider = RecordingProvider({L1: [completion("cheap")], L3: [completion("gold")]})
    engine = shadow_engine(config, provider, ApprovingEvaluator())

    with caplog.at_level(logging.INFO, logger="ai_cascade_proxy.telemetry"):
        await engine.run([{"role": "user", "content": "q"}])
    payload = json.loads(caplog.records[-1].getMessage())

    assert payload["cascade_mode"] == "shadow"
    assert payload["shadow_model"] == L3
    assert payload["outbound_calls"] == 2


async def test_active_mode_reports_no_shadow_cost(tmp_path, caplog):
    config = configured(tmp_path)
    provider = RecordingProvider({L1: [completion("cheap")]})
    engine = CascadeEngine(Settings(), config, provider=provider, evaluator=ApprovingEvaluator())

    with caplog.at_level(logging.INFO, logger="ai_cascade_proxy.telemetry"):
        await engine.run([{"role": "user", "content": "q"}])
    payload = json.loads(caplog.records[-1].getMessage())

    assert payload["cascade_mode"] == "active"
    assert payload["shadow_cost_usd"] == 0.0
    assert payload["shadow_model"] is None


# ---------------------------------------------------------------- failures

async def test_a_failed_cascade_does_not_hide_the_shadow_answer(tmp_path, caplog):
    """Shadow mode never makes a request fail that a direct final call can answer."""
    config = configured(tmp_path)
    provider = RecordingProvider({L1: [completion("cheap")], L3: [completion("gold answer")]})

    class UnavailableEvaluator:
        async def evaluate(self, prompt, response, *, threshold=None, system=""):
            # fail_closed: the cascade aborts even though level 1 answered.
            raise JevUnavailable("evaluator down")

    engine = shadow_engine(config, provider, UnavailableEvaluator())

    with caplog.at_level(logging.INFO, logger="ai_cascade_proxy.telemetry"):
        result = await engine.run([{"role": "user", "content": "q"}])
    payload = json.loads(caplog.records[-1].getMessage())

    assert answer_text(result) == "gold answer"
    assert result.cascade_response is None
    assert payload["cascade_mode"] == "shadow"
    assert payload["shadow_model"] == L3


async def test_a_failed_final_level_still_fails_the_request(tmp_path):
    config = configured(tmp_path)
    provider = RecordingProvider({
        L1: [completion("cheap")],
        L3: [ProviderError("boom", upstream_status=503)],
    })
    engine = shadow_engine(config, provider, ApprovingEvaluator())

    with pytest.raises(ProviderError):
        await engine.run([{"role": "user", "content": "q"}])


async def test_a_failed_cascade_plus_a_failed_final_call_raises_the_cascade_error(tmp_path):
    config = configured(tmp_path)
    provider = RecordingProvider({
        L1: [ProviderError("boom", upstream_status=500)],
        L2: [ProviderError("boom", upstream_status=500)],
        L3: [ProviderError("boom", upstream_status=500)],
    })
    engine = shadow_engine(config, provider, ApprovingEvaluator())

    # The cascade raises UpstreamCascadeFailed; the direct call then fails too and
    # its ProviderError is what the client sees (it is the answer's real failure).
    with pytest.raises(ProviderError):
        await engine.run([{"role": "user", "content": "q"}])


async def test_an_unavailable_evaluator_does_not_break_shadow_mode(tmp_path):
    config = configured(tmp_path)
    provider = RecordingProvider({L1: [completion("cheap")], L3: [completion("gold")]})

    class UnavailableEvaluator:
        async def evaluate(self, prompt, response, *, threshold=None, system=""):
            raise JevUnavailable("evaluator down")

    engine = shadow_engine(config, provider, UnavailableEvaluator())

    result = await engine.run([{"role": "user", "content": "q"}])

    assert answer_text(result) == "gold"


# ---------------------------------------------------------------- the store

def test_an_enabled_store_appends_one_json_line(tmp_path):
    store = ShadowStore(tmp_path / "shadow.jsonl")
    record = ShadowRecord(
        request_id="req-1",
        created_at="2025-01-01T00:00:00Z",
        cascade_model=L1,
        cascade_response="cheap answer",
        shadow_model=L3,
        shadow_response="gold answer",
    )

    store.append(record)

    lines = (tmp_path / "shadow.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["cascade_model"] == L1
    assert payload["cascade_response"] == "cheap answer"
    assert payload["shadow_response"] == "gold answer"
    assert payload["identical"] is False


def test_the_store_appends_rather_than_overwrites(tmp_path):
    store = ShadowStore(tmp_path / "shadow.jsonl")
    for index in range(3):
        store.append(ShadowRecord(
            request_id=f"req-{index}", created_at="t", cascade_model=L1,
            cascade_response="a", shadow_model=L3, shadow_response="b"))

    assert len((tmp_path / "shadow.jsonl").read_text(encoding="utf-8").splitlines()) == 3


def test_identical_answers_are_marked_as_such(tmp_path):
    record = ShadowRecord(
        request_id="req-1", created_at="t", cascade_model=L1,
        cascade_response="same", shadow_model=L3, shadow_response="same")

    assert record.identical is True


def test_the_store_creates_the_directory(tmp_path):
    store = ShadowStore(tmp_path / "nested" / "deeper" / "shadow.jsonl")

    store.append(ShadowRecord(request_id="r", created_at="t", cascade_model=L1,
                              cascade_response="a", shadow_model=L3, shadow_response="b"))

    assert (tmp_path / "nested" / "deeper" / "shadow.jsonl").exists()


async def test_the_engine_writes_the_pair_when_the_store_is_enabled(tmp_path):
    config = configured(tmp_path)
    target = tmp_path / "shadow.jsonl"
    provider = RecordingProvider({L1: [completion("cheap")], L3: [completion("gold")]})
    engine = shadow_engine(
        config, provider, ApprovingEvaluator(),
        settings=Settings(cascade_mode="shadow", shadow_store=True,
                          shadow_store_path=str(target)),
    )

    await engine.run([{"role": "user", "content": "q"}])

    payload = json.loads(target.read_text(encoding="utf-8").splitlines()[0])
    assert payload["cascade_response"] == "cheap"
    assert payload["shadow_response"] == "gold"


def test_multimodal_content_is_flattened_when_stored(tmp_path):
    """The store keeps comparable text, not raw provider payloads."""
    from src.shadow import record_for

    record = record_for(
        "req-1",
        cascade_model=L1,
        cascade_response={"choices": [{"message": {"content": [
            {"type": "text", "text": "first"}, {"type": "image_url"},
            {"type": "text", "text": "second"}]}}]},
        shadow_model=L3,
        shadow_response={"choices": [{"message": {"content": "plain"}}]},
    )

    assert record.cascade_response == "first\nsecond"
    assert record.shadow_response == "plain"
    assert record.identical is False


def test_a_malformed_answer_is_stored_as_empty_rather_than_crashing(tmp_path):
    from src.shadow import record_for

    record = record_for("req-1", cascade_model=L1, cascade_response=None,
                        shadow_model=L3, shadow_response={"choices": []})

    assert record.cascade_response == ""
    assert record.shadow_response == ""


async def test_shadow_is_skipped_when_the_call_cap_is_already_spent(tmp_path, caplog):
    """A spent budget must not be widened by shadowing: it degrades, not overspends."""
    config = configured(tmp_path)
    provider = RecordingProvider({L1: [completion("cheap")]})
    engine = shadow_engine(config, provider, ApprovingEvaluator(), max_outbound_calls=1)

    with caplog.at_level(logging.WARNING, logger="ai_cascade_proxy.cascade"):
        result = await engine.run([{"role": "user", "content": "q"}])

    # The cascade's answer is returned because no budget remains for the direct call.
    assert provider.calls == [L1]
    assert answer_text(result) == "cheap"
    assert any("Shadow call skipped" in record.getMessage() for record in caplog.records)


async def test_nothing_is_written_when_the_store_is_off(tmp_path):
    config = configured(tmp_path)
    target = tmp_path / "shadow.jsonl"
    provider = RecordingProvider({L1: [completion("cheap")], L3: [completion("gold")]})
    engine = shadow_engine(
        config, provider, ApprovingEvaluator(),
        settings=Settings(cascade_mode="shadow", shadow_store=False,
                          shadow_store_path=str(target)),
    )

    await engine.run([{"role": "user", "content": "q"}])

    assert not target.exists()


# ------------------------------------------------------------- interactions

async def test_a_bypassed_request_is_not_shadowed(tmp_path):
    """A bypass already answers with one specific level; a second call adds nothing."""
    config = configured(tmp_path)
    provider = RecordingProvider({L3: [completion("only call")]})
    engine = shadow_engine(config, provider, ApprovingEvaluator())

    result = await engine.run_bypass([{"role": "user", "content": "q"}], level_number=3)

    assert provider.calls == [L3]
    assert result.cascade_response is None


async def test_the_api_returns_the_shadow_answer(tmp_path):
    from fastapi.testclient import TestClient

    from src.main import create_app

    config = configured(tmp_path)
    provider = RecordingProvider({L1: [completion("cheap")], L3: [completion("gold")]})
    app = create_app(
        settings=Settings(cascade_mode="shadow"),
        cascade_config=config,
        provider=provider,
        evaluator=ApprovingEvaluator(),
    )

    with TestClient(app) as client:
        body = client.post("/v1/chat/completions", json={
            "model": "auto", "messages": [{"role": "user", "content": "q"}]}).json()

    assert body["choices"][0]["message"]["content"] == "gold"
    assert body["model"] == L3


def test_streaming_passthrough_cannot_be_compared_and_says_so(tmp_path, caplog):
    """passthrough_l3 never runs the cascade, so shadow mode has nothing to store."""
    from src.main import create_app

    config = configured(tmp_path)
    with caplog.at_level(logging.WARNING, logger="ai_cascade_proxy.api"):
        create_app(
            settings=Settings(cascade_mode="shadow", stream_mode="passthrough_l3"),
            cascade_config=config,
            provider=RecordingProvider({}),
            evaluator=ApprovingEvaluator(),
        )

    assert any(
        "passthrough_l3" in record.getMessage() and "shadow" in record.getMessage()
        for record in caplog.records
    )
