"""Phase 0: the engine reads every model, threshold and price from CascadeConfig."""

from __future__ import annotations

import pytest

from src.cascade_engine import CascadeEngine
from src.jev_evaluator import EvaluationResult
from src.settings import Settings


class FakeEvaluator:
    def __init__(self, results):
        self.results = iter(results)
        self.prompts = []
        self.thresholds = []

    async def evaluate(self, prompt, response, *, threshold=None, system=""):
        self.prompts.append(prompt)
        self.thresholds.append(threshold)
        return next(self.results)


class FakeProvider:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    async def complete(self, model, messages, timeout_ms=None, **kwargs):
        self.calls.append((model, messages, kwargs, timeout_ms))
        return next(self.responses)


def response(text, prompt_tokens=10, completion_tokens=5):
    return {
        "id": "provider-response",
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop", "index": 0}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens},
    }


def cascade(provider, evaluator, config):
    return CascadeEngine(Settings(), config, provider=provider, evaluator=evaluator)


@pytest.mark.asyncio
async def test_cascade_stops_at_level_1_if_jev_passes(cascade_config):
    provider = FakeProvider([response("good answer")])
    evaluator = FakeEvaluator([EvaluationResult(approved=True, score=0.95)])

    result = await cascade(provider, evaluator, cascade_config).run(
        [{"role": "user", "content": "Question"}], requested_model="auto"
    )

    assert result.response["choices"][0]["message"]["content"] == "good answer"
    assert result.model_used == "openai/model-one"
    assert result.hops_count == 1
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_cascade_escalates_to_level_2_if_level_1_fails(cascade_config):
    provider = FakeProvider([response("bad answer"), response("better answer")])
    evaluator = FakeEvaluator([
        EvaluationResult(approved=False, score=0.2),
        EvaluationResult(approved=True, score=0.95),
    ])

    result = await cascade(provider, evaluator, cascade_config).run(
        [{"role": "user", "content": "Question"}], requested_model="auto"
    )

    assert result.response["choices"][0]["message"]["content"] == "better answer"
    assert result.model_used == "anthropic/model-two"
    assert result.hops_count == 2
    assert len(provider.calls) == 2


@pytest.mark.asyncio
async def test_anti_loop_hard_cap(cascade_config):
    provider = FakeProvider([response("first"), response("second"), response("frontier")])
    evaluator = FakeEvaluator([
        EvaluationResult(approved=False, score=0.2),
        EvaluationResult(approved=False, score=0.3),
    ])

    result = await cascade(provider, evaluator, cascade_config).run(
        [{"role": "user", "content": "Question"}], requested_model="auto"
    )

    assert result.response["choices"][0]["message"]["content"] == "frontier"
    assert result.model_used == "anthropic/model-three"
    assert result.hops_count == 3
    assert len(provider.calls) == 3
    assert len(evaluator.prompts) == 2


@pytest.mark.asyncio
async def test_cascade_rejects_only_the_requested_model(cascade_config):
    provider = FakeProvider([response("single tier")])
    evaluator = FakeEvaluator([])

    result = await cascade(provider, evaluator, cascade_config).run(
        [{"role": "user", "content": "Question"}], requested_model="anthropic/model-two"
    )

    assert result.model_used == "anthropic/model-two"
    assert result.hops_count == 1
    assert len(provider.calls) == 1
    assert evaluator.prompts == []


@pytest.mark.asyncio
async def test_unknown_model_is_rejected(cascade_config):
    engine = cascade(FakeProvider([]), FakeEvaluator([]), cascade_config)

    with pytest.raises(ValueError, match="Unsupported model"):
        await engine.run([{"role": "user", "content": "Question"}], requested_model="openai/other")


@pytest.mark.asyncio
async def test_cascade_context_preservation(cascade_config):
    provider = FakeProvider([response("wrong"), response("right")])
    evaluator = FakeEvaluator([
        EvaluationResult(approved=False, score=0.1),
        EvaluationResult(approved=True, score=0.9),
    ])
    original_messages = [
        {"role": "system", "content": "Be concise"},
        {"role": "user", "content": "What is 2+2?"},
    ]

    await cascade(provider, evaluator, cascade_config).run(original_messages, requested_model="auto")

    second_messages = provider.calls[1][1]
    assert second_messages[:2] == original_messages
    assert second_messages[-2]["role"] == "assistant"
    assert second_messages[-2]["content"] == "wrong"
    assert second_messages[-1]["role"] == "system"
    assert "score 0.1" in second_messages[-1]["content"]


@pytest.mark.asyncio
async def test_engine_uses_the_threshold_of_the_level_being_evaluated(cascade_config):
    """JEV is mocked to 0.75: rejected at level 1 (0.80), approved at level 2 (0.70)."""
    provider = FakeProvider([response("level one"), response("level two")])
    evaluator = FakeEvaluator([
        EvaluationResult(approved=False, score=0.75),
        EvaluationResult(approved=True, score=0.75),
    ])

    result = await cascade(provider, evaluator, cascade_config).run(
        [{"role": "user", "content": "Question"}], requested_model="auto"
    )

    assert result.model_used == "anthropic/model-two"
    assert result.hops_count == 2
    assert len(evaluator.prompts) == 2


@pytest.mark.asyncio
async def test_engine_applies_level_params_from_config(cascade_config):
    provider = FakeProvider([response("ok")])
    evaluator = FakeEvaluator([EvaluationResult(approved=True, score=0.99)])

    await cascade(provider, evaluator, cascade_config).run(
        [{"role": "user", "content": "Question"}],
        requested_model="auto",
        temperature=1.0,
        max_tokens=999,
    )

    model, _, kwargs, timeout_ms = provider.calls[0]
    assert model == "openai/model-one"
    assert kwargs == {"temperature": 0.2, "max_tokens": 1024}
    assert timeout_ms == 30000


@pytest.mark.asyncio
async def test_engine_forwards_params_it_does_not_configure(cascade_config):
    provider = FakeProvider([response("ok")])
    evaluator = FakeEvaluator([EvaluationResult(approved=True, score=0.99)])

    await cascade(provider, evaluator, cascade_config).run(
        [{"role": "user", "content": "Question"}],
        requested_model="auto",
        top_p=0.9,
        stop=["\n"],
    )

    _, _, kwargs, _ = provider.calls[0]
    assert kwargs == {"temperature": 0.2, "max_tokens": 1024, "top_p": 0.9, "stop": ["\n"]}


@pytest.mark.asyncio
async def test_telemetry_cost_comes_from_config_pricing(cascade_config, caplog):
    import json
    import logging

    provider = FakeProvider([response("ok", prompt_tokens=1_000_000, completion_tokens=1_000_000)])
    evaluator = FakeEvaluator([EvaluationResult(approved=True, score=0.99)])
    with caplog.at_level(logging.INFO, logger="ai_cascade_proxy.telemetry"):
        await cascade(provider, evaluator, cascade_config).run(
            [{"role": "user", "content": "Question"}], requested_model="auto"
        )

    telemetry = next(
        json.loads(record.getMessage())
        for record in caplog.records
        if '"model_used"' in record.getMessage()
    )
    # Level 1 fixture prices: 1.0 input + 2.0 output USD per Mtok, full Mtok of each.
    assert telemetry["cost_usd"] == pytest.approx(3.0)
    assert telemetry["baseline_cost_usd"] == pytest.approx(21.0)
    assert telemetry["net_savings_usd"] == pytest.approx(18.0)
