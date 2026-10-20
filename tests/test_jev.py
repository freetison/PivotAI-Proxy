import httpx
import pytest

from src.jev_evaluator import JevEvaluator
from src.settings import Settings


THRESHOLD = 0.8


def evaluator(client=None, settings=None):
    return JevEvaluator(
        settings or Settings(jev_api_key="test-jev-key", jev_api_url="https://jev.test/evaluate"),
        client=client,
    )


@pytest.mark.asyncio
async def test_jev_approves_valid_response():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer test-jev-key"
        body = __import__("json").loads(request.content)
        assert body["prompt"] == "What is 2 + 2?"
        assert body["response"] == "4"
        return httpx.Response(200, json={"score": 0.95})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    result = await evaluator(client).evaluate("What is 2 + 2?", "4", threshold=THRESHOLD)
    await client.aclose()

    assert result.approved is True
    assert result.score == 0.95


@pytest.mark.asyncio
async def test_jev_rejects_invalid_response():
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"score": 0.3}))
    )
    result = await evaluator(client).evaluate("Explain photosynthesis", "Plants eat sunlight", threshold=THRESHOLD)
    await client.aclose()

    assert result.approved is False
    assert result.score == 0.3


@pytest.mark.asyncio
async def test_jev_timeout_is_unavailable_not_approval():
    # Without a cascade config the evaluator falls back to the default policy,
    # 'escalate': no score means "not approved", and the answer is never fabricated.
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("JEV timed out", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    result = await evaluator(client).evaluate("Question", "Answer", threshold=THRESHOLD)
    await client.aclose()

    assert result.approved is False
    assert result.score is None
    assert result.fallback is True
    assert result.status == "unavailable"


@pytest.mark.asyncio
async def test_jev_server_error_is_unavailable():
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(503))
    )
    result = await evaluator(client).evaluate("Question", "Answer", threshold=THRESHOLD)
    await client.aclose()

    assert result.approved is False
    assert result.fallback is True
    assert result.status == "unavailable"


@pytest.mark.asyncio
async def test_jev_rejects_invalid_scores_as_unavailable():
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"score": "invalid"}))
    )
    result = await evaluator(client).evaluate("Question", "Answer", threshold=THRESHOLD)
    await client.aclose()

    assert result.approved is False
    assert result.fallback is True
    assert result.score is None
    assert result.status == "unavailable"


@pytest.mark.asyncio
async def test_jev_unconfigured_is_unavailable():
    result = await JevEvaluator(Settings()).evaluate("Question", "Answer", threshold=THRESHOLD)

    assert result.approved is False
    assert result.fallback is True
    assert result.score is None
    assert result.status == "unavailable"
