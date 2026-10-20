"""Phase 0: optional startup check that every configured model id exists at its provider."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from src.cascade_config import (
    CascadeConfigError,
    load_cascade_config,
    validate_models_against_providers,
)
from src.settings import Settings
from tests.conftest import TEST_API_KEYS, base_config_dict, write_config_data


def openai_handler(request: httpx.Request) -> httpx.Response:
    assert request.url.path.endswith("/models")
    return httpx.Response(200, json={"data": [{"id": "model-one"}, {"id": "model-two"}]})


def make_provider(handler, tmp_path: Path):
    from src.provider import ModelProvider

    settings = Settings(
        provider_base_urls={"openai": "https://provider.test/v1",
                            "anthropic": "https://anthropic.test/v1"},
        openai_api_key=TEST_API_KEYS["openai"],
        anthropic_api_key=TEST_API_KEYS["anthropic"],
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return ModelProvider(settings, client=client), client


@pytest.mark.asyncio
async def test_missing_model_is_reported(tmp_path: Path, cascade_config):
    def handler(request: httpx.Request) -> httpx.Response:
        if "anthropic" in request.url.host:
            return httpx.Response(200, json={"data": [{"id": "model-two"}]})
        return httpx.Response(200, json={"data": [{"id": "something-else"}]})

    provider, client = make_provider(handler, tmp_path)
    try:
        with pytest.raises(CascadeConfigError, match="openai/model-one"):
            await validate_models_against_providers(cascade_config, provider)
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_all_models_present_passes(tmp_path: Path, cascade_config):
    def handler(request: httpx.Request) -> httpx.Response:
        if "anthropic" in request.url.host:
            return httpx.Response(200, json={"data": [{"id": "model-two"}, {"id": "model-three"}]})
        return httpx.Response(200, json={"data": [{"id": "model-one"}]})

    provider, client = make_provider(handler, tmp_path)
    try:
        await validate_models_against_providers(cascade_config, provider)
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_provider_failure_is_reported_as_config_error(cascade_config, tmp_path: Path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    provider, client = make_provider(handler, tmp_path)
    try:
        with pytest.raises(CascadeConfigError, match="Could not list models"):
            await validate_models_against_providers(cascade_config, provider)
    finally:
        await client.aclose()


def test_flag_is_off_by_default():
    assert Settings().validate_models_on_startup is False
