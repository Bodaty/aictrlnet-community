"""Unit tests for the Trello adapter (mocked httpx — no network)."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from adapters.models import (
    AdapterConfig,
    AdapterCategory,
    AdapterRequest,
    AdapterStatus,
)
from adapters.implementations.integration.trello_adapter import TrelloAdapter


def _make_adapter():
    config = AdapterConfig(
        name="trello",
        version="1.0.0",
        category=AdapterCategory.INTEGRATION,
        api_key="test-key",
        credentials={"api_token": "test-token"},
    )
    adapter = TrelloAdapter(config)
    adapter._initialized = True
    adapter.status = AdapterStatus.READY
    return adapter


def _client_returning(payload, *, raises=None):
    """A mock httpx client whose .request() returns `payload` as JSON (or raises)."""
    client = MagicMock()
    if raises is not None:
        client.request = AsyncMock(side_effect=raises)
    else:
        resp = MagicMock()
        resp.raise_for_status = MagicMock(return_value=None)
        resp.json = MagicMock(return_value=payload)
        client.request = AsyncMock(return_value=resp)
    return client


@pytest.fixture
def adapter():
    return _make_adapter()


def test_requires_credentials(monkeypatch):
    monkeypatch.delenv("TRELLO_API_KEY", raising=False)
    monkeypatch.delenv("TRELLO_API_TOKEN", raising=False)
    with pytest.raises(ValueError):
        TrelloAdapter(AdapterConfig(name="trello", category=AdapterCategory.INTEGRATION))


def test_env_fallback_credentials(monkeypatch):
    monkeypatch.setenv("TRELLO_API_KEY", "env-key")
    monkeypatch.setenv("TRELLO_API_TOKEN", "env-token")
    a = TrelloAdapter(AdapterConfig(name="trello", category=AdapterCategory.INTEGRATION))
    assert a.api_key == "env-key"
    assert a.token == "env-token"
    assert a._auth == {"key": "env-key", "token": "env-token"}


def test_capabilities_cover_core_ops(adapter):
    names = {c.name for c in adapter.get_capabilities()}
    for expected in ("create_card", "move_card", "add_comment", "list_cards", "register_webhook"):
        assert expected in names


@pytest.mark.asyncio
async def test_create_card_success(adapter):
    adapter.client = _client_returning({"id": "card123", "name": "Test"})
    r = await adapter.execute(
        AdapterRequest(capability="create_card", parameters={"list_id": "L1", "name": "Test"})
    )
    assert r.status == "success"
    assert r.data["id"] == "card123"
    # Trello param mapping: friendly -> trello names, auth merged in.
    _, kwargs = adapter.client.request.call_args
    assert kwargs["params"]["idList"] == "L1"
    assert kwargs["params"]["name"] == "Test"
    assert kwargs["params"]["key"] == "test-key"


@pytest.mark.asyncio
async def test_move_card_maps_target_list(adapter):
    adapter.client = _client_returning({"id": "card123"})
    r = await adapter.execute(
        AdapterRequest(capability="move_card", parameters={"card_id": "card123", "target_list_id": "L2"})
    )
    assert r.status == "success"
    args, kwargs = adapter.client.request.call_args
    assert args[0] == "PUT"
    assert args[1] == "/cards/card123"
    assert kwargs["params"]["idList"] == "L2"


@pytest.mark.asyncio
async def test_add_comment_success(adapter):
    adapter.client = _client_returning({"id": "comment1"})
    r = await adapter.execute(
        AdapterRequest(capability="add_comment", parameters={"card_id": "c1", "text": "hi"})
    )
    assert r.status == "success"
    args, kwargs = adapter.client.request.call_args
    assert args[1] == "/cards/c1/actions/comments"
    assert kwargs["params"]["text"] == "hi"


@pytest.mark.asyncio
async def test_list_param_serialized(adapter):
    adapter.client = _client_returning({"id": "c1"})
    await adapter.execute(
        AdapterRequest(
            capability="create_card",
            parameters={"list_id": "L1", "name": "x", "label_ids": ["a", "b"]},
        )
    )
    _, kwargs = adapter.client.request.call_args
    assert kwargs["params"]["idLabels"] == "a,b"  # list -> comma-joined


@pytest.mark.asyncio
async def test_list_cards_wraps_bare_list(adapter):
    # Trello returns a bare array; the adapter must wrap it so the workflow
    # adapter node (which sets keys on the result) doesn't crash on a list.
    adapter.client = _client_returning([{"id": "c1"}, {"id": "c2"}])
    r = await adapter.execute(
        AdapterRequest(capability="list_cards", parameters={"list_id": "L1"})
    )
    assert r.status == "success"
    assert r.data["count"] == 2
    assert r.data["items"][0]["id"] == "c1"


@pytest.mark.asyncio
async def test_api_error_returns_error_response(adapter):
    adapter.client = _client_returning(None, raises=RuntimeError("boom"))
    r = await adapter.execute(
        AdapterRequest(capability="get_card", parameters={"card_id": "c1"})
    )
    assert r.status == "error"
    assert "boom" in r.error


@pytest.mark.asyncio
async def test_unknown_capability_raises(adapter):
    with pytest.raises(ValueError):
        await adapter.execute(AdapterRequest(capability="nope", parameters={}))
