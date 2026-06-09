"""Regression: Perplexity answer-engine adapter (Phase B1).

covers: geo-phase-b1 perplexity-adapter

The `perplexity` adapter exposes a single `answer` capability that normalises
Perplexity's Sonar response to the engine-agnostic shape
{content, citations, search_results, model} that GEO's compute-facts reads.
These tests pin the capability surface, the query extraction, the response
normalisation (with a stubbed HTTP client — no network), and env-key
resolution. No live API call.
"""
from unittest.mock import AsyncMock, MagicMock

import pytest

from adapters.models import AdapterConfig, AdapterCategory, AdapterRequest
from adapters.implementations.ai.perplexity_adapter import PerplexityAdapter


def _adapter(api_key="k"):
    return PerplexityAdapter(
        AdapterConfig(name="perplexity", category=AdapterCategory.AI, api_key=api_key)
    )


def test_capabilities_expose_answer():
    caps = _adapter().get_capabilities()
    answer = next((c for c in caps if c.name == "answer"), None)
    assert answer is not None
    assert "query" in answer.required_parameters


def test_extract_query_variants():
    a = _adapter()
    assert a._extract_query({"query": "q"}) == "q"
    assert a._extract_query({"prompt": "p"}) == "p"
    assert a._extract_query({"messages": [{"role": "user", "content": "m"}]}) == "m"
    assert a._extract_query({}) is None


@pytest.mark.asyncio
async def test_answer_normalizes_perplexity_payload():
    a = _adapter()
    fake_resp = MagicMock()
    fake_resp.raise_for_status = MagicMock()
    fake_resp.json = MagicMock(return_value={
        "choices": [{"message": {"content": "BrandX is a widget platform."}}],
        "citations": ["https://a.com", "https://b.com"],
        "search_results": [{"title": "A", "url": "https://a.com"}],
        "model": "sonar",
        "usage": {"total_tokens": 42},
        "id": "abc",
    })
    a.client = AsyncMock()
    a.client.post = AsyncMock(return_value=fake_resp)

    resp = await a.execute(AdapterRequest(capability="answer", parameters={"query": "what is brandx"}))
    assert resp.status == "success"
    d = resp.data
    assert d["content"] == "BrandX is a widget platform."
    assert d["citations"] == ["https://a.com", "https://b.com"]
    assert d["search_results"] == [{"title": "A", "url": "https://a.com"}]
    assert d["model"] == "sonar"


@pytest.mark.asyncio
async def test_answer_missing_query_errors_not_crash():
    a = _adapter()
    a.client = AsyncMock()
    resp = await a.execute(AdapterRequest(capability="answer", parameters={}))
    assert resp.status == "error"
    assert "query" in (resp.error or "").lower()


@pytest.mark.asyncio
async def test_unknown_capability_raises():
    a = _adapter()
    with pytest.raises(ValueError):
        await a.execute(AdapterRequest(capability="bogus", parameters={}))


def test_env_key_resolution(monkeypatch):
    monkeypatch.setenv("PERPLEXITY_API_KEY", "env-key-123")
    a = PerplexityAdapter(AdapterConfig(name="perplexity", category=AdapterCategory.AI))
    assert a.api_key == "env-key-123"
