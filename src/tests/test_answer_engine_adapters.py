"""Regression: OpenAI + Gemini `answer` capability normalisation (GEO Phase B3).

covers: geo-phase-b3 openai-answer gemini-answer

Both engines must normalise their web-search/grounding response to the same
engine-agnostic GEO contract the perplexity adapter uses:
  {content, citations, search_results, model}
so compute-facts is engine-independent. These tests stub the HTTP client with
the documented OpenAI (Chat Completions web_search annotations) and Gemini
(generateContent google_search groundingMetadata) response shapes — no network,
no API keys (which aren't present locally; live multi-engine runs on the Beast).
"""
from unittest.mock import AsyncMock, MagicMock

import pytest

from adapters.models import AdapterConfig, AdapterCategory, AdapterRequest
from adapters.implementations.ai.openai_adapter import OpenAIAdapter
from business_adapters.implementations.ai.gemini_adapter import GeminiAdapter


def _resp(payload):
    r = MagicMock()
    r.raise_for_status = MagicMock()
    r.json = MagicMock(return_value=payload)
    return r


@pytest.mark.asyncio
async def test_openai_answer_normalizes_annotations():
    a = OpenAIAdapter(AdapterConfig(name="openai", category=AdapterCategory.AI, api_key="x"))
    a.client = AsyncMock()
    a.client.post = AsyncMock(return_value=_resp({
        "choices": [{"message": {
            "content": "Acme and Globex lead.",
            "annotations": [
                {"type": "url_citation", "url_citation": {"url": "https://acme.com", "title": "Acme"}},
                {"type": "url_citation", "url_citation": {"url": "https://acme.com", "title": "dup"}},
                {"type": "url_citation", "url_citation": {"url": "https://globex.com", "title": "Globex"}},
            ],
        }}],
        "usage": {"total_tokens": 10}, "model": "gpt-4o-search-preview", "id": "x",
    }))
    resp = await a._handle_answer(AdapterRequest(capability="answer", parameters={"query": "best widget"}))
    assert resp.status == "success"
    d = resp.data
    assert d["content"] == "Acme and Globex lead."
    assert d["citations"] == ["https://acme.com", "https://globex.com"]  # deduped
    assert {s["url"] for s in d["search_results"]} == {"https://acme.com", "https://globex.com"}


@pytest.mark.asyncio
async def test_openai_answer_missing_query_errors():
    a = OpenAIAdapter(AdapterConfig(name="openai", category=AdapterCategory.AI, api_key="x"))
    a.client = AsyncMock()
    resp = await a._handle_answer(AdapterRequest(capability="answer", parameters={}))
    assert resp.status == "error"
    assert "query" in (resp.error or "").lower()


@pytest.mark.asyncio
async def test_gemini_answer_normalizes_grounding():
    a = GeminiAdapter(AdapterConfig(name="gemini", category=AdapterCategory.AI, api_key="x"))
    a.client = AsyncMock()
    a.client.post = AsyncMock(return_value=_resp({
        "candidates": [{
            "content": {"parts": [{"text": "BrandX "}, {"text": "is great."}]},
            "groundingMetadata": {
                "groundingChunks": [
                    {"web": {"uri": "https://a.com", "title": "A"}},
                    {"web": {"uri": "https://b.com", "title": "B"}},
                ],
                "webSearchQueries": ["brandx"],
            },
        }],
        "usageMetadata": {"totalTokenCount": 5},
    }))
    resp = await a._handle_answer(AdapterRequest(capability="answer", parameters={"query": "what is brandx"}))
    assert resp.status == "success"
    d = resp.data
    assert d["content"] == "BrandX is great."
    assert d["citations"] == ["https://a.com", "https://b.com"]
    assert {s["url"] for s in d["search_results"]} == {"https://a.com", "https://b.com"}


@pytest.mark.asyncio
async def test_gemini_answer_missing_query_errors():
    a = GeminiAdapter(AdapterConfig(name="gemini", category=AdapterCategory.AI, api_key="x"))
    a.client = AsyncMock()
    resp = await a._handle_answer(AdapterRequest(capability="answer", parameters={}))
    assert resp.status == "error"


def test_both_expose_answer_capability():
    o = OpenAIAdapter(AdapterConfig(name="openai", category=AdapterCategory.AI, api_key="x"))
    g = GeminiAdapter(AdapterConfig(name="gemini", category=AdapterCategory.AI, api_key="x"))
    assert any(c.name == "answer" for c in o.get_capabilities())
    assert any(c.name == "answer" for c in g.get_capabilities())
