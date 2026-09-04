"""Regression: OpenAI + Gemini `answer` capability normalisation (GEO Phase B3).

covers: geo-phase-b3 openai-answer gemini-answer

Both engines must normalise their web-search/grounding response to the same
engine-agnostic GEO contract the perplexity adapter uses:
  {content, citations, search_results, model}
so compute-facts is engine-independent. These tests stub the HTTP client with
the documented OpenAI (Responses API hosted web_search tool, url_citation
annotations on output_text — the Chat Completions search models were retired) and Gemini
(generateContent google_search groundingMetadata) response shapes — no network,
no API keys (which aren't present locally; live multi-engine runs on the Beast).
"""
from unittest.mock import AsyncMock, MagicMock

import pytest

from adapters.models import AdapterConfig, AdapterCategory, AdapterRequest
from adapters.implementations.ai.openai_adapter import OpenAIAdapter
from adapters.implementations.ai.claude_adapter import ClaudeAdapter
from business_adapters.implementations.ai.gemini_adapter import GeminiAdapter


def _resp(payload):
    r = MagicMock()
    r.raise_for_status = MagicMock()
    r.json = MagicMock(return_value=payload)
    return r


@pytest.mark.asyncio
async def test_openai_answer_uses_responses_web_search_and_normalizes_citations():
    """OpenAI retired the Chat Completions search models: `gpt-4o-search-preview`
    returned HTTP 404 "deprecated" on all 18 GEO combos of both 2026-09-01 legs
    while the workflow still reported completed. The answer capability must call
    the Responses API with the hosted web_search tool and normalise its
    url_citation annotations to the engine-agnostic contract."""
    a = OpenAIAdapter(AdapterConfig(name="openai", category=AdapterCategory.AI, api_key="x"))
    a.client = AsyncMock()
    a.client.post = AsyncMock(return_value=_resp({
        "id": "resp_x", "model": "gpt-5.6", "status": "completed",
        "output": [
            {"type": "web_search_call", "id": "ws_1", "status": "completed"},
            {"type": "message", "id": "msg_1", "status": "completed", "role": "assistant",
             "content": [{"type": "output_text", "text": "Acme and Globex lead.",
                          "annotations": [
                              {"type": "url_citation", "start_index": 0, "end_index": 4,
                               "url": "https://acme.com", "title": "Acme"},
                              {"type": "url_citation", "start_index": 9, "end_index": 15,
                               "url": "https://acme.com", "title": "dup"},
                              {"type": "url_citation", "start_index": 9, "end_index": 15,
                               "url": "https://globex.com", "title": "Globex"},
                          ]}]},
        ],
        "usage": {"input_tokens": 3, "output_tokens": 7, "total_tokens": 10},
    }))
    resp = await a._handle_answer(AdapterRequest(
        capability="answer", parameters={"query": "best widget", "max_tokens": 300}))
    assert resp.status == "success"
    d = resp.data
    assert d["content"] == "Acme and Globex lead."
    assert d["citations"] == ["https://acme.com", "https://globex.com"]  # deduped, ordered
    assert {s["url"] for s in d["search_results"]} == {"https://acme.com", "https://globex.com"}
    assert len(d["search_results"]) == 2
    assert d["model"] == "gpt-5.6"
    assert resp.tokens_used == 10

    # The request itself: Responses API, hosted web_search tool, search forced so a
    # GEO answer can never come from model memory without citations, and no
    # retired chat-search model anywhere in the payload.
    path = a.client.post.call_args.args[0]
    body = a.client.post.call_args.kwargs["json"]
    assert path == "/responses"
    assert body["input"] == "best widget"
    assert body["tools"][0]["type"] == "web_search"
    assert body["tool_choice"] == "required"
    assert body["max_output_tokens"] == 300
    assert "messages" not in body and "max_tokens" not in body
    assert "search-preview" not in body["model"]


@pytest.mark.asyncio
async def test_openai_answer_multiple_output_text_parts_are_joined():
    a = OpenAIAdapter(AdapterConfig(name="openai", category=AdapterCategory.AI, api_key="x"))
    a.client = AsyncMock()
    a.client.post = AsyncMock(return_value=_resp({
        "id": "resp_y", "model": "gpt-5.6",
        "output": [{"type": "message", "role": "assistant", "content": [
            {"type": "output_text", "text": "Part one. ", "annotations": []},
            {"type": "output_text", "text": "Part two.",
             "annotations": [{"type": "url_citation", "url": "https://c.com", "title": "C"}]},
        ]}],
        "usage": {"total_tokens": 4},
    }))
    resp = await a._handle_answer(AdapterRequest(capability="answer", parameters={"query": "q"}))
    assert resp.status == "success"
    assert resp.data["content"] == "Part one. Part two."
    assert resp.data["citations"] == ["https://c.com"]


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


@pytest.mark.asyncio
async def test_claude_answer_normalizes_web_search():
    a = ClaudeAdapter(AdapterConfig(name="claude", category=AdapterCategory.AI, api_key="x"))
    a.client = AsyncMock()
    a.client.post = AsyncMock(return_value=_resp({
        "content": [
            {"type": "text", "text": "BrandX is great.",
             "citations": [{"type": "web_search_result_location", "url": "https://a.com", "title": "A"}]},
            {"type": "web_search_tool_result",
             "content": [{"type": "web_search_result", "url": "https://b.com", "title": "B"}]},
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5},
        "model": "claude-opus-4-8", "id": "msg_x",
    }))
    resp = await a._handle_answer(AdapterRequest(capability="answer", parameters={"query": "what is brandx"}))
    assert resp.status == "success"
    d = resp.data
    assert d["content"] == "BrandX is great."
    assert d["citations"] == ["https://a.com", "https://b.com"]
    assert {s["url"] for s in d["search_results"]} == {"https://a.com", "https://b.com"}


@pytest.mark.asyncio
async def test_claude_answer_missing_query_errors():
    a = ClaudeAdapter(AdapterConfig(name="claude", category=AdapterCategory.AI, api_key="x"))
    a.client = AsyncMock()
    resp = await a._handle_answer(AdapterRequest(capability="answer", parameters={}))
    assert resp.status == "error"


def test_all_engines_expose_answer_capability():
    o = OpenAIAdapter(AdapterConfig(name="openai", category=AdapterCategory.AI, api_key="x"))
    g = GeminiAdapter(AdapterConfig(name="gemini", category=AdapterCategory.AI, api_key="x"))
    c = ClaudeAdapter(AdapterConfig(name="claude", category=AdapterCategory.AI, api_key="x"))
    assert any(cap.name == "answer" for cap in o.get_capabilities())
    assert any(cap.name == "answer" for cap in g.get_capabilities())
    assert any(cap.name == "answer" for cap in c.get_capabilities())
