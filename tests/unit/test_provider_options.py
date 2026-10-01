"""Tier 0 — provider options every model call sends (spec §7.5).

Ollama requests carry `num_ctx` (fixed buckets, the loaded context reused when it
is big enough and within the cap) and `keep_alive`; `tool_choice="none"` means
no tools in every adapter (it used to be sent as a tool named "none").
"""
import json

import httpx
import pytest

from adapters.models import AdapterCategory, AdapterConfig
from adapters.tool_calling import ToolCallingRequest
from llm import ollama_options
from llm.ollama_options import estimate_tokens, request_options, size_for, with_request_options

TOOL = {"name": "get_help", "description": "help", "parameters": {"type": "object", "properties": {}}}


@pytest.fixture(autouse=True)
def nothing_loaded(monkeypatch):
    async def none_loaded(base_url):
        return {}

    monkeypatch.setattr(ollama_options, "_loaded_contexts", none_loaded)
    for name in ("OLLAMA_NUM_CTX_MAX", "OLLAMA_NUM_CTX", "OLLAMA_KEEP_ALIVE"):
        monkeypatch.delenv(name, raising=False)


def test_every_request_gets_the_same_floor_until_it_needs_more(monkeypatch):
    # One size for every caller, worker and edition: a different num_ctx reloads the model.
    assert size_for(100) == 16384
    assert size_for(16000) == 16384
    assert size_for(20000) == 32768
    monkeypatch.setenv("OLLAMA_NUM_CTX", "8192")
    assert size_for(100) == 8192


def test_an_oversized_request_is_capped_and_says_so(monkeypatch, caplog):
    monkeypatch.setenv("OLLAMA_NUM_CTX_MAX", "32768")
    with caplog.at_level("WARNING", logger="llm.ollama_options"):
        assert size_for(200_000) == 32768
    assert "drop the oldest context" in caplog.text


async def test_cold_start_never_shrinks_below_the_floor():
    # No model loaded (or /api/ps unreachable): still the floor, never a smaller bucket.
    options = await request_options("http://ollama", "llama3.1:8b", 2000, 2000, {"temperature": 0.4})
    assert options == {"temperature": 0.4, "num_ctx": 16384}


async def test_a_larger_loaded_context_is_reused_not_shrunk(monkeypatch):
    async def loaded(base_url):
        return {"llama3.1:8b-instruct-q4_K_M": 32768, "glm-ocr:latest": 32768}

    monkeypatch.setattr(ollama_options, "_loaded_contexts", loaded)
    assert (await request_options("http://o", "llama3.1:8b-instruct-q4_K_M", 2000, 2000))["num_ctx"] == 32768
    assert (await request_options("http://o", "glm-ocr", 2000, 2000))["num_ctx"] == 32768  # untagged name


async def test_a_context_above_the_cap_is_not_reused(monkeypatch):
    async def loaded(base_url):
        return {"llama3.1:8b:latest": 65536, "llama3.1:8b": 65536}  # the host default

    monkeypatch.setattr(ollama_options, "_loaded_contexts", loaded)
    assert (await request_options("http://o", "llama3.1:8b", 2000, 2000))["num_ctx"] == 16384


async def test_keep_alive_only_when_asked_and_a_callers_value_wins():
    payload = await with_request_options("http://o", {"model": "m", "prompt": "x"})
    assert "keep_alive" not in payload and payload["options"]["num_ctx"] == 16384
    payload = await with_request_options("http://o", {"model": "m", "prompt": "x"}, keep_alive="30m")
    assert payload["keep_alive"] == "30m"
    payload = await with_request_options("http://o", {"model": "m", "keep_alive": 0}, keep_alive="30m")
    assert payload["keep_alive"] == 0


def test_estimate_counts_the_answer_budget():
    assert estimate_tokens(3500, 2000) == int((1000 + 2000) * 1.3)


def _capture():
    sent = {}

    def handler(request):
        sent["path"] = request.url.path
        sent["json"] = json.loads(request.content)
        if request.url.path.endswith("/api/chat"):
            return httpx.Response(200, json={"message": {"role": "assistant", "content": "ok"}, "done": True})
        if "anthropic" in str(request.url) or request.url.path.endswith("/messages"):
            return httpx.Response(200, json={"content": [{"type": "text", "text": "ok"}],
                                             "usage": {"input_tokens": 1, "output_tokens": 1}})
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "ok"}}],
                                         "usage": {"prompt_tokens": 1, "completion_tokens": 1}})

    return sent, httpx.MockTransport(handler)


def _request(choice):
    return ToolCallingRequest(messages=[{"role": "user", "content": "hi"}], tools=[TOOL],
                              model="m", tool_choice=choice)


async def test_ollama_sends_options_and_omits_tools_for_none():
    from adapters.implementations.ai.ollama_adapter import OllamaAdapter

    adapter = OllamaAdapter(AdapterConfig(name="o", version="1", category=AdapterCategory.AI,
                                          base_url="http://ollama", credentials={}))
    sent, transport = _capture()
    adapter.client = httpx.AsyncClient(base_url="http://ollama", transport=transport)
    await adapter.chat_with_tools(_request("none"))
    assert "tools" not in sent["json"]
    assert sent["json"]["options"]["num_ctx"] == 16384 and sent["json"]["keep_alive"] == "30m"
    await adapter.chat_with_tools(_request("auto"))
    assert [t["function"]["name"] for t in sent["json"]["tools"]] == ["get_help"]


@pytest.mark.parametrize("module,cls,expected", [
    ("adapters.implementations.ai.openai_adapter", "OpenAIAdapter", "none"),
    ("adapters.implementations.ai.vllm_adapter", "VLLMAdapter", "none"),
    ("adapters.implementations.ai.claude_adapter", "ClaudeAdapter", {"type": "none"}),
])
async def test_tool_choice_none_is_not_a_tool_named_none(module, cls, expected):
    import importlib

    adapter_cls = getattr(importlib.import_module(module), cls)
    adapter = adapter_cls(AdapterConfig(name="a", version="1", category=AdapterCategory.AI,
                                        base_url="http://provider", credentials={"api_key": "k"}))
    sent, transport = _capture()
    adapter.client = httpx.AsyncClient(base_url="http://provider", transport=transport)
    try:
        await adapter.chat_with_tools(_request("none"))
    except Exception:
        pass  # the stub answer may not parse for every provider; the request is what matters
    assert sent["json"]["tool_choice"] == expected
