"""Tool-call history in either shape, reasoning off on request, and stream errors that surface.

- OpenAI-compatible clients send assistant tool calls back as
  ``{id, type, function: {name, arguments: "<json>"}}`` and tool turns without ``content``;
  both adapters used to KeyError on the second tool turn.
- ``think=False`` reaches the provider (Ollama ``think``, vLLM ``enable_thinking``).
- An adapter ``error`` event used to be dropped (an empty, successful completion), and a
  failure after text had streamed re-ran the request, sending that text twice.
"""
import asyncio
import json

import httpx
import pytest

from adapters.models import AdapterCategory, AdapterConfig
from adapters.tool_calling import ToolCallingRequest, ToolCallingStreamEvent, normalize_tool_call
from llm import ollama_options

TOOL = {"name": "confirm_identity", "description": "d", "parameters": {"type": "object", "properties": {}}}

HISTORY = [
    {"role": "system", "content": "s"},
    {"role": "user", "content": "Yes, this is Margaret."},
    {
        "role": "assistant",
        "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "confirm_identity", "arguments": "{}"}},
            {"id": "call_2", "type": "function", "function": {"name": "record", "arguments": '{"a": 1}'}},
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": '{"ok": true}'},
    {"role": "tool", "tool_call_id": "call_2"},
]


@pytest.fixture(autouse=True)
def nothing_loaded(monkeypatch):
    async def none_loaded(base_url):
        return {}

    monkeypatch.setattr(ollama_options, "_loaded_contexts", none_loaded)


def test_normalize_accepts_both_shapes():
    assert normalize_tool_call({"id": "1", "name": "n", "arguments": {"a": 1}}) == {
        "id": "1", "name": "n", "arguments": {"a": 1}}
    assert normalize_tool_call(
        {"id": "2", "type": "function", "function": {"name": "n", "arguments": '{"a": 1}'}}
    ) == {"id": "2", "name": "n", "arguments": {"a": 1}}
    assert normalize_tool_call({"function": {"name": "n", "arguments": ""}})["arguments"] == {}
    assert normalize_tool_call({"function": {"name": "n", "arguments": "not json"}})["arguments"] == {}


def _capture(reply):
    sent = {}

    def handler(request):
        sent["json"] = json.loads(request.content)
        return httpx.Response(200, json=reply)

    return sent, httpx.MockTransport(handler)


def _config():
    return AdapterConfig(name="a", version="1", category=AdapterCategory.AI,
                         base_url="http://provider", credentials={"api_key": "k"})


async def test_ollama_takes_openai_shaped_history_and_think():
    from adapters.implementations.ai.ollama_adapter import OllamaAdapter

    adapter = OllamaAdapter(_config())
    sent, transport = _capture({"message": {"role": "assistant", "content": "ok"}, "done": True})
    adapter.client = httpx.AsyncClient(base_url="http://provider", transport=transport)

    await adapter.chat_with_tools(ToolCallingRequest(messages=HISTORY, tools=[TOOL], model="m", think=False))
    msgs = sent["json"]["messages"]
    assert msgs[2]["tool_calls"] == [
        {"function": {"name": "confirm_identity", "arguments": {}}},
        {"function": {"name": "record", "arguments": {"a": 1}}},
    ]
    assert msgs[2]["content"] == "" and msgs[4] == {"role": "tool", "content": ""}
    assert sent["json"]["think"] is False

    await adapter.chat_with_tools(ToolCallingRequest(messages=HISTORY, tools=[TOOL], model="m"))
    assert "think" not in sent["json"]


async def test_ollama_stream_takes_the_same_history():
    from adapters.implementations.ai.ollama_adapter import OllamaAdapter

    adapter = OllamaAdapter(_config())
    sent, transport = _capture({"message": {"role": "assistant", "content": "ok"}, "done": True})
    adapter.client = httpx.AsyncClient(base_url="http://provider", transport=transport)
    events = [e async for e in adapter.chat_with_tools_stream(
        ToolCallingRequest(messages=HISTORY, tools=[TOOL], model="m", think=False))]
    assert sent["json"]["messages"][2]["tool_calls"][1]["function"]["arguments"] == {"a": 1}
    assert sent["json"]["think"] is False
    assert [e.type for e in events][-1] == "done"


async def test_vllm_encodes_arguments_once_and_keeps_ids():
    from adapters.implementations.ai.vllm_adapter import VLLMAdapter

    adapter = VLLMAdapter(_config())
    sent, transport = _capture({"choices": [{"message": {"role": "assistant", "content": "ok"}}],
                                "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
    adapter.client = httpx.AsyncClient(base_url="http://provider", transport=transport)
    try:
        await adapter.chat_with_tools(ToolCallingRequest(messages=HISTORY, tools=[TOOL], model="m", think=False))
    except Exception:
        pass  # the request is what matters
    msgs = sent["json"]["messages"]
    calls = msgs[2]["tool_calls"]
    assert [c["id"] for c in calls] == ["call_1", "call_2"]
    assert json.loads(calls[1]["function"]["arguments"]) == {"a": 1}  # not a JSON string of a string
    assert msgs[3]["tool_call_id"] == "call_1" and msgs[4]["content"] == ""
    assert sent["json"]["chat_template_kwargs"] == {"enable_thinking": False}


class _FakeAdapter:
    def __init__(self, events):
        self._events = events
        self.requests = []

    async def chat_with_tools_stream(self, request):
        self.requests.append(request)
        for e in self._events:
            if isinstance(e, BaseException):
                raise e
            yield e


@pytest.fixture
def service(monkeypatch):
    from llm import service as svc
    from llm.models import LLMToolResponse, ModelProvider, ModelTier

    monkeypatch.delenv("LLM_SCRIPTED_MODE", raising=False)
    s = svc.LLMService()
    calls = {"text_tools": 0}

    async def select(_request):
        return "qwen3:14b", ModelTier.QUALITY

    async def text_tools(**kwargs):
        calls["text_tools"] += 1
        return LLMToolResponse(text="fallback", model_used="qwen3:14b", provider=ModelProvider.OLLAMA)

    monkeypatch.setattr(s.generation_engine, "_select_model", select)
    monkeypatch.setattr(s, "_supports_native_tools", lambda provider, model: True)
    monkeypatch.setattr(s, "_generate_with_text_tools", text_tools)
    return s, calls, svc


def _use(monkeypatch, svc, adapter):
    async def get(provider):
        return adapter

    monkeypatch.setattr(svc._AdapterProvider, "get", get)


async def _collect(s, **kw):
    return [e async for e in s.generate_with_tools_stream(prompt=None, tools=[], messages=HISTORY, **kw)]


async def test_error_after_text_is_relayed_and_not_rerun(service, monkeypatch):
    s, calls, svc = service
    _use(monkeypatch, svc, _FakeAdapter([
        ToolCallingStreamEvent(type="text_delta", text="Thank you, Margaret."),
        ToolCallingStreamEvent(type="error", text="upstream closed"),
    ]))
    events = await _collect(s)
    assert [e["type"] for e in events] == ["text_delta", "error", "complete"]
    assert "upstream closed" in events[1]["message"]
    assert events[2]["response"].text == "Thank you, Margaret."
    assert calls["text_tools"] == 0


async def test_error_before_any_text_falls_back_once(service, monkeypatch):
    s, calls, svc = service
    _use(monkeypatch, svc, _FakeAdapter([ToolCallingStreamEvent(type="error", text="boom")]))
    events = await _collect(s)
    assert [e["type"] for e in events] == ["error", "text_delta", "complete"]
    assert events[-1]["response"].text == "fallback" and calls["text_tools"] == 1


async def test_think_reaches_the_adapter_request(service, monkeypatch):
    s, _, svc = service
    adapter = _FakeAdapter([ToolCallingStreamEvent(type="done")])
    _use(monkeypatch, svc, adapter)
    await _collect(s, think=False)
    assert adapter.requests[0].think is False


async def test_cancellation_is_never_swallowed(service, monkeypatch):
    # Barge-in cancels the consumer; the model request must abort, not fall back.
    s, calls, svc = service
    _use(monkeypatch, svc, _FakeAdapter([
        ToolCallingStreamEvent(type="text_delta", text="Hello"),
        asyncio.CancelledError(),
    ]))
    with pytest.raises(asyncio.CancelledError):
        await _collect(s)
    assert calls["text_tools"] == 0

