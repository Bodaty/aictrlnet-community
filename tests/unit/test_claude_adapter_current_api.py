"""The Claude adapter sends request bodies current Claude models accept.

Current models reject sampling parameters (temperature, top_p, top_k) with a 400,
the current Opus (claude-opus-5-5) rejects a forced tool_choice ("any" or a named
tool), and its effort defaults to medium unless set. Each test captures the JSON the
adapter actually posts.
"""
import json

import httpx
import pytest

from adapters.implementations.ai.claude_adapter import ClaudeAdapter
from adapters.models import AdapterCategory, AdapterConfig, AdapterRequest
from adapters.tool_calling import ToolCallingRequest

SAMPLING = {"temperature", "top_p", "top_k"}


def _reply(stream=False):
    return {"id": "msg_1", "model": "claude-opus-5-5", "stop_reason": "end_turn",
            "content": [{"type": "thinking", "thinking": ""}, {"type": "text", "text": "hello"}],
            "usage": {"input_tokens": 10, "output_tokens": 5}}


@pytest.fixture
def adapter():
    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=_reply())

    a = ClaudeAdapter(AdapterConfig(name="claude", category=AdapterCategory.AI, api_key="k"))
    a.client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://api.anthropic.com/v1")
    a.sent = sent
    return a


async def test_chat_completion_sends_no_sampling_parameters(adapter):
    response = await adapter.execute(AdapterRequest(capability="chat_completion", parameters={
        "model": "claude-opus-5-5", "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.2, "top_p": 0.9, "top_k": 5}))
    assert response.status == "success"
    assert response.data["choices"][0]["message"]["content"] == "hello"   # thinking block skipped
    assert not SAMPLING & set(adapter.sent[-1])


async def test_text_completion_sends_no_sampling_parameters(adapter):
    await adapter.execute(AdapterRequest(capability="text_completion", parameters={
        "model": "claude-opus-5-5", "prompt": "hi", "temperature": 0.5}))
    assert not SAMPLING & set(adapter.sent[-1])


async def test_the_capabilities_no_longer_offer_temperature(adapter):
    for cap in adapter.get_capabilities():
        assert not SAMPLING & set(cap.parameters or {}), cap.name


async def test_opus_5_5_gets_an_explicit_effort(adapter):
    await adapter.execute(AdapterRequest(capability="chat_completion", parameters={
        "model": "claude-opus-5-5", "messages": [{"role": "user", "content": "hi"}]}))
    assert adapter.sent[-1]["output_config"] == {"effort": "high"}


@pytest.mark.parametrize("model", ["claude-opus-4-8", "claude-sonnet-4-5", "claude-haiku-4-5"])
async def test_models_that_default_to_high_or_reject_effort_get_none(adapter, model):
    await adapter.execute(AdapterRequest(capability="chat_completion", parameters={
        "model": model, "messages": [{"role": "user", "content": "hi"}]}))
    assert "output_config" not in adapter.sent[-1]


async def test_a_requested_effort_is_passed_through(adapter):
    await adapter.execute(AdapterRequest(capability="chat_completion", parameters={
        "model": "claude-opus-5-5", "messages": [{"role": "user", "content": "hi"}], "effort": "low"}))
    assert adapter.sent[-1]["output_config"] == {"effort": "low"}


def _tool_request(tool_choice, schema=None):
    schema = schema or {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"],
                        "additionalProperties": False}
    return ToolCallingRequest(
        model="claude-opus-5-5", tool_choice=tool_choice, temperature=0.3, system_prompt="Be brief.",
        messages=[{"role": "user", "content": "look it up"}],
        tools=[{"name": "search", "description": "Search", "parameters": schema},
               {"name": "fetch", "description": "Fetch", "parameters": schema}])


@pytest.mark.parametrize("tool_choice", ["required", "search"])
async def test_a_forced_tool_choice_becomes_auto_with_strict_tools_and_an_instruction(adapter, tool_choice):
    await adapter.chat_with_tools(_tool_request(tool_choice))
    body = adapter.sent[-1]
    assert not SAMPLING & set(body)
    assert body["tool_choice"] == {"type": "auto"}
    assert all(t.get("strict") is True for t in body["tools"])
    system = body["system"][0]["text"] if isinstance(body["system"], list) else body["system"]
    assert system.startswith("Be brief.")
    if tool_choice == "required":
        assert "You must call one of the tools" in system
    else:
        assert "You must call the search tool" in system


async def test_a_tool_whose_schema_is_not_strict_compatible_is_not_marked_strict(adapter):
    loose = {"type": "object", "properties": {"q": {"type": "string"}}}
    await adapter.chat_with_tools(_tool_request("required", schema=loose))
    assert all("strict" not in t for t in adapter.sent[-1]["tools"])
    assert adapter.sent[-1]["tool_choice"] == {"type": "auto"}


@pytest.mark.parametrize("tool_choice,expected", [("auto", None), ("none", {"type": "none"})])
async def test_auto_and_none_are_unchanged(adapter, tool_choice, expected):
    await adapter.chat_with_tools(_tool_request(tool_choice))
    body = adapter.sent[-1]
    assert body.get("tool_choice") == expected
    assert all("strict" not in t for t in body["tools"])
    system = body["system"][0]["text"] if isinstance(body["system"], list) else body["system"]
    assert system == "Be brief."


def test_the_default_anthropic_model_is_the_current_opus():
    from llm.tier_resolver import PROVIDER_DEFAULT_MODELS

    assert PROVIDER_DEFAULT_MODELS["anthropic"] == "claude-opus-5-5"


@pytest.mark.parametrize("model,cost", [
    ("claude-opus-5-5", 4.0 + 20.0),
    ("claude-opus-4-8", 5.0 + 25.0),
    ("claude-sonnet-5-5", 2.0 + 10.0),
    ("claude-haiku-5-5", 0.10 + 0.50),
])
def test_cost_uses_the_models_own_price(adapter, model, cost):
    assert adapter._estimate_cost(model, 1_000_000, 1_000_000) == pytest.approx(cost)


async def test_the_health_check_asks_a_model_that_is_still_served(adapter):
    await adapter._perform_health_check()
    assert adapter.sent[-1]["model"] == "claude-haiku-5-5"


async def test_the_geo_answer_capability_stays_on_its_measured_model_and_search_tool(adapter):
    """GEO measurements are a series: their model moves only by a deliberate, recorded switch."""
    await adapter.execute(AdapterRequest(capability="answer", parameters={"query": "what is brandx"}))
    body = adapter.sent[-1]
    assert body["model"] == "claude-opus-4-8"
    assert body["tools"] == [{"type": "web_search_20250305", "name": "web_search"}]
    assert not SAMPLING & set(body)
    answer = next(c for c in adapter.get_capabilities() if c.name == "answer")
    assert answer.parameters["model"]["default"] == "claude-opus-4-8"
