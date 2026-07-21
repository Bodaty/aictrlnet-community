"""Unit tests for LLMGenerationEngine.generate() local-fallback honesty.

Covers the /llm/generate fallback path defects (2026-07-09 analysis):
- fallback responses must report the ACTUAL model/provider/cost, with
  disclosure metadata (requested_model / fallback_model / fallback_reason)
- the non-fallback path must be unchanged
"""
from unittest.mock import AsyncMock

import pytest

from llm.generation import LLMGenerationEngine
from llm.models import LLMRequest, LLMResponse, ModelProvider, ModelTier


@pytest.fixture
def engine():
    return LLMGenerationEngine()


def _local_response(model="gemma3:27b", tokens=42):
    return LLMResponse(
        text="local fallback text",
        model_used=model,
        provider=ModelProvider.OLLAMA,
        tier=ModelTier.BALANCED,
        tokens_used=tokens,
    )


def _wire_fallback(engine, monkeypatch, *, ollama=("gemma3:27b",), vllm=()):
    """Force generate() into the local-fallback path deterministically (no network)."""
    monkeypatch.setenv("DEFAULT_LLM_MODEL", "gemini-2.0-flash-vertex")  # not locally servable
    monkeypatch.setattr(engine, "_is_cloud_environment", lambda: False)
    monkeypatch.setattr(engine, "_get_ollama_models", AsyncMock(return_value=list(ollama)))
    monkeypatch.setattr(engine, "_get_vllm_models", AsyncMock(return_value=list(vllm)))


async def test_fallback_response_reports_actual_model_and_provider(engine, monkeypatch):
    _wire_fallback(engine, monkeypatch)
    monkeypatch.setattr(
        engine, "_generate_with_adapter",
        AsyncMock(side_effect=RuntimeError("ANTHROPIC_API_KEY not configured")),
    )
    monkeypatch.setattr(engine, "_generate_with_ollama", AsyncMock(return_value=_local_response()))

    response = await engine.generate(LLMRequest(prompt="hi", model_override="claude-3-sonnet"))

    assert response.model_used == "gemma3:27b"
    assert response.provider == ModelProvider.OLLAMA


async def test_fallback_cost_is_priced_from_actual_model(engine, monkeypatch):
    _wire_fallback(engine, monkeypatch)
    monkeypatch.setattr(
        engine, "_generate_with_adapter",
        AsyncMock(side_effect=RuntimeError("ANTHROPIC_API_KEY not configured")),
    )
    monkeypatch.setattr(
        engine, "_generate_with_ollama",
        AsyncMock(return_value=_local_response(tokens=100_000)),
    )

    response = await engine.generate(LLMRequest(prompt="hi", model_override="claude-3-sonnet"))

    # local models are not in the pricing table -> free, NOT claude pricing
    assert response.cost == 0.0


async def test_fallback_discloses_requested_and_fallback_model(engine, monkeypatch):
    _wire_fallback(engine, monkeypatch)
    monkeypatch.setattr(
        engine, "_generate_with_adapter",
        AsyncMock(side_effect=RuntimeError("ANTHROPIC_API_KEY not configured")),
    )
    monkeypatch.setattr(engine, "_generate_with_ollama", AsyncMock(return_value=_local_response()))

    response = await engine.generate(LLMRequest(prompt="hi", model_override="claude-3-sonnet"))

    assert response.metadata["requested_model"] == "claude-3-sonnet"
    assert response.metadata["fallback_model"] == "gemma3:27b"
    assert response.metadata["fallback_reason"]


# --- fallback picker (Fix 3) ------------------------------------------------

QWEN = "Qwen/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit"


def _wire_picker(engine, monkeypatch, *, default, vllm, ollama):
    monkeypatch.setenv("DEFAULT_LLM_MODEL", default)
    monkeypatch.setattr(engine, "_get_vllm_models", AsyncMock(return_value=list(vllm)))
    monkeypatch.setattr(engine, "_get_ollama_models", AsyncMock(return_value=list(ollama)))


async def test_picker_honors_vllm_prefixed_configured_default(engine, monkeypatch):
    _wire_picker(engine, monkeypatch, default=f"vllm:{QWEN}", vllm=[QWEN], ollama=["gemma3:27b"])
    assert await engine._pick_fallback_local_model() == f"vllm:{QWEN}"


async def test_picker_prefers_vllm_over_ollama_when_default_not_local(engine, monkeypatch):
    _wire_picker(engine, monkeypatch, default="gemini-2.0-flash-vertex", vllm=[QWEN], ollama=["gemma3:27b"])
    assert await engine._pick_fallback_local_model() == f"vllm:{QWEN}"


async def test_picker_falls_back_to_ollama_when_vllm_empty(engine, monkeypatch):
    _wire_picker(engine, monkeypatch, default="gemini-2.0-flash-vertex", vllm=[], ollama=["gemma3:27b"])
    assert await engine._pick_fallback_local_model() == "gemma3:27b"


async def test_picker_honors_pulled_ollama_configured_default(engine, monkeypatch):
    _wire_picker(engine, monkeypatch, default="llama3.2:3b", vllm=[QWEN], ollama=["gemma3:27b", "llama3.2:3b"])
    assert await engine._pick_fallback_local_model() == "llama3.2:3b"


async def test_picker_returns_none_when_nothing_local(engine, monkeypatch):
    _wire_picker(engine, monkeypatch, default="gemini-2.0-flash-vertex", vllm=[], ollama=[])
    assert await engine._pick_fallback_local_model() is None


async def test_picker_never_returns_the_model_that_just_failed(engine, monkeypatch):
    _wire_picker(engine, monkeypatch, default=f"vllm:{QWEN}", vllm=[QWEN, "other/Model"], ollama=[])
    assert await engine._pick_fallback_local_model(exclude=f"vllm:{QWEN}") == "vllm:other/Model"


async def test_generate_dispatches_vllm_fallback_via_adapter_path(engine, monkeypatch):
    _wire_picker(engine, monkeypatch, default=f"vllm:{QWEN}", vllm=[QWEN], ollama=[])
    monkeypatch.setattr(engine, "_is_cloud_environment", lambda: False)

    vllm_response = LLMResponse(
        text="qwen text",
        model_used=f"vllm:{QWEN}",
        provider=ModelProvider.VLLM,
        tier=ModelTier.QUALITY,
        tokens_used=9,
    )
    adapter_mock = AsyncMock(side_effect=[RuntimeError("ANTHROPIC_API_KEY not configured"), vllm_response])
    monkeypatch.setattr(engine, "_generate_with_adapter", adapter_mock)
    ollama_mock = AsyncMock()
    monkeypatch.setattr(engine, "_generate_with_ollama", ollama_mock)

    response = await engine.generate(LLMRequest(prompt="hi", model_override="claude-3-sonnet"))

    assert response.model_used == f"vllm:{QWEN}"
    assert response.provider == ModelProvider.VLLM
    assert response.metadata["requested_model"] == "claude-3-sonnet"
    assert response.metadata["fallback_model"] == f"vllm:{QWEN}"
    ollama_mock.assert_not_awaited()
    # the retry goes through the adapter path with the PREFIXED vllm model
    retry_call = adapter_mock.await_args_list[1]
    assert retry_call.args[1] == f"vllm:{QWEN}"
    assert retry_call.args[2] == ModelProvider.VLLM


# --- capability resolution (Fix 2) -----------------------------------------


class _Cap:
    def __init__(self, name):
        self.name = name


class _StubChatAdapter:
    """Adapter double that validates capability like BaseAdapter.validate_request."""

    def __init__(self, capability_names):
        self._capability_names = list(capability_names)
        self.last_request = None

    def get_capabilities(self):
        return [_Cap(n) for n in self._capability_names]

    async def execute(self, request):
        if request.capability not in self._capability_names:
            raise ValueError(f"Unknown capability: {request.capability}")
        self.last_request = request
        from types import SimpleNamespace
        return SimpleNamespace(
            data={"choices": [{"message": {"content": "stub reply"}}]},
            metadata={},
            tokens_used=7,
            cost=0.0,
            status="success",
        )


def test_resolve_chat_capability_picks_registered_name():
    from llm.generation import resolve_chat_capability

    assert resolve_chat_capability(_StubChatAdapter(["chat_completion"])) == "chat_completion"
    assert resolve_chat_capability(_StubChatAdapter(["chat"])) == "chat"
    assert resolve_chat_capability(_StubChatAdapter(["chat_completion", "chat"])) == "chat_completion"


async def test_execute_adapter_chat_reaches_chat_completion_only_adapter(engine):
    adapter = _StubChatAdapter(["answer", "chat_completion", "text_completion"])  # Claude-shaped

    response = await engine._execute_adapter_chat(
        adapter, LLMRequest(prompt="hi"), "claude-3-sonnet", ModelProvider.ANTHROPIC
    )

    assert response.text == "stub reply"
    assert adapter.last_request.capability == "chat_completion"


async def test_execute_adapter_chat_still_works_for_chat_only_adapter(engine):
    adapter = _StubChatAdapter(["chat"])  # Gemini/Cohere-shaped

    response = await engine._execute_adapter_chat(
        adapter, LLMRequest(prompt="hi"), "gemini-2.0-flash", ModelProvider.GEMINI
    )

    assert response.text == "stub reply"
    assert adapter.last_request.capability == "chat"


async def test_generate_reaches_anthropic_adapter_without_fallback(engine, monkeypatch):
    """A working Claude-shaped adapter must be reachable end-to-end (no fallback)."""
    adapter = _StubChatAdapter(["chat_completion"])

    async def fake_get(provider):
        return adapter

    monkeypatch.setattr("llm.service._AdapterProvider.get", fake_get)
    # if the capability were still wrong, the fallback path must not mask it
    monkeypatch.setattr(engine, "_get_ollama_models", AsyncMock(return_value=[]))
    monkeypatch.setattr(engine, "_get_vllm_models", AsyncMock(return_value=[]))
    monkeypatch.setattr(engine, "_is_cloud_environment", lambda: False)

    response = await engine.generate(LLMRequest(prompt="hi", model_override="claude-3-sonnet"))

    assert response.text == "stub reply"
    assert response.model_used == "claude-3-sonnet"
    assert "fallback_model" not in response.metadata


async def test_non_fallback_path_unchanged(engine, monkeypatch):
    adapter_response = LLMResponse(
        text="cloud text",
        model_used="claude-3-sonnet",
        provider=ModelProvider.ANTHROPIC,
        tier=ModelTier.PREMIUM,
        tokens_used=1000,
    )
    monkeypatch.setattr(engine, "_generate_with_adapter", AsyncMock(return_value=adapter_response))

    response = await engine.generate(LLMRequest(prompt="hi", model_override="claude-3-sonnet"))

    assert response.model_used == "claude-3-sonnet"
    assert response.provider == ModelProvider.ANTHROPIC
    assert response.cost == engine._calculate_cost("claude-3-sonnet", 1000)
    assert "fallback_model" not in response.metadata
    assert "requested_model" not in response.metadata
