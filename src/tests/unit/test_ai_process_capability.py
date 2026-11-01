"""Unit tests for AIProcessNode._call_adapter capability handling.

Pre-fix, _call_adapter sent capability="generate" (failure silently swallowed)
then capability="chat" (unprotected) — but Claude/OpenAI adapters register only
chat_completion/answer/text_completion, so any aiProcess node routed to those
adapters hard-failed. The chat-style fallback must send the capability the
adapter actually registers, and the primary failure must be logged.
"""
import logging

import pytest

from adapters.models import AdapterCapability, AdapterResponse
from nodes.implementations.ai_process_node import AIProcessNode
from nodes.models import NodeConfig, NodeType


class _StubAdapter:
    """Validates capability like BaseAdapter.validate_request; chat-style reply."""

    def __init__(self, capability_names):
        self._capability_names = list(capability_names)
        self.requests = []

    def get_capabilities(self):
        return [AdapterCapability(name=n, description=n) for n in self._capability_names]

    async def execute(self, request):
        self.requests.append(request)
        if request.capability not in self._capability_names:
            raise ValueError(f"Unknown capability: {request.capability}")
        return AdapterResponse(
            request_id=request.id,
            capability=request.capability,
            status="success",
            data={"choices": [{"message": {"content": "stub reply"}}]},
            duration_ms=1.0,
        )


@pytest.fixture
def node(monkeypatch):
    monkeypatch.setenv("DEFAULT_LLM_MODEL", "llama3.2:3b")
    return AIProcessNode(NodeConfig(id="n", name="ai", type=NodeType.TASK, parameters={}))


async def test_call_adapter_reaches_chat_completion_only_adapter(node):
    adapter = _StubAdapter(["answer", "chat_completion", "text_completion"])  # Claude-shaped

    response = await node._call_adapter(
        adapter, "generate", {"prompt": "hi", "model": "claude-3-sonnet-20240229"}
    )

    assert response.status != "error"
    assert response.data["text"] == "stub reply"
    assert adapter.requests[-1].capability == "chat_completion"


async def test_call_adapter_still_works_for_chat_adapter(node):
    adapter = _StubAdapter(["chat"])  # Ollama/Gemini/Cohere-shaped

    response = await node._call_adapter(adapter, "generate", {"prompt": "hi", "model": "llama3.2:3b"})

    assert response.data["text"] == "stub reply"
    assert adapter.requests[-1].capability == "chat"


async def test_call_adapter_template_hardcoded_claude_model_regression(node):
    """System templates hardcode model=claude-3.5-sonnet — the capability path
    must not hard-fail the node when that routes to the claude adapter."""
    adapter = _StubAdapter(["chat_completion"])

    response = await node._call_adapter(adapter, "generate", {"prompt": "hi", "model": "claude-3.5-sonnet"})

    assert response.status != "error"
    assert response.data["text"] == "stub reply"


async def test_call_adapter_logs_primary_capability_failure(node, caplog):
    adapter = _StubAdapter(["chat_completion"])

    with caplog.at_level(logging.WARNING, logger="nodes.implementations.ai_process_node"):
        await node._call_adapter(adapter, "generate", {"prompt": "hi", "model": "claude-3-sonnet-20240229"})

    assert any("generate" in record.getMessage() for record in caplog.records)


async def test_default_model_prefers_org_setting(monkeypatch, node):
    class Org:
        preferred_model = "gpt-4o"
        trial_mode = False
        allowed_providers = []

        def has_own_key(self, provider=None):
            return True

    seen = []

    async def fake_get_org_llm_settings(tenant_id, db):
        seen.append(tenant_id)
        return Org()

    monkeypatch.setattr("llm.org_llm_settings.get_org_llm_settings", fake_get_org_llm_settings)
    await node._load_org_llm_settings("t-doc")
    model = await node._resolve_default_model()
    assert seen == ["t-doc"]
    assert model == "gpt-4o"


async def test_ai_process_reused_node_resolves_each_input(monkeypatch):
    from adapters.base_adapter import BaseAdapter
    from adapters.registry import adapter_registry
    from adapters.models import AdapterCategory

    recorded = []

    class _RecordingAdapter(BaseAdapter):
        async def initialize(self):
            return None

        async def shutdown(self):
            return None

        def get_capabilities(self):
            return [AdapterCapability(name="generate", description="generate")]

        async def execute(self, request):
            recorded.append(request)
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={"text": "ok"},
                duration_ms=0.0,
            )

    monkeypatch.delenv("LLM_SCRIPTED_MODE", raising=False)
    adapter_registry.register_adapter_class("recording_a1", _RecordingAdapter, AdapterCategory.AI)

    request_parameters = {"page": "{{page_id}}"}
    node = AIProcessNode(NodeConfig(
        id="n",
        name="ai",
        type=NodeType.TASK,
        parameters={
            "ai_task": "custom",
            "adapter_id": "recording_a1",
            "capability": "generate",
            "request_parameters": request_parameters,
        },
    ))

    for page_id in ("p1", "p2", "p3"):
        await node.execute({"page_id": page_id}, {})

    assert [r.parameters["page"] for r in recorded] == ["p1", "p2", "p3"]
    assert node.config.parameters["request_parameters"] == {"page": "{{page_id}}"}
