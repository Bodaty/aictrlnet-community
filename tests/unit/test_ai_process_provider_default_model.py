"""An AI step whose adapter is a named provider, with no model set anywhere, gets a
model that provider serves, never the deployment's default for another provider.

The org's preferred_provider picks the adapter; when preferred_model is empty the
step used to be sent DEFAULT_LLM_MODEL (a vLLM or Ollama name on a self-hosted box).
"""
import pytest

from nodes.implementations.ai_process_node import AIProcessNode
from nodes.models import NodeConfig, NodeType

VLLM_DEFAULT = "vllm:cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit"


class _Org:
    """An org that picked a provider and no model, with its own key."""

    def __init__(self, provider, model=None):
        self.preferred_provider = provider
        self.preferred_model = model
        self.trial_mode = False
        self.allowed_providers = []

    def has_own_key(self, provider=None):
        return True


@pytest.fixture
def node():
    return AIProcessNode(NodeConfig(id="n", name="ai", type=NodeType.TASK, parameters={}))


async def _default_for(node, monkeypatch, adapter_id, org, env_default=VLLM_DEFAULT):
    monkeypatch.setenv("DEFAULT_LLM_MODEL", env_default)

    async def fake_get_org_llm_settings(tenant_id, db):
        return org

    monkeypatch.setattr("llm.org_llm_settings.get_org_llm_settings", fake_get_org_llm_settings)
    node._org_llm_settings_cache = None
    await node._load_org_llm_settings("t-1")
    node._resolved_adapter_id = adapter_id
    return await node._resolve_default_model()


async def test_anthropic_with_no_model_runs_an_anthropic_model(node, monkeypatch):
    model = await _default_for(node, monkeypatch, "claude", _Org("anthropic"))
    assert model == "claude-opus-5-5"


async def test_anthropic_alias_adapter_runs_an_anthropic_model(node, monkeypatch):
    model = await _default_for(node, monkeypatch, "anthropic", _Org("claude"))
    assert model == "claude-opus-5-5"


async def test_openai_with_no_model_runs_an_openai_model(node, monkeypatch):
    model = await _default_for(node, monkeypatch, "openai", _Org("openai"))
    assert model == "gpt-5.6"


async def test_deepseek_with_no_model_runs_a_deepseek_model(node, monkeypatch):
    model = await _default_for(node, monkeypatch, "deepseek", _Org("deepseek"))
    assert model == "deepseek-chat"


async def test_ollama_on_a_vllm_deployment_runs_an_ollama_model(node, monkeypatch):
    model = await _default_for(node, monkeypatch, "ollama", _Org("ollama"))
    assert model == "llama3.2:3b"


async def test_ollama_on_an_ollama_deployment_keeps_the_deployment_model(node, monkeypatch):
    model = await _default_for(node, monkeypatch, "ollama", _Org("ollama"), env_default="llama3.1:8b-instruct-q4_K_M")
    assert model == "llama3.1:8b-instruct-q4_K_M"


async def test_vllm_on_a_vllm_deployment_keeps_the_deployment_model(node, monkeypatch):
    model = await _default_for(node, monkeypatch, "vllm", _Org("vllm"))
    assert model == VLLM_DEFAULT


async def test_vllm_with_no_vllm_model_anywhere_is_refused_with_a_reason(node, monkeypatch):
    with pytest.raises(ValueError, match="vllm"):
        await _default_for(node, monkeypatch, "vllm", _Org("vllm"), env_default="llama3.2:3b")


async def test_gemini_through_the_llm_service_bridge_keeps_the_deployment_model(node, monkeypatch):
    """Gemini and Vertex are reached through llm-service, which serves the
    deployment's own model selection; the step's model is not the cloud's choice."""
    model = await _default_for(node, monkeypatch, "llm-service", _Org("gemini"))
    assert model == VLLM_DEFAULT


async def test_the_org_model_is_kept_when_it_fits_the_provider(node, monkeypatch):
    model = await _default_for(node, monkeypatch, "claude", _Org("anthropic", "claude-sonnet-4-6"))
    assert model == "claude-sonnet-4-6"


async def test_the_deployment_model_is_kept_when_it_fits_the_cloud_provider(node, monkeypatch):
    model = await _default_for(node, monkeypatch, "openai", _Org("openai"), env_default="gpt-4o")
    assert model == "gpt-4o"


async def test_a_step_with_its_own_model_is_never_changed(node, monkeypatch):
    """_call_adapter fills the model only when the step names none."""
    from adapters.models import AdapterCapability, AdapterResponse

    class _Adapter:
        def __init__(self):
            self.requests = []

        def get_capabilities(self):
            return [AdapterCapability(name="generate", description="generate")]

        async def execute(self, request):
            self.requests.append(request)
            return AdapterResponse(request_id=request.id, capability=request.capability,
                                   status="success", data={"text": "ok"}, duration_ms=0.0)

    monkeypatch.setenv("DEFAULT_LLM_MODEL", VLLM_DEFAULT)
    node._resolved_adapter_id = "claude"
    adapter = _Adapter()
    await node._call_adapter(adapter, "generate", {"prompt": "hi", "model": "claude-haiku-4-5"})
    assert adapter.requests[-1].parameters["model"] == "claude-haiku-4-5"


async def test_execute_records_the_adapter_the_step_resolved(monkeypatch):
    """The org's preferred provider picks the adapter; the default model then follows it."""
    from adapters.base_adapter import BaseAdapter
    from adapters.models import AdapterCapability, AdapterCategory, AdapterResponse
    from adapters.registry import adapter_registry

    recorded = []

    class _ClaudeShaped(BaseAdapter):
        async def initialize(self):
            return None

        async def shutdown(self):
            return None

        def get_capabilities(self):
            return [AdapterCapability(name="generate", description="generate")]

        async def execute(self, request):
            recorded.append(request)
            return AdapterResponse(request_id=request.id, capability=request.capability,
                                   status="success", data={"text": "ok"}, duration_ms=0.0)

    monkeypatch.delenv("LLM_SCRIPTED_MODE", raising=False)
    monkeypatch.setenv("DEFAULT_LLM_MODEL", VLLM_DEFAULT)
    previous = adapter_registry.get_adapter_class("claude")
    adapter_registry.register_adapter_class("claude", _ClaudeShaped, AdapterCategory.AI)
    try:
        node = AIProcessNode(NodeConfig(id="n", name="ai", type=NodeType.TASK,
                                        parameters={"ai_task": "generate", "prompt": "hi"}))

        async def org_adapter(tenant_id):
            return "claude"

        async def no_trial_change(adapter_id, tenant_id):
            return adapter_id

        async def no_limits(tenant_id):
            return None

        async def no_credentials(adapter_id, tenant_id, user_id=None):
            return {}

        async def org_settings(tenant_id):
            node._org_llm_settings_cache = _Org("anthropic")
            return node._org_llm_settings_cache

        monkeypatch.setattr(node, "_resolve_org_adapter", org_adapter)
        monkeypatch.setattr(node, "_enforce_trial_restriction", no_trial_change)
        monkeypatch.setattr(node, "_check_llm_limits", no_limits)
        monkeypatch.setattr(node, "_load_org_llm_settings", org_settings)
        monkeypatch.setattr(
            "nodes.implementations.ai_process_node.get_adapter_credentials_for_tenant", no_credentials
        )

        await node.execute({}, {"tenant_id": "t-1"})
    finally:
        if previous is not None:
            adapter_registry.register_adapter_class("claude", previous, AdapterCategory.AI)

    assert recorded and recorded[0].parameters["model"] == "claude-opus-5-5"
