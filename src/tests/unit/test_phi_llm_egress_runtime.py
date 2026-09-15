"""R-04 runtime behaviour: a provider adapter cannot be BUILT for a provider
that may not receive PHI, and no layer between the adapter and the workflow
converts that refusal into a silent skip.

covers: r04-llm-phi-guard

Companion to tests/integration/regressions/test_phi_llm_egress_guard.py (the
host-side spec with the boot guard and the ledgers). These need the real
adapter classes and the generation engine, so they run in the Community
container under `make test-unit-community`.

The test egress guard is default-on here: a provider that IS reached raises
SmokeEgressBlocked, which is the WRONG exception, so every refusal test
asserts PHIEgressRefused specifically.
"""

import socket
from unittest.mock import AsyncMock

import pytest

from adapters.models import AdapterCategory, AdapterConfig
from core import phi_egress
from core.phi_egress import PHI_REFUSAL_MARKER, PHIEgressRefused


@pytest.fixture(autouse=True)
def _phi_env(monkeypatch):
    """PHI mode on with only vLLM allowlisted, unless a test overrides."""
    monkeypatch.setenv("AICTRLNET_PHI_MODE", "true")
    monkeypatch.setenv("AICTRLNET_PHI_LLM_PROVIDERS", "vllm")
    monkeypatch.delenv("AICTRLNET_PHI_BAA_PROVIDERS", raising=False)
    phi_egress.reset_endpoint_cache()
    yield
    phi_egress.reset_endpoint_cache()


def _cfg(name, **kw):
    return AdapterConfig(name=name, version="1.0.0", category=AdapterCategory.AI, **kw)


def _addrinfo(*ips):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0)) for ip in ips]


# --- Construction is the choke point -----------------------------------------


def test_cloud_adapter_cannot_be_built_when_not_allowlisted():
    from adapters.implementations.ai.openai_adapter import OpenAIAdapter

    with pytest.raises(PHIEgressRefused) as excinfo:
        OpenAIAdapter(_cfg("openai", api_key="sk-test"))
    message = str(excinfo.value)
    assert message.startswith(PHI_REFUSAL_MARKER)
    assert "openai" in message and "AICTRLNET_PHI_LLM_PROVIDERS" in message


def test_refusal_wins_over_missing_credentials(monkeypatch):
    """OpenAIAdapter.__init__ ends by raising on a missing key. The PHI check
    must come first, or a practice machine with no cloud key would report
    'API key is required' instead of the refusal."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from adapters.implementations.ai.openai_adapter import OpenAIAdapter

    with pytest.raises(PHIEgressRefused):
        OpenAIAdapter(_cfg("openai"))


def test_claude_alias_normalises_to_anthropic(monkeypatch):
    monkeypatch.setenv("AICTRLNET_PHI_LLM_PROVIDERS", "claude")
    monkeypatch.setenv("AICTRLNET_PHI_BAA_PROVIDERS", "anthropic")
    from adapters.implementations.ai.claude_adapter import ClaudeAdapter

    ClaudeAdapter(_cfg("claude", api_key="sk-ant-test"))


def test_allowlisted_local_adapter_builds_at_private_address():
    from adapters.implementations.ai.vllm_adapter import VLLMAdapter

    adapter = VLLMAdapter(_cfg("vllm", base_url="http://10.0.0.5:8000"))
    assert adapter.base_url == "http://10.0.0.5:8000/v1"


def test_allowlisted_local_adapter_refused_at_public_host(monkeypatch):
    """'vllm' means on-machine. A vLLM adapter aimed at a hosted endpoint is a
    cloud provider wearing a local name, and per-tenant adapter_configs rows
    can set base_url, so this is checked at construction, not only at boot."""
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: _addrinfo("104.18.0.1"))
    from adapters.implementations.ai.vllm_adapter import VLLMAdapter

    with pytest.raises(PHIEgressRefused) as excinfo:
        VLLMAdapter(_cfg("vllm", base_url="https://vllm.example.com"))
    assert "vllm.example.com" in str(excinfo.value)


def test_ollama_not_allowlisted_is_refused_even_though_local():
    from adapters.implementations.ai.ollama_adapter import OllamaAdapter

    with pytest.raises(PHIEgressRefused, match="ollama"):
        OllamaAdapter(_cfg("ollama", base_url="http://localhost:11434"))


def test_phi_mode_off_changes_nothing(monkeypatch):
    monkeypatch.setenv("AICTRLNET_PHI_MODE", "false")
    from adapters.implementations.ai.openai_adapter import OpenAIAdapter

    OpenAIAdapter(_cfg("openai", api_key="sk-test"))


def test_provider_adapters_declare_static_providers():
    from adapters.implementations.ai.claude_adapter import ClaudeAdapter
    from adapters.implementations.ai.llm_service_adapter import LLMServiceAdapter
    from adapters.implementations.ai.vllm_adapter import VLLMAdapter

    assert ClaudeAdapter.PHI_PROVIDERS == frozenset({"anthropic"})
    assert VLLMAdapter.PHI_PROVIDERS == frozenset({"vllm"})
    assert LLMServiceAdapter.PHI_PROVIDERS == frozenset()


# --- llm/service.py must propagate, not return None ---------------------------


async def test_adapter_provider_get_propagates_refusal(monkeypatch):
    """_AdapterProvider.get wraps construction and initialize() in a
    swallow-everything handler that returns None, which the engine reads as
    'provider unavailable' and then falls back. A refusal must escape."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    from llm.models import ModelProvider
    from llm.service import _AdapterProvider

    _AdapterProvider.reset()
    try:
        with pytest.raises(PHIEgressRefused):
            await _AdapterProvider.get(ModelProvider.ANTHROPIC)
    finally:
        _AdapterProvider.reset()


def test_create_adapter_propagates_refusal(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    from llm.models import ModelProvider
    from llm.service import _AdapterProvider

    with pytest.raises(PHIEgressRefused):
        _AdapterProvider._create_adapter(ModelProvider.OPENAI)


# --- The llm-service bridge and the raw Ollama paths -------------------------


async def test_llm_service_bridge_execute_propagates_refusal(monkeypatch):
    """execute() converts every exception into an error RESULT, and
    ai_process_node retries an error result as chat — a second egress attempt.
    _generate's direct-call failure also falls back to an HTTP hop."""
    from adapters.implementations.ai.llm_service_adapter import LLMServiceAdapter
    from adapters.models import AdapterRequest

    adapter = LLMServiceAdapter(_cfg("llm-service"))
    adapter.initialized = True
    adapter._initialized = True

    async def refuse(*a, **k):
        raise PHIEgressRefused(f"{PHI_REFUSAL_MARKER} no")

    import llm.service as llm_service_module

    monkeypatch.setattr(llm_service_module.llm_service, "generate", refuse)
    monkeypatch.setattr(
        type(adapter), "client", property(lambda self: (_ for _ in ()).throw(AssertionError("HTTP hop reached")))
    )
    with pytest.raises(PHIEgressRefused):
        await adapter.execute(AdapterRequest(capability="generate", parameters={"prompt": "note"}))


def test_llm_service_bridge_refuses_non_local_service_url(monkeypatch):
    """_chat and _embedding have no in-process path: they POST the prompt to
    LLM_SERVICE_INTERNAL_URL. Under PHI mode that must be a local address."""
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: _addrinfo("104.18.0.1"))
    monkeypatch.setenv("LLM_SERVICE_INTERNAL_URL", "https://llm.example.com")
    from adapters.implementations.ai.llm_service_adapter import LLMServiceAdapter

    with pytest.raises(PHIEgressRefused):
        LLMServiceAdapter(_cfg("llm-service"))


def test_llm_service_bridge_builds_at_default_local_url(monkeypatch):
    monkeypatch.delenv("LLM_SERVICE_INTERNAL_URL", raising=False)
    from adapters.implementations.ai.llm_service_adapter import LLMServiceAdapter

    LLMServiceAdapter(_cfg("llm-service"))


def test_model_adapters_factory_refuses_ollama_when_not_allowlisted():
    from services.model_adapters import get_model_adapter

    with pytest.raises(PHIEgressRefused, match="ollama"):
        get_model_adapter("llama3.1:8b", "http://host.docker.internal:11434")


async def test_generation_raw_ollama_post_refused(monkeypatch):
    from llm.generation import LLMGenerationEngine
    from llm.models import LLMRequest

    engine = LLMGenerationEngine()
    with pytest.raises(PHIEgressRefused, match="ollama"):
        await engine._generate_with_ollama(LLMRequest(prompt="note", task_type="general"), "llama3.1:8b")


async def test_call_adapter_chat_retry_does_not_mask_refusal():
    from nodes.implementations.ai_process_node import AIProcessNode
    from nodes.models import NodeConfig, NodeType

    class _RefusingAdapter:
        def get_capabilities(self):
            return []

        async def execute(self, request):
            raise PHIEgressRefused(f"{PHI_REFUSAL_MARKER} no")

    node = AIProcessNode(NodeConfig(id="n", name="ai", type=NodeType.TASK, parameters={}))
    with pytest.raises(PHIEgressRefused):
        await node._call_adapter(_RefusingAdapter(), "generate", {"prompt": "x", "model": "m"})


# --- Fallback chain, auto-selection, model listing ---------------------------


async def test_generate_refuses_before_any_attempt_and_never_falls_back(monkeypatch):
    """A refusal is decided before the primary provider is touched, and the
    local-fallback branch is never entered on it."""
    from llm.generation import LLMGenerationEngine
    from llm.models import LLMRequest

    monkeypatch.setenv("DEFAULT_LLM_MODEL", "claude-3-haiku-20240307")
    engine = LLMGenerationEngine()
    monkeypatch.setattr(engine, "_is_cloud_environment", lambda: False)
    attempted = AsyncMock(side_effect=AssertionError("provider attempted"))
    picker = AsyncMock(side_effect=AssertionError("fallback consulted"))
    monkeypatch.setattr(engine, "_generate_with_adapter", attempted)
    monkeypatch.setattr(engine, "_pick_fallback_local_model", picker)

    with pytest.raises(PHIEgressRefused, match="anthropic"):
        await engine.generate(LLMRequest(prompt="note", task_type="general"))
    assert not attempted.called and not picker.called


async def test_fallback_picker_never_crosses_to_a_non_allowlisted_provider(monkeypatch):
    from llm.generation import LLMGenerationEngine

    monkeypatch.setenv("DEFAULT_LLM_MODEL", "gemini-2.0-flash-vertex")
    engine = LLMGenerationEngine()
    monkeypatch.setattr(engine, "_get_ollama_models", AsyncMock(return_value=["gemma3:27b"]))
    monkeypatch.setattr(engine, "_get_vllm_models", AsyncMock(return_value=[]))
    assert await engine._pick_fallback_local_model() is None

    monkeypatch.setattr(engine, "_get_vllm_models", AsyncMock(return_value=["gemma-4-26b"]))
    assert await engine._pick_fallback_local_model() == "vllm:gemma-4-26b"


async def test_fallback_picker_skips_a_non_allowlisted_configured_default(monkeypatch):
    from llm.generation import LLMGenerationEngine

    monkeypatch.setenv("DEFAULT_LLM_MODEL", "gemma3:27b")  # ollama, not allowed
    engine = LLMGenerationEngine()
    monkeypatch.setattr(engine, "_get_ollama_models", AsyncMock(return_value=["gemma3:27b"]))
    monkeypatch.setattr(engine, "_get_vllm_models", AsyncMock(return_value=["gemma-4-26b"]))
    assert await engine._pick_fallback_local_model() == "vllm:gemma-4-26b"


def _register(name, cls):
    from adapters.registry import adapter_registry

    adapter_registry.register_adapter_class(name, cls, AdapterCategory.AI, description=name)


async def test_auto_select_skips_non_allowlisted_classes(monkeypatch):
    from adapters.implementations.ai.llm_service_adapter import LLMServiceAdapter
    from adapters.implementations.ai.ollama_adapter import OllamaAdapter
    from adapters.implementations.ai.openai_adapter import OpenAIAdapter
    from nodes.implementations.ai_process_node import AIProcessNode
    from nodes.models import NodeConfig, NodeType

    monkeypatch.setenv("DEFAULT_LLM_MODEL", "llama3.1:8b")  # -> ollama, not allowed
    _register("ollama", OllamaAdapter)
    _register("openai", OpenAIAdapter)
    _register("llm-service", LLMServiceAdapter)
    node = AIProcessNode(NodeConfig(id="n", name="ai", type=NodeType.TASK, parameters={}))
    # The bridge declares no providers and is guarded transitively.
    assert await node._auto_select_adapter("generate") == "llm-service"


async def test_auto_select_refuses_rather_than_picking_any_adapter(monkeypatch):
    """Pre-fix the last resort was `available[0]`: whatever happened to be
    registered first, which under PHI mode is a cloud provider as often as not."""
    from adapters.implementations.ai.ollama_adapter import OllamaAdapter
    from adapters.implementations.ai.openai_adapter import OpenAIAdapter
    from nodes.implementations.ai_process_node import AIProcessNode
    from nodes.models import NodeConfig, NodeType

    monkeypatch.setenv("DEFAULT_LLM_MODEL", "llama3.1:8b")
    _register("openai", OpenAIAdapter)
    _register("ollama", OllamaAdapter)
    node = AIProcessNode(NodeConfig(id="n", name="ai", type=NodeType.TASK, parameters={}))
    with pytest.raises(PHIEgressRefused, match="AICTRLNET_PHI_LLM_PROVIDERS"):
        await node._auto_select_adapter("generate")


async def test_model_listing_hides_providers_that_would_refuse(monkeypatch):
    from llm.generation import LLMGenerationEngine
    from llm.models import ModelProvider

    engine = LLMGenerationEngine()
    monkeypatch.setattr(engine, "_get_ollama_models", AsyncMock(return_value=[]))
    monkeypatch.setattr(engine, "_get_vllm_models", AsyncMock(return_value=[]))
    assert await engine.get_available_models() == []

    monkeypatch.setenv("AICTRLNET_PHI_LLM_PROVIDERS", "vllm,openai")
    monkeypatch.setenv("AICTRLNET_PHI_BAA_PROVIDERS", "openai")
    providers = {m.provider for m in await engine.get_available_models()}
    assert providers == {ModelProvider.OPENAI}


async def test_care_gap_shaped_node_fails_with_the_refusal_not_a_silent_skip(monkeypatch):
    """The R-04 exposure end to end: an aiProcess node with no adapter set,
    PHI mode on, nothing allowlisted. Through BaseNode.run(), the way the
    loop node drives it, the result must be FAILED carrying the marker, and
    no adapter may have been executed."""
    from adapters.implementations.ai.openai_adapter import OpenAIAdapter
    from nodes.implementations.ai_process_node import AIProcessNode
    from nodes.models import NodeConfig, NodeInstance, NodeStatus, NodeType

    monkeypatch.setenv("AICTRLNET_PHI_LLM_PROVIDERS", "")
    monkeypatch.setenv("DEFAULT_LLM_MODEL", "gpt-4o")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    executed = []
    monkeypatch.setattr(
        OpenAIAdapter, "execute", AsyncMock(side_effect=lambda r: executed.append(r))
    )
    _register("openai", OpenAIAdapter)

    config = NodeConfig(
        id="classify-one-note",
        name="Classify One Note Against One Gap",
        type=NodeType.TASK,
        parameters={
            "ai_task": "generate",
            "prompt": "GAP: {{item.measure_key}}\nNOTE: {{item.note}}",
        },
    )
    node = AIProcessNode(config)
    instance = NodeInstance(
        node_config=config,
        workflow_instance_id="wf-care-gap",
        input_data={"item": {"measure_key": "a1c", "note": "HbA1c 7.9 on 2026-09-01"}},
    )
    result = await node.run(instance, workflow_variables={})

    assert result.status == NodeStatus.FAILED, result
    assert PHI_REFUSAL_MARKER in (result.error or ""), result.error
    assert executed == []


async def test_auto_select_refuses_outright_when_nothing_is_allowlisted(monkeypatch):
    """Found by the manual template run: with an empty allowlist the bridge
    (no declared providers) was still eligible, and the node failed on the
    bridge's health check with 'All connection attempts failed' instead of
    the refusal. Nothing can serve, so say so."""
    from adapters.implementations.ai.llm_service_adapter import LLMServiceAdapter
    from adapters.implementations.ai.ollama_adapter import OllamaAdapter
    from nodes.implementations.ai_process_node import AIProcessNode
    from nodes.models import NodeConfig, NodeType

    monkeypatch.setenv("AICTRLNET_PHI_LLM_PROVIDERS", "")
    monkeypatch.setenv("DEFAULT_LLM_MODEL", "llama3.1:8b")
    _register("ollama", OllamaAdapter)
    _register("llm-service", LLMServiceAdapter)
    node = AIProcessNode(NodeConfig(id="n", name="ai", type=NodeType.TASK, parameters={}))
    with pytest.raises(PHIEgressRefused, match="AICTRLNET_PHI_LLM_PROVIDERS"):
        await node._auto_select_adapter("generate")
