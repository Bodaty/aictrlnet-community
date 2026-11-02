"""The trial restriction never replaces a model on the deployment's own machine.

A tenant with no LLM settings is in trial mode, and trial mode swaps a node's
adapter for the system default so a trial can't spend on a paid provider. A
local provider (Ollama, vLLM, llama.cpp) costs nothing to call, and a node
that names one usually needs that one: the Documents OCR step reads page
images through the llama.cpp adapter, and swapping it for the chat default
failed every page read on a trial-mode practice.
"""
import pytest

from llm.org_llm_settings import OrgLLMSettings
from nodes.implementations.ai_process_node import AIProcessNode
from nodes.models import NodeConfig, NodeType


@pytest.fixture
def node():
    return AIProcessNode(NodeConfig(id="n", name="ai", type=NodeType.TASK, parameters={}))


@pytest.fixture
def trial_tenant(monkeypatch):
    async def settings(tenant_id, db):
        return OrgLLMSettings()  # the default: trial mode, no own key

    monkeypatch.setattr("llm.org_llm_settings.get_org_llm_settings", settings)


@pytest.fixture
def system_default(node, monkeypatch):
    async def auto_select(task):
        return "ollama"

    monkeypatch.setattr(node, "_auto_select_adapter", auto_select)


@pytest.mark.parametrize("adapter", ["llamacpp", "vllm", "ollama"])
async def test_a_local_adapter_is_kept_in_trial_mode(node, trial_tenant, system_default, adapter):
    assert await node._enforce_trial_restriction(adapter, "tenant-a") == adapter


async def test_a_paid_adapter_is_still_replaced_in_trial_mode(node, trial_tenant, system_default):
    assert await node._enforce_trial_restriction("openai", "tenant-a") == "ollama"
