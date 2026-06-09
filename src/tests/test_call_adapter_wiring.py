"""Regression: BaseNode.call_adapter wires a real, credentialed AdapterConfig.

covers: geo-phase-b1 call-adapter-credentials

Pre-fix, call_adapter instantiated `adapter_class({})` — an empty dict, not an
AdapterConfig — and never started the adapter, so a credentialed engine could
neither construct nor authenticate (the adapter-node path was effectively dead).
Post-fix it resolves credentials, builds a real AdapterConfig, and runs the full
lifecycle. This test registers a capturing adapter and asserts the config it
receives is a real AdapterConfig carrying the resolved credentials.
"""
from unittest.mock import AsyncMock, patch

import pytest

from nodes.base_node import BaseNode
from nodes.models import NodeConfig, NodeType
from adapters.registry import adapter_registry
from adapters.base_adapter import BaseAdapter
from adapters.models import (
    AdapterConfig, AdapterCapability, AdapterResponse, AdapterCategory,
)

_captured = {}


class CapturingAdapter(BaseAdapter):
    def __init__(self, config):
        _captured["config"] = config
        super().__init__(config)

    async def initialize(self):
        pass

    async def shutdown(self):
        pass

    def get_capabilities(self):
        return [AdapterCapability(name="ping", description="ping")]

    async def execute(self, request):
        return AdapterResponse(
            request_id=request.id, capability=request.capability,
            status="success", data={"ok": True}, duration_ms=1.0,
        )


class _Node(BaseNode):
    async def execute(self, input_data, context):
        return {}


@pytest.mark.asyncio
async def test_call_adapter_builds_real_credentialed_config():
    adapter_registry.register_adapter_class("capturing-test", CapturingAdapter, AdapterCategory.AI)
    node = _Node(NodeConfig(id="n", name="n", type=NodeType.TASK, parameters={"timeout": 123}))

    with patch(
        "nodes.template_utils.get_adapter_credentials_for_tenant",
        new=AsyncMock(return_value={"api_key": "abc", "extra": 1}),
    ):
        data = await node.call_adapter("capturing-test", "ping", {"q": 1}, context={"tenant_id": "org-x"})

    cfg = _captured["config"]
    assert isinstance(cfg, AdapterConfig)            # not the old {}
    assert cfg.name == "capturing-test"
    assert cfg.category == AdapterCategory.AI
    assert cfg.api_key == "abc"                       # credentials resolved + wired
    assert cfg.credentials == {"api_key": "abc", "extra": 1}
    assert cfg.timeout_seconds == 123                 # node timeout propagated
    assert data["ok"] is True


@pytest.mark.asyncio
async def test_call_adapter_unknown_adapter_raises():
    node = _Node(NodeConfig(id="n", name="n", type=NodeType.TASK, parameters={}))
    with pytest.raises(ValueError):
        await node.call_adapter("does-not-exist", "ping", {})
