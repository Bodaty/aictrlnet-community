"""Regression: descriptive template nodes degrade gracefully, never crash.

covers: bug2-platformintegration bug2-agentdelegation bug2-datasource

The Bug 2 QA sweep found that auto-generated system templates carry *descriptive
placeholder* nodes (prompt/analysis config, no operational params) typed
platformIntegration / agentDelegation / dataSource. Once preserve_types made
aiProcess nodes execute live, these stopped passing through and crashed entire
workflows:
  - PlatformIntegrationNode.__init__ took (node_id, config) but the registry
    constructs node_class(config) -> TypeError.
  - IAMNode send_message raised "target_agent is required".
  - DataSourceNode static source raised "data parameter is required".

Post-fix, an unconfigured node of each type completes with an honest, non-crash
output instead of raising. These tests construct each node directly with a
descriptive config and assert the graceful result.
"""
from unittest.mock import MagicMock

import pytest

from nodes.models import NodeConfig, NodeType
from nodes.implementations.platform_integration_node import PlatformIntegrationNode
from nodes.implementations.iam_node import IAMNode
from nodes.implementations.data_source_node import DataSourceNode


def _cfg(parameters):
    return NodeConfig(id="n1", name="descriptive", type=NodeType.TASK, parameters=parameters)


@pytest.mark.asyncio
async def test_platform_integration_descriptive_node_skips():
    # Descriptive config: free-text platform, no workflow_id/credential_id.
    node = PlatformIntegrationNode(_cfg({
        "platform": "Multi-Channel Communications System",
        "description": "Send personalized collection comms",
        "integration_capabilities": ["template_management"],
    }))
    out = await node.execute({}, {})
    assert out["status"] == "skipped"
    assert out["executed"] is False
    assert "no platform integration configured" in out["reason"]


@pytest.mark.asyncio
async def test_agent_delegation_without_target_skips():
    # agentDelegation -> IAMNode, default operation send_message, no target_agent.
    node = IAMNode(_cfg({
        "agent": "Collections Coordinator",
        "prompt": "Coordinate with the collections team",
    }))
    out = await node.execute({}, {"db": MagicMock()})
    assert out.get("status") == "skipped"
    assert out.get("executed") is False
    assert "target_agent" in out.get("reason", "")


@pytest.mark.asyncio
async def test_data_source_static_without_data_yields_empty():
    # Static dataSource with no inline `data` must not raise.
    node = DataSourceNode(_cfg({"source_type": "static"}))
    out = await node.execute({}, {})
    assert out.get("data") == {}
    assert out.get("source_type") == "static"


@pytest.mark.asyncio
async def test_data_source_default_source_type_no_data_yields_empty():
    # No source_type at all (defaults to static) + no data -> graceful empty.
    node = DataSourceNode(_cfg({}))
    out = await node.execute({}, {})
    assert out.get("data") == {}
