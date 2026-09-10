"""Shared helpers for the Tier A execution sweep.

The sweep executes every surface once and rejects placeholder success. These
helpers mirror what the engine does at runtime (workflow_execution.py:271-309)
so the gate tests production resolution, not a test-only path.
"""
from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Type, Union
from nodes.models import NodeConfig, NodeInstance, NodeType, NodeStatus

# Enum members with NO implementation in any edition (verified 2026-09-10).
# This set is a FINDING (A-2), not an acceptance. It must shrink, never grow:
# a new dead enum value fails test_every_enum_member_resolves_or_is_declared_dead.
UNIMPLEMENTED_NODE_TYPES = frozenset({
    NodeType.JOIN, NodeType.AGGREGATE, NodeType.DATABASE, NodeType.SEND_MESSAGE,
    NodeType.WAIT_MESSAGE, NodeType.ERROR_HANDLER, NodeType.RETRY, NodeType.COMPENSATE,
})

# Enum members that resolve only through their alias string (registry.py:42-96,
# edition_nodes.py). NodeConfig(type=member) alone raises; with custom_node_type it works.
ALIAS_ONLY_NODE_TYPES = frozenset({
    NodeType.LOOP, NodeType.PARALLEL, NodeType.HUMAN_TASK, NodeType.API_CALL, NodeType.WEBHOOK,
})

COMMUNITY_ALIASES = {
    "start", "end", "decision", "task", "adapter", "transform", "approval",
    "dataSource", "aiProcess", "apiCall", "mcp", "iam", "notification", "mcpClient",
    "mcpServer", "platform", "fileProcess", "docGeneration", "browserAutomation",
    "approvalRequest", "humanAgent", "human_agent", "humanTask", "human_task",
    "process", "contextTransformation", "agentDelegation", "hybrid",
    "platformIntegration", "ai_process", "data_source", "api_call",
    "openaiAdapter", "conditionNode", "postgresqlAdapter", "emailAdapter",
}
# Exactly the tuples in editions/business/src/nodes/edition_nodes.py (14 entries).
BUSINESS_ALIASES = COMMUNITY_ALIASES | {
    "code", "parallel", "loop", "webhook", "schedule",
    "iamAdvanced", "remoteTool", "careGapEngine",
}
# Enterprise hand-copies the Business list (13 entries, missing careGapEngine - A-15) and adds 5.
ENTERPRISE_ALIASES = BUSINESS_ALIASES | {
    "enhancedAdapter", "orchestration", "orchestrationPattern", "compliance", "federation",
}
EXPECTED_ALIASES = {
    "community": COMMUNITY_ALIASES,
    "business": BUSINESS_ALIASES,
    "enterprise": ENTERPRISE_ALIASES,
}


def make_config(alias: str, **params: Any) -> NodeConfig:
    """Build a NodeConfig exactly as the engine does from a template node's raw type."""
    try:
        node_type = NodeType(alias)
    except ValueError:
        node_type = NodeType.TASK
    params.setdefault("custom_node_type", alias)
    return NodeConfig(id=f"tier-a-{alias}", name=f"tier-a {alias}", type=node_type, parameters=params)


def make_instance(config: NodeConfig, input_data: Dict[str, Any], context: Dict[str, Any]) -> NodeInstance:
    wf_instance = context.get("workflow_instance_id") or str(uuid.uuid4())
    return NodeInstance(node_config=config, workflow_instance_id=wf_instance,
                        input_data=input_data, context=context)


# Output shapes that mean "nothing happened" - accepted only where a battery row
# names them explicitly via allow=...
PLACEHOLDER_MARKERS = (
    ("status", "skipped"),
    ("status", "timeout"),
    ("status", "feature_pending"),
    ("_dry_run", True),
    ("note", "no static data configured"),
)


def assert_real(result, *, required_keys=(), forbid_keys=("error",), allow=()):
    """A node result is real when it COMPLETED, produced a non-empty dict, carries
    no truthy error key, and matches no placeholder marker except those allowed."""
    assert result.status == NodeStatus.COMPLETED, f"status={result.status} error={result.error}"
    out = result.output_data
    assert isinstance(out, dict) and out, f"empty output: {out!r}"
    for k in forbid_keys:
        assert not out.get(k), f"output carries {k}={out.get(k)!r}"
    for k in required_keys:
        assert k in out, f"missing {k}; keys={sorted(out)}"
    for key, val in PLACEHOLDER_MARKERS:
        if (key, val) in allow:
            continue
        assert out.get(key) != val, f"placeholder success: {key}={val!r} - output={out!r}"


@dataclass
class Row:
    """One battery row: the CORRECT expectation for a node, not today's behaviour."""
    alias: str
    params: Dict[str, Any] = field(default_factory=dict)
    input: Dict[str, Any] = field(default_factory=dict)
    required_keys: Tuple[str, ...] = ()
    allow: Tuple[Tuple[str, Any], ...] = ()
    raises: Optional[Union[Type[BaseException], Tuple[Type[BaseException], ...]]] = None
    match: str = ""
    note: str = ""


def _ctx(db) -> Dict[str, Any]:
    """The context keys the engine populates (workflow_execution.py:~324-347)."""
    wf = str(uuid.uuid4())
    return {"db": db, "workflow_id": wf, "workflow_instance_id": str(uuid.uuid4()),
            "workflow_definition_id": wf, "workflow_name": "tier-a", "user_id": "tier-a-user",
            "tenant_id": "default-tenant", "is_dry_run": False}
