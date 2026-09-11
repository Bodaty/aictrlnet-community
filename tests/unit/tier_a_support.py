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


_TIER_A_REGISTERED = False


def ensure_edition_nodes_registered():
    """Order-independent registration for the Tier A node gates, matching the
    real app.

    The node_registry singleton (registry.py) is built by whichever test imports
    it first; under the full suite that can leave only Community defaults, so
    edition aliases go missing though they pass in isolation. This mirrors the
    app lifespan's defensive repair (aictrlnet_enterprise/core/app.py:186-203):
    load the edition's OWN edition_nodes.py by file path and register it.

    - Runs at most once per process, and only if registration was actually lost
      (so in the normal case where registry.py's import hook already registered,
      this is a pure no-op and pollutes nothing).
    - Loads the CURRENT edition's edition_nodes.py only. In the enterprise
      container that is enterprise/edition_nodes.py, which registers the business
      node list it carries - NOT business/edition_nodes.py - so careGapEngine
      stays unregistered in Enterprise exactly as the shipped app leaves it (A-15).
    """
    global _TIER_A_REGISTERED
    if _TIER_A_REGISTERED:
        return
    import importlib.util
    import os
    from nodes.models import NodeConfig, NodeType
    from nodes.registry import node_registry

    def _resolves(alias):
        try:
            cfg = NodeConfig(id="probe", name="probe", type=NodeType.TASK, parameters={"custom_node_type": alias})
            node = node_registry.create_node(cfg)
            return type(node).__name__ != "TaskNode"
        except Exception:  # noqa: BLE001
            return False

    # Enterprise container: prefer enterprise's own file (registers its business
    # list). Business container: business file. Community: neither exists.
    candidates = [
        "/workspace/editions/enterprise/src/nodes/edition_nodes.py",
        "/workspace/editions/business/src/nodes/edition_nodes.py",
    ]
    if not _resolves("code"):  # a business/enterprise alias missing => registration lost
        for path in candidates:
            if not os.path.exists(path):
                continue
            try:
                spec = importlib.util.spec_from_file_location(f"_tier_a_edition_nodes_{abs(hash(path))}", path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                mod.register_edition_nodes(node_registry)
            except Exception:  # noqa: BLE001 - the gate reports what stays unregistered
                pass
            break  # the first existing candidate is this edition's own file
    _TIER_A_REGISTERED = True
