"""A resumed run follows the branch its decision node chose (G21)."""
import sys as _sys, pathlib as _pathlib
_sys.path.append(str(_pathlib.Path(__file__).resolve().parent))

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nodes.executor import NodeExecutor
from nodes.models import NodeConfig, NodeInstance, NodeStatus, NodeType, WorkflowInstance
from nodes.registry import node_registry


def _instance(node_id, node_type, status=NodeStatus.PENDING, next_nodes=(), parameters=None,
              output_data=None):
    return NodeInstance(
        id=node_id,
        node_config=NodeConfig(id=node_id, name=node_id, type=node_type, parameters=parameters or {}),
        workflow_instance_id="wf-1",
        status=status,
        next_nodes=list(next_nodes),
        input_data={"doc": "x"},
        output_data=output_data or {},
    )


def _resumed_workflow(selected_branch, parameters):
    decision = _instance(
        "decide", NodeType.DECISION, status=NodeStatus.COMPLETED, next_nodes=["A", "B"],
        parameters=parameters, output_data={"selected_branch": selected_branch},
    )
    return WorkflowInstance(
        id="wf-1", template_id="t", name="resume",
        node_instances={
            "decide": decision,
            "A": _instance("A", NodeType.TASK),
            "B": _instance("B", NodeType.TASK),
        },
    )


async def _resume(workflow):
    executor = NodeExecutor()
    ran = []

    async def fake_execute_node(node_instance, workflow_instance):
        ran.append(node_instance.id)
        return SimpleNamespace(status=NodeStatus.COMPLETED, next_node_ids=[])

    executor.execute_node = fake_execute_node
    await executor._execute_from_nodes(workflow, ["decide"], {})
    return ran


@pytest.mark.asyncio
async def test_rehydrated_decision_node_queues_only_selected_branch():
    workflow = _resumed_workflow("b", {"branch_mapping": {"a": ["A"], "b": ["B"]}})
    assert await _resume(workflow) == ["B"]


@pytest.mark.asyncio
async def test_rehydrated_decision_node_without_routing_config_queues_all_next_nodes():
    workflow = _resumed_workflow("default", {})
    assert sorted(await _resume(workflow)) == ["A", "B"]


@pytest.mark.asyncio
async def test_unmatched_branch_without_default_raises():
    instance = _instance(
        "decide", NodeType.DECISION, next_nodes=["A", "B"],
        parameters={"branch_mapping": {"a": ["A"]}},
        output_data={"selected_branch": "zzz"},
    )
    node = node_registry.create_node(instance.node_config)
    with pytest.raises(ValueError, match="zzz"):
        await node._determine_next_nodes(instance, {})


@pytest.mark.asyncio
async def test_unmatched_branch_with_default_keeps_next_nodes():
    instance = _instance(
        "decide", NodeType.DECISION, next_nodes=["A", "B"],
        parameters={"branch_mapping": {"a": ["A"]}, "default_branch": "a"},
        output_data={"selected_branch": "zzz"},
    )
    node = node_registry.create_node(instance.node_config)
    assert await node._determine_next_nodes(instance, {}) == ["A", "B"]


@pytest.mark.asyncio
async def test_resume_does_not_instantiate_non_decision_nodes():
    unregistered = next(t for t in NodeType if t not in node_registry._node_classes)
    first = _instance(
        "first", unregistered, status=NodeStatus.COMPLETED, next_nodes=["A"],
        output_data={"ok": True},
    )
    workflow = WorkflowInstance(
        id="wf-1", template_id="t", name="resume",
        node_instances={"first": first, "A": _instance("A", NodeType.TASK)},
    )
    executor = NodeExecutor()
    ran = []

    async def fake_execute_node(node_instance, workflow_instance):
        ran.append(node_instance.id)
        return SimpleNamespace(status=NodeStatus.COMPLETED, next_node_ids=[])

    executor.execute_node = fake_execute_node
    await executor._execute_from_nodes(workflow, ["first"], {})
    assert ran == ["A"]
