"""A step skipped by its skip_condition lets the run continue (G29).

A definition's `skip_condition` must reach the step, and a skipped step must
hand its input on to the steps after it, the way a completed step does.
"""
import sys as _sys, pathlib as _pathlib
_sys.path.append(str(_pathlib.Path(__file__).resolve().parent))

from types import SimpleNamespace

import pytest

from nodes.executor import NodeExecutor
from nodes.models import NodeConfig, NodeInstance, NodeStatus, NodeType, WorkflowInstance
from services.workflow_execution import node_config_from_definition


def _instance(node_id, next_nodes=(), skip_condition=None):
    return NodeInstance(
        id=node_id,
        node_config=NodeConfig(id=node_id, name=node_id, type=NodeType.TASK,
                               skip_condition=skip_condition),
        workflow_instance_id="wf-1",
        next_nodes=list(next_nodes),
    )


@pytest.mark.asyncio
async def test_a_skipped_step_passes_its_input_to_the_next_step():
    workflow = WorkflowInstance(
        id="wf-1", template_id="t", name="skip",
        node_instances={
            "first": _instance("first", ["skipped"]),
            "skipped": _instance("skipped", ["last"], skip_condition="skip_me"),
            "last": _instance("last"),
        },
    )
    executor = NodeExecutor()
    seen = {}

    async def fake_execute_node(node_instance, workflow_instance):
        seen[node_instance.id] = dict(node_instance.input_data)
        if node_instance.id == "first":
            node_instance.output_data = {"skip_me": True, "carried": 1}
            return SimpleNamespace(status=NodeStatus.COMPLETED, next_node_ids=["skipped"])
        if node_instance.id == "skipped":
            return SimpleNamespace(status=NodeStatus.SKIPPED, next_node_ids=["last"])
        return SimpleNamespace(status=NodeStatus.COMPLETED, next_node_ids=[])

    executor.execute_node = fake_execute_node
    await executor._execute_from_nodes(workflow, ["first"], {"document_id": "d-1"})
    assert list(seen) == ["first", "skipped", "last"]
    assert seen["last"] == {"document_id": "d-1", "skip_me": True, "carried": 1}


@pytest.mark.asyncio
async def test_the_node_skips_on_its_condition():
    from nodes.registry import node_registry

    instance = _instance("skipped", ["last"], skip_condition="pages > 0 or not extract")
    instance.input_data = {"pages": 0, "extract": False}
    node = node_registry.create_node(instance.node_config)
    result = await node.run(instance, {})
    assert result.status == NodeStatus.SKIPPED and result.next_node_ids == ["last"]


def test_the_definition_skip_condition_reaches_the_step():
    config = node_config_from_definition({
        "id": "extract", "type": "aiProcess", "name": "Extract",
        "parameters": {"skip_condition": "unreadable_pages > 0", "timeout_seconds": 300},
    })
    assert config.skip_condition == "unreadable_pages > 0"
    assert config.timeout_seconds == 300
    assert config.parameters["custom_node_type"] == "aiProcess"


def test_a_step_without_a_skip_condition_has_none():
    config = node_config_from_definition({"id": "s", "type": "task", "parameters": {}})
    assert config.skip_condition is None and config.timeout_seconds == 300


@pytest.mark.parametrize("node_type", [
    "decision", "conditionNode", "approval", "approvalRequest", "humanAgent", "human_task",
])
def test_a_decision_or_approval_step_cannot_be_skipped(node_type):
    with pytest.raises(ValueError, match="never skipped"):
        node_config_from_definition({
            "id": "gate", "type": node_type, "parameters": {"skip_condition": "true"},
        })


def test_a_decision_or_approval_step_without_a_skip_condition_is_fine():
    assert node_config_from_definition({"id": "gate", "type": "approval", "parameters": {}}).skip_condition is None


@pytest.mark.asyncio
async def test_a_resumed_run_re_evaluates_a_skipped_step_and_finishes():
    """Resume rehydrates only completed steps. A skipped step runs its condition
    again on the same rehydrated input, skips again and hands on; the run ends."""
    import uuid
    from datetime import datetime

    from sqlalchemy import delete, select

    from core.database import bind_session_tenant, get_session_maker
    from models.community import WorkflowDefinition
    from models.workflow_execution import (
        NodeExecution, NodeExecutionStatus, WorkflowExecution, WorkflowExecutionStatus,
    )
    from services.workflow_execution import WorkflowExecutionService

    tenant = "default-tenant"
    definition = {
        "nodes": [
            {"id": "s", "type": "start", "parameters": {}},
            {"id": "skipped", "type": "task", "parameters": {"skip_condition": "skip_me"}},
            {"id": "e", "type": "end", "parameters": {}},
        ],
        "edges": [{"source": "s", "target": "skipped"}, {"source": "skipped", "target": "e"}],
    }
    workflow_id = str(uuid.uuid4())
    maker = get_session_maker()
    async with maker() as db:
        bind_session_tenant(db, tenant)
        db.add(WorkflowDefinition(id=workflow_id, name=f"g29-{workflow_id[:8]}", tenant_id=tenant,
                                  definition=definition))
        await db.flush()
        execution = WorkflowExecution(
            workflow_id=workflow_id, tenant_id=tenant, context={}, input_data={},
            status=WorkflowExecutionStatus.RESUMING, started_at=datetime.utcnow())
        db.add(execution)
        await db.flush()
        for node_id, node_type, status, outputs in (
            ("s", "start", NodeExecutionStatus.COMPLETED, {"skip_me": True}),
            ("skipped", "task", NodeExecutionStatus.SKIPPED, None),
            ("e", "end", NodeExecutionStatus.PENDING, None),
        ):
            db.add(NodeExecution(execution_id=execution.id, node_id=node_id, node_type=node_type,
                                 status=status, inputs={}, outputs=outputs))
        await db.commit()
        execution_id = execution.id

    try:
        async with maker() as db:
            await WorkflowExecutionService(db)._execute_workflow_locally(execution_id, tenant_id=tenant)
        async with maker() as db:
            bind_session_tenant(db, tenant)
            run = (await db.execute(select(WorkflowExecution).where(
                WorkflowExecution.id == execution_id).execution_options(populate_existing=True))).scalar_one()
            rows = {r.node_id: r for r in (await db.execute(select(NodeExecution).where(
                NodeExecution.execution_id == execution_id).execution_options(
                    populate_existing=True))).scalars().all()}
        assert run.status == WorkflowExecutionStatus.COMPLETED
        assert rows["skipped"].status == NodeExecutionStatus.SKIPPED
        assert rows["e"].status == NodeExecutionStatus.COMPLETED
        assert rows["e"].inputs == {"skip_me": True}
    finally:
        async with maker() as db:
            bind_session_tenant(db, tenant)
            await db.execute(delete(NodeExecution).where(NodeExecution.execution_id == execution_id))
            await db.execute(delete(WorkflowExecution).where(WorkflowExecution.id == execution_id))
            await db.execute(delete(WorkflowDefinition).where(WorkflowDefinition.id == workflow_id))
            await db.commit()
