"""Cancelling a run stops it and keeps it cancelled (G22).

Runs the real start path: start_execution hands the run to a background
task on its own session, and cancel_execution arrives on another session
while a step is in flight.
"""
import asyncio
from types import SimpleNamespace

import pytest
import pytest_asyncio
from fastapi import HTTPException
from sqlalchemy import delete, select, update

from core.database import bind_session_tenant, get_session_maker
from events.event_bus import event_bus
from models.community import WorkflowDefinition
from models.workflow_execution import (
    NodeExecution, NodeExecutionStatus, WorkflowExecution, WorkflowExecutionStatus,
)
from nodes.base_node import BaseNode
from nodes.executor import NodeExecutor
from nodes.models import NodeConfig, NodeInstance, NodeStatus, NodeType, WorkflowInstance
from nodes.registry import node_registry
from services import workflow_execution as workflow_execution_module
from services.workflow_execution import WorkflowExecutionService

TENANT = "default-tenant"
STEP_TYPE = "b2CancelTestStep"
ran = []


class _Step(BaseNode):
    async def execute(self, input_data, context):
        ran.append(self.config.id)
        await asyncio.sleep(self.config.parameters.get("sleep", 0))
        if self.config.parameters.get("fail"):
            raise RuntimeError("step failed after the run was cancelled")
        return {self.config.id: "done"}


@pytest_asyncio.fixture
async def step_type():
    """Registers the test step and gives this test loop its own NullPool
    session maker: the app's pooled connections are bound to the first loop
    that opened them, and pytest-asyncio runs each test on a fresh loop."""
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
    from sqlalchemy.pool import NullPool
    from core import database
    from core.config import get_settings

    engine = create_async_engine(
        str(get_settings().DATABASE_URL), poolclass=NullPool,
        connect_args={"server_settings": database._SESSION_DEFAULTS},
    )
    saved = database._async_session_maker
    database._async_session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    ran.clear()
    node_registry.register_custom_node(STEP_TYPE, _Step)
    yield
    node_registry._custom_nodes.pop(STEP_TYPE, None)
    database._async_session_maker = saved
    await engine.dispose()


def _session():
    db = get_session_maker()()
    bind_session_tenant(db, TENANT)
    return db


@pytest_asyncio.fixture
async def workflows(step_type):
    created = []
    yield created
    async with _session() as db:
        for workflow_id in created:
            execution_ids = select(WorkflowExecution.id).where(WorkflowExecution.workflow_id == workflow_id)
            await db.execute(delete(NodeExecution).where(NodeExecution.execution_id.in_(execution_ids)))
            await db.execute(delete(WorkflowExecution).where(WorkflowExecution.workflow_id == workflow_id))
            await db.execute(delete(WorkflowDefinition).where(WorkflowDefinition.id == workflow_id))
        await db.commit()


def _sequence(*steps):
    nodes = [{"id": node_id, "type": STEP_TYPE, "parameters": params} for node_id, params in steps]
    edges = [{"source": a["id"], "target": b["id"]} for a, b in zip(nodes, nodes[1:])]
    return {"nodes": nodes, "edges": edges}


async def _start(workflows, definition):
    async with _session() as db:
        workflow = WorkflowDefinition(tenant_id=TENANT, name="b2-cancel", definition=definition, active=True)
        db.add(workflow)
        await db.commit()
        workflows.append(workflow.id)
        service = WorkflowExecutionService(db)
        execution = await service.create_execution(workflow.id, tenant_id=TENANT)
        await service.start_execution(execution.id)
    return execution.id


async def _wait_until_running(node_id):
    for _ in range(250):
        if node_id in ran:
            return
        await asyncio.sleep(0.02)
    raise AssertionError(f"{node_id} never started; ran={ran}")


async def _cancel(execution_id):
    async with _session() as db:
        return await WorkflowExecutionService(db).cancel_execution(execution_id)


async def _finish():
    await asyncio.gather(*list(workflow_execution_module._background_tasks))


async def _state(execution_id):
    async with _session() as db:
        status = (await db.execute(
            select(WorkflowExecution.status).where(WorkflowExecution.id == execution_id)
        )).scalar_one()
        rows = (await db.execute(
            select(NodeExecution).where(NodeExecution.execution_id == execution_id)
        )).scalars().all()
    return status, {row.node_id: row.status for row in rows}


@pytest.mark.asyncio
async def test_cancel_stops_before_next_step(step_type, workflows):
    execution_id = await _start(workflows, _sequence(("s1", {}), ("s2", {"sleep": 1}), ("s3", {})))
    await _wait_until_running("s2")

    await _cancel(execution_id)
    await _finish()

    status, nodes = await _state(execution_id)
    assert ran == ["s1", "s2"]
    assert status == WorkflowExecutionStatus.CANCELLED
    assert nodes["s3"] == NodeExecutionStatus.CANCELLED


@pytest.mark.asyncio
async def test_cancel_finished_run_is_noop(step_type, workflows):
    execution_id = await _start(workflows, _sequence(("only", {})))
    await _finish()

    response = await _cancel(execution_id)

    assert response.status == WorkflowExecutionStatus.COMPLETED
    status, nodes = await _state(execution_id)
    assert status == WorkflowExecutionStatus.COMPLETED
    assert nodes["only"] == NodeExecutionStatus.COMPLETED


@pytest.mark.asyncio
async def test_cancelled_run_publishes_no_failed_event(step_type, workflows, monkeypatch):
    failed_events = []
    publish = event_bus.publish

    async def recording_publish(event_name, data, **kwargs):
        if event_name == "workflow.execution.failed":
            failed_events.append(data)
        await publish(event_name, data, **kwargs)

    monkeypatch.setattr(event_bus, "publish", recording_publish)
    execution_id = await _start(
        workflows, _sequence(("s1", {}), ("s2", {"sleep": 0.5, "fail": True}), ("s3", {}))
    )
    await _wait_until_running("s2")
    await _cancel(execution_id)
    await _finish()

    status, nodes = await _state(execution_id)
    assert [e for e in failed_events if e.get("execution_id") == str(execution_id)] == []
    assert status == WorkflowExecutionStatus.CANCELLED
    assert nodes["s2"] == NodeExecutionStatus.CANCELLED
    assert "s3" not in ran


@pytest.mark.asyncio
async def test_exception_escaping_a_cancelled_run_ends_it_cancelled(step_type, workflows, monkeypatch):
    failed_events = []
    publish = event_bus.publish
    cancelled = asyncio.Event()

    async def recording_publish(event_name, data, **kwargs):
        if event_name == "workflow.execution.failed":
            failed_events.append(data)
        await publish(event_name, data, **kwargs)

    async def raising_run(self, workflow_instance, node_ids, initial_data, is_cancelled=None):
        ran.append("started")
        await cancelled.wait()
        raise RuntimeError("raised after the run was cancelled")

    monkeypatch.setattr(event_bus, "publish", recording_publish)
    monkeypatch.setattr(NodeExecutor, "_execute_from_nodes", raising_run)
    execution_id = await _start(workflows, _sequence(("s1", {})))
    await _wait_until_running("started")
    await _cancel(execution_id)
    cancelled.set()
    await _finish()

    status, _ = await _state(execution_id)
    assert status == WorkflowExecutionStatus.CANCELLED
    assert [e for e in failed_events if e.get("execution_id") == str(execution_id)] == []


async def _running_execution(workflows):
    """A RUNNING run with one PENDING node row and no executor behind it."""
    async with _session() as db:
        workflow = WorkflowDefinition(
            tenant_id=TENANT, name="b2-cancel", definition=_sequence(("only", {})), active=True
        )
        db.add(workflow)
        await db.commit()
        workflows.append(workflow.id)
        execution = await WorkflowExecutionService(db).create_execution(workflow.id, tenant_id=TENANT)
        await db.execute(
            update(WorkflowExecution).where(WorkflowExecution.id == execution.id)
            .values(status=WorkflowExecutionStatus.RUNNING)
        )
        db.add(NodeExecution(
            execution_id=execution.id, node_id="only", node_type=STEP_TYPE,
            status=NodeExecutionStatus.PENDING,
        ))
        await db.commit()
    return workflow.id, execution.id


def _record_events(monkeypatch, name):
    events = []
    publish = event_bus.publish

    async def recording_publish(event_name, data, **kwargs):
        if event_name == name:
            events.append(data)
        await publish(event_name, data, **kwargs)

    monkeypatch.setattr(event_bus, "publish", recording_publish)
    return events


@pytest.mark.asyncio
async def test_cancel_never_overwrites_a_final_write_that_lands_first(step_type, workflows, monkeypatch):
    _, execution_id = await _running_execution(workflows)
    cancelled_events = _record_events(monkeypatch, "workflow.execution.cancelled")

    async with _session() as db:
        execute = db.execute
        calls = []

        async def execute_then_finish(statement, *args, **kwargs):
            result = await execute(statement, *args, **kwargs)
            calls.append(statement)
            if len(calls) == 1:
                # The run's final write commits after cancel read the row
                # as RUNNING and before cancel writes.
                async with _session() as runner:
                    await runner.execute(
                        update(WorkflowExecution).where(WorkflowExecution.id == execution_id)
                        .values(status=WorkflowExecutionStatus.COMPLETED)
                    )
                    await runner.commit()
            return result

        monkeypatch.setattr(db, "execute", execute_then_finish)
        response = await WorkflowExecutionService(db).cancel_execution(execution_id)

    assert response.status == WorkflowExecutionStatus.COMPLETED
    status, nodes = await _state(execution_id)
    assert status == WorkflowExecutionStatus.COMPLETED
    assert nodes["only"] == NodeExecutionStatus.PENDING
    assert cancelled_events == []


@pytest.mark.asyncio
async def test_cancel_endpoint_uses_the_service_and_is_idempotent(step_type, workflows, monkeypatch):
    from api.v1.endpoints.workflows import cancel_workflow

    workflow_id, execution_id = await _running_execution(workflows)
    cancelled_events = _record_events(monkeypatch, "workflow.execution.cancelled")
    user = {"is_superuser": True, "tenant_id": TENANT}

    async with _session() as db:
        first = await cancel_workflow(workflow_id, db=db, current_user=user)
    assert first == {"message": f"Workflow {workflow_id} cancelled successfully"}
    status, nodes = await _state(execution_id)
    assert status == WorkflowExecutionStatus.CANCELLED
    assert nodes["only"] == NodeExecutionStatus.CANCELLED
    assert len(cancelled_events) == 1

    async with _session() as db:
        with pytest.raises(HTTPException) as second:
            await cancel_workflow(workflow_id, db=db, current_user=user)
    assert second.value.status_code == 404
    assert second.value.detail == "No active workflow instance found"
    assert (await _state(execution_id))[0] == WorkflowExecutionStatus.CANCELLED
    assert len(cancelled_events) == 1


@pytest.mark.asyncio
async def test_executor_checks_cancellation_between_batches():
    def node(node_id, next_nodes=()):
        return NodeInstance(
            id=node_id, workflow_instance_id="wf-1", next_nodes=list(next_nodes),
            node_config=NodeConfig(id=node_id, name=node_id, type=NodeType.TASK),
        )

    workflow = WorkflowInstance(
        id="wf-1", template_id="t", name="cancel",
        node_instances={"a": node("a", ["b"]), "b": node("b", ["c"]), "c": node("c")},
    )
    executor = NodeExecutor()
    executed = []

    async def fake_execute_node(node_instance, workflow_instance):
        executed.append(node_instance.id)
        return SimpleNamespace(status=NodeStatus.COMPLETED, next_node_ids=node_instance.next_nodes)

    async def is_cancelled():
        return executed == ["a", "b"]

    executor.execute_node = fake_execute_node
    await executor._execute_from_nodes(workflow, ["a"], {}, is_cancelled=is_cancelled)
    assert executed == ["a", "b"]
