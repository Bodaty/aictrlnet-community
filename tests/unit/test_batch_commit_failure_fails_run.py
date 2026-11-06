"""A run whose shared session can't commit between steps fails with the cause.

Between batches the run ends the transaction on the session its steps share.
When that commit fails (a step left a write on the session that the database
refuses), the run must end FAILED with that error at that point, and no later
step may run. Logging it and carrying on ran the remaining steps and then
failed the run on an unrelated error.

Runs the real start path against the real database: the step leaves an
unflushed row whose primary key already exists, so the commit raises a
duplicate-key IntegrityError.
"""
import asyncio
import uuid

import pytest
import pytest_asyncio
from sqlalchemy import delete, select

from core.database import bind_session_tenant, get_session_maker
from models.community import WorkflowDefinition
from models.workflow_execution import (
    NodeExecution, NodeExecutionStatus, WorkflowExecution, WorkflowExecutionStatus,
)
from nodes.base_node import BaseNode
from nodes.registry import node_registry
from services import workflow_execution as workflow_execution_module
from services.workflow_execution import WorkflowExecutionService

# Its own tenant: the Tier A sweeps delete every default-tenant row created
# while they run (tests/smoke_common/sweep_residue.py), and they run beside
# this test (other xdist workers, other editions' suites).
TENANT = f"t-{uuid.uuid4().hex[:12]}"
BAD_WRITE = "d12bBadWriteStep"
STEP = "d12bPlainStep"
ran = []


class _BadWrite(BaseNode):
    """Leaves a write on the run's session that can't commit: a second row
    with the run's own workflow id."""

    async def execute(self, input_data, context):
        ran.append(self.config.id)
        context["db"].add(WorkflowDefinition(
            id=context["workflow_definition_id"], tenant_id=TENANT,
            name="d12b-duplicate", definition={"nodes": [], "edges": []}, active=True))
        return {self.config.id: "done"}


class _Step(BaseNode):
    async def execute(self, input_data, context):
        ran.append(self.config.id)
        return {self.config.id: "done"}


@pytest_asyncio.fixture
async def step_types():
    """Registers the test steps and gives this test loop its own NullPool
    session maker (pooled connections are bound to the loop that opened them)."""
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
    node_registry.register_custom_node(BAD_WRITE, _BadWrite)
    node_registry.register_custom_node(STEP, _Step)
    created = []
    yield created
    node_registry._custom_nodes.pop(BAD_WRITE, None)
    node_registry._custom_nodes.pop(STEP, None)
    async with _session() as db:
        for workflow_id in created:
            execution_ids = select(WorkflowExecution.id).where(WorkflowExecution.workflow_id == workflow_id)
            await db.execute(delete(NodeExecution).where(NodeExecution.execution_id.in_(execution_ids)))
            await db.execute(delete(WorkflowExecution).where(WorkflowExecution.workflow_id == workflow_id))
            await db.execute(delete(WorkflowDefinition).where(WorkflowDefinition.id == workflow_id))
        await db.commit()
    database._async_session_maker = saved
    await engine.dispose()


def _session():
    db = get_session_maker()()
    bind_session_tenant(db, TENANT)
    return db


@pytest.mark.asyncio
async def test_failed_commit_between_steps_fails_the_run_with_its_cause(step_types):
    definition = {
        "nodes": [
            {"id": "first", "type": STEP, "parameters": {}},
            {"id": "bad", "type": BAD_WRITE, "parameters": {}},
            {"id": "after", "type": STEP, "parameters": {}},
        ],
        "edges": [{"source": "first", "target": "bad"},
                  {"source": "bad", "target": "after"}],
    }
    async with _session() as db:
        workflow = WorkflowDefinition(tenant_id=TENANT, name="d12b-batch-commit",
                                      definition=definition, active=True)
        db.add(workflow)
        await db.commit()
        step_types.append(workflow.id)
        service = WorkflowExecutionService(db)
        execution = await service.create_execution(workflow.id, tenant_id=TENANT)
        await service.start_execution(execution.id)

    await asyncio.wait_for(
        asyncio.gather(*list(workflow_execution_module._background_tasks)), timeout=60)

    async with _session() as db:
        run = (await db.execute(
            select(WorkflowExecution).where(WorkflowExecution.id == execution.id)
        )).scalar_one()
        rows = {r.node_id: r.status for r in (await db.execute(
            select(NodeExecution).where(NodeExecution.execution_id == execution.id)
        )).scalars().all()}
        definitions = (await db.execute(
            select(WorkflowDefinition.name).where(WorkflowDefinition.id == workflow.id)
        )).scalars().all()

    assert ran == ["first", "bad"], f"a step ran after the failed commit: {ran}"
    assert rows["after"] == NodeExecutionStatus.PENDING
    assert run.status == WorkflowExecutionStatus.FAILED
    error = str((run.error_details or {}).get("error"))
    assert "duplicate key" in error, f"the run failed on something else: {error}"
    assert definitions == ["d12b-batch-commit"]
