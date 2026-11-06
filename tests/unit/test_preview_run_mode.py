"""A preview run stops before anything acts.

Runs the real start path: create_execution threads the trigger flags into the
run, start_execution hands it to a background task on its own session, and the
policy registered in nodes.preview decides each step.
"""
import asyncio

import pytest
import pytest_asyncio
from sqlalchemy import delete, select, text

from core.database import bind_session_tenant, get_session_maker
from events.event_bus import event_bus
from models.community import WorkflowDefinition
from models.workflow_execution import (
    NodeExecution, NodeExecutionStatus, WorkflowExecution, WorkflowExecutionStatus,
)
from nodes import preview as preview_module
from nodes.base_node import BaseNode
from nodes.executor import NodeExecutor
from nodes.preview import PreviewStop, get_preview_policy, set_preview_policy
from nodes.registry import node_registry
from services import workflow_execution as workflow_execution_module
from services.workflow_execution import WorkflowExecutionService

TENANT = "default-tenant"
STEP_TYPE = "previewTestStep"
ran = []
seen = {}


class _Step(BaseNode):
    async def execute(self, input_data, context):
        ran.append(self.config.id)
        variables = context.get("workflow_variables") or {}
        seen[self.config.id] = {
            "preview": context.get("preview"),
            "preview_real_steps": context.get("preview_real_steps"),
            "is_dry_run": context.get("is_dry_run"),
            "_preview": variables.get("_preview"),
            "_preview_real_steps": variables.get("_preview_real_steps"),
        }
        await asyncio.sleep(self.config.parameters.get("sleep", 0))
        return {self.config.id: "done"}


class _Policy:
    """Decides by step id; records every question it is asked."""

    def __init__(self, decisions, default="run"):
        self.decisions = decisions
        self.default = default
        self.asked = []

    def classify(self, effective_type, node_config, real_steps):
        self.asked.append((node_config.id, effective_type, list(real_steps)))
        return self.decisions.get(node_config.id, self.default)

    def simulate(self, effective_type, node_config, input_data):
        return {"would_have": f"{effective_type}:{node_config.id}"}


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
    saved_policy = preview_module._registered_policy
    database._async_session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    ran.clear()
    seen.clear()
    set_preview_policy(None)
    node_registry.register_custom_node(STEP_TYPE, _Step)
    yield
    node_registry._custom_nodes.pop(STEP_TYPE, None)
    preview_module._registered_policy = saved_policy
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


def _definition(steps, edges=None):
    """start -> steps... -> end, in sequence unless edges are given."""
    nodes = [{"id": "start", "type": "start", "parameters": {}}]
    nodes += [{"id": node_id, "type": STEP_TYPE, "parameters": params} for node_id, params in steps]
    nodes.append({"id": "end", "type": "end", "parameters": {}})
    if edges is None:
        edges = [{"source": a["id"], "target": b["id"]} for a, b in zip(nodes, nodes[1:])]
    return {"nodes": nodes, "edges": edges}


async def _run(workflows, definition, trigger_metadata=None):
    async with _session() as db:
        workflow = WorkflowDefinition(tenant_id=TENANT, name="preview-run", definition=definition, active=True)
        db.add(workflow)
        await db.commit()
        workflows.append(workflow.id)
        service = WorkflowExecutionService(db)
        execution = await service.create_execution(
            workflow.id, tenant_id=TENANT, trigger_metadata=trigger_metadata,
        )
        await service.start_execution(execution.id)
    await asyncio.gather(*list(workflow_execution_module._background_tasks))
    async with _session() as db:
        run = (await db.execute(
            select(WorkflowExecution).where(WorkflowExecution.id == execution.id)
        )).scalar_one()
        rows = (await db.execute(
            select(NodeExecution).where(NodeExecution.execution_id == execution.id)
        )).scalars().all()
    return run, {row.node_id: row for row in rows}


@pytest.mark.asyncio
async def test_a_preview_without_a_policy_runs_nothing(workflows):
    run, rows = await _run(workflows, _definition([("work", {})]), {"preview": True})

    assert ran == []
    assert run.status == WorkflowExecutionStatus.COMPLETED
    assert run.error_details is None
    assert run.execution_metadata["preview"] is True
    stopped = run.execution_metadata["preview_stopped"]
    assert stopped["node_id"] == "work"
    assert stopped["reason"] == f"stopped_before_{STEP_TYPE}"
    assert "detail" in stopped
    assert rows["start"].status == NodeExecutionStatus.COMPLETED
    assert rows["work"].status == NodeExecutionStatus.SKIPPED
    assert rows["end"].status == NodeExecutionStatus.PENDING


@pytest.mark.asyncio
async def test_a_stop_ends_the_run_cleanly(workflows, monkeypatch):
    failed_events = []
    publish = event_bus.publish

    async def recording_publish(event_name, data, **kwargs):
        if event_name == "workflow.execution.failed":
            failed_events.append(data)
        await publish(event_name, data, **kwargs)

    monkeypatch.setattr(event_bus, "publish", recording_publish)
    policy = _Policy({"halt": "stop"})
    set_preview_policy(policy)
    # start -> first -> (halt, side) -> after_side -> end: halt and side run in
    # one batch; nothing after the stop starts, on either branch.
    definition = _definition(
        [("first", {}), ("halt", {}), ("side", {"sleep": 0.2}), ("after_side", {})],
        edges=[
            {"source": "start", "target": "first"},
            {"source": "first", "target": "halt"},
            {"source": "first", "target": "side"},
            {"source": "side", "target": "after_side"},
            {"source": "halt", "target": "end"},
            {"source": "after_side", "target": "end"},
        ],
    )
    run, rows = await _run(workflows, definition, {"preview": True})

    assert "halt" not in ran and "after_side" not in ran
    assert run.status == WorkflowExecutionStatus.COMPLETED
    assert run.error_details is None
    assert run.execution_metadata["preview"] is True
    assert run.execution_metadata["preview_stopped"]["node_id"] == "halt"
    assert rows["halt"].status == NodeExecutionStatus.SKIPPED
    assert rows["after_side"].status == NodeExecutionStatus.PENDING
    assert rows["end"].status == NodeExecutionStatus.PENDING
    assert [e for e in failed_events if e.get("execution_id") == str(run.id)] == []


@pytest.mark.asyncio
async def test_a_stop_is_never_retried(step_type):
    from nodes.models import NodeConfig, NodeInstance, NodeStatus, NodeType, WorkflowInstance

    def instance(node_id, node_type, next_nodes=(), previous=(), retries=0):
        parameters = {} if node_type != STEP_TYPE else {"custom_node_type": STEP_TYPE}
        config_type = NodeType(node_type) if node_type in ("start", "end") else NodeType.TASK
        return NodeInstance(
            id=node_id, workflow_instance_id="wf-preview",
            node_config=NodeConfig(id=node_id, name=node_id, type=config_type, parameters=parameters,
                                   retry_count=retries, retry_delay_seconds=0),
            next_nodes=list(next_nodes), previous_nodes=list(previous),
        )

    workflow = WorkflowInstance(
        id="wf-preview", template_id="t", name="retry",
        variables={"_preview": True, "_preview_real_steps": []},
        node_instances={
            "start": instance("start", "start", ["work"]),
            "work": instance("work", STEP_TYPE, ["end"], ["start"], retries=3),
            "end": instance("end", "end", [], ["work"]),
        },
    )
    calls = []
    original_run = BaseNode.run

    async def counting_run(self, node_instance, variables):
        calls.append(node_instance.id)
        return await original_run(self, node_instance, variables)

    BaseNode.run = counting_run
    try:
        await NodeExecutor().execute_workflow(workflow)
    finally:
        BaseNode.run = original_run

    assert calls == ["start", "work"]
    assert ran == []
    assert workflow.status == "completed"
    assert workflow.node_instances["work"].status == NodeStatus.SKIPPED
    assert workflow.metadata["preview_stopped"]["node_id"] == "work"


@pytest.mark.asyncio
async def test_a_stop_from_the_gate_ends_the_run_like_a_step_stop(workflows):
    class _StoppingGate:
        async def check(self, node_instance, workflow_instance, context):
            if node_instance.id == "work" and context.get("preview"):
                raise PreviewStop("work", "would_ask_a_person", {"who": "reviewer"})

    set_preview_policy(_Policy({}))
    saved_gate = NodeExecutor._autonomy_gate
    NodeExecutor.set_autonomy_gate(_StoppingGate())
    try:
        run, rows = await _run(workflows, _definition([("work", {}), ("next", {})]), {"preview": True})
    finally:
        NodeExecutor.set_autonomy_gate(saved_gate)

    assert ran == []
    assert run.status == WorkflowExecutionStatus.COMPLETED
    assert run.execution_metadata["preview_stopped"] == {
        "node_id": "work", "reason": "would_ask_a_person", "detail": {"who": "reviewer"},
    }
    assert rows["work"].status == NodeExecutionStatus.SKIPPED
    assert rows["next"].status == NodeExecutionStatus.PENDING


@pytest.mark.asyncio
async def test_simulated_steps_are_marked(workflows):
    set_preview_policy(_Policy({"outward": "simulate"}))
    run, rows = await _run(workflows, _definition([("inward", {}), ("outward", {})]), {"preview": True})

    assert ran == ["inward"]
    assert run.status == WorkflowExecutionStatus.COMPLETED
    assert run.execution_metadata["preview"] is True
    assert "preview_stopped" not in run.execution_metadata
    assert rows["outward"].status == NodeExecutionStatus.COMPLETED
    assert rows["outward"].outputs == {"would_have": f"{STEP_TYPE}:outward", "_preview": True}
    assert rows["end"].status == NodeExecutionStatus.COMPLETED


@pytest.mark.asyncio
async def test_preview_flag_reaches_every_node_context(workflows):
    policy = _Policy({})
    set_preview_policy(policy)
    run, rows = await _run(
        workflows, _definition([("a", {}), ("b", {})]),
        {"preview": True, "preview_real_steps": ["a"], "preview_id": "p-1"},
    )

    assert ran == ["a", "b"]
    for node_id in ("a", "b"):
        assert seen[node_id] == {
            "preview": True, "preview_real_steps": ["a"], "is_dry_run": False,
            "_preview": True, "_preview_real_steps": ["a"],
        }
    # Start and end are the engine's own markers: never asked.
    assert [(node_id, real) for node_id, _, real in policy.asked] == [("a", ["a"]), ("b", ["a"])]
    assert run.context["preview"] is True
    assert run.context["preview_real_steps"] == ["a"]
    assert run.execution_metadata == {"preview": True}
    async with _session() as db:
        stored = (await db.execute(
            text("SELECT context->>'preview' FROM workflow_executions WHERE id = :id"),
            {"id": str(run.id)},
        )).scalar_one()
    assert stored == "true"


@pytest.mark.asyncio
async def test_preview_and_dry_run_are_independent(workflows):
    policy = _Policy({})
    set_preview_policy(policy)

    dry, _ = await _run(workflows, _definition([("d", {})]), {"is_dry_run": True})
    assert policy.asked == []
    assert dry.context["is_dry_run"] is True and dry.context["preview"] is False
    assert "preview" not in (dry.execution_metadata or {})
    assert seen["d"]["preview"] is False and seen["d"]["_preview"] is False

    preview, _ = await _run(workflows, _definition([("p", {})]), {"preview": True})
    assert preview.context["preview"] is True and preview.context["is_dry_run"] is False
    assert seen["p"]["is_dry_run"] is False
    assert [node_id for node_id, _, _ in policy.asked] == ["p"]

    plain, _ = await _run(workflows, _definition([("n", {})]))
    assert plain.context["preview"] is False
    assert "preview" not in (plain.execution_metadata or {})
    assert [node_id for node_id, _, _ in policy.asked] == ["p"]


def test_the_default_policy_stops_every_step():
    set_preview_policy(None)
    assert get_preview_policy().classify("anything", None, ["anything"]) == "stop"
    stop = PreviewStop("n", "why")
    assert (stop.node_id, stop.reason, stop.detail) == ("n", "why", {})


@pytest.mark.asyncio
async def test_the_run_context_keeps_a_preview_when_variables_lose_it(step_type):
    """A step's outputs may overwrite the run's variables; the preview holds."""
    from nodes.models import NodeConfig, NodeInstance, NodeStatus, NodeType

    set_preview_policy(None)
    instance = NodeInstance(
        id="work", workflow_instance_id="wf-preview",
        node_config=NodeConfig(id="work", name="work", type=NodeType.TASK,
                               parameters={"custom_node_type": STEP_TYPE}),
        context={"preview": True, "preview_real_steps": []},
    )
    node = node_registry.create_node(instance.node_config)
    with pytest.raises(PreviewStop) as stopped:
        await node.run(instance, {"_preview": False})
    assert stopped.value.node_id == "work"
    assert ran == []
