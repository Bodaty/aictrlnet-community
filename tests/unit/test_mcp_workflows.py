"""Workflows created and run over MCP: owned by the caller, and run as the REST endpoint runs them.

- ``POST /mcp/workflows`` (MCPWorkflowService) created a workflow with no run-as owner, so its
  unattended runs used no one's connections.
- The MCP tool ``create_workflow`` goes through NLP generation, which records the caller.
- The mcp-llm ``workflow`` tool built ``WorkflowService()`` with no session and passed ``inputs``
  to a method that takes none, so every call failed (TypeError, answered as a 500). It now runs the
  REST execute endpoint itself: the same licence check, tenant check and execution path.
"""
from types import SimpleNamespace

import pytest
from fastapi import HTTPException


class _Result:
    def __init__(self, value):
        self.value = value

    def scalar_one_or_none(self):
        return self.value


class _Db:
    def __init__(self, found=None):
        self.found, self.added = found, []

    async def execute(self, *args, **kwargs):
        return _Result(self.found)

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        pass

    async def refresh(self, obj):
        pass


async def test_a_workflow_created_over_rest_mcp_is_owned_by_the_caller():
    from mcp_server.services.workflow import MCPWorkflowService

    db = _Db()
    result = await MCPWorkflowService(db).create_workflow_via_mcp(
        {"name": "From MCP", "steps": [{"id": "a", "type": "task"}],
         "run_as_user_id": "victim", "owner_id": "victim"},
        owner_id="caller-1")
    assert result["status"] == "created"
    assert db.added[0].run_as_user_id == "caller-1"


async def test_an_nlp_generated_workflow_is_owned_by_the_person_asking(monkeypatch):
    """The MCP create_workflow tool's path (NLPService) records its caller as owner."""
    from services.nlp import NLPService

    svc = NLPService.__new__(NLPService)
    svc.db, svc.edition = _Db(), "community"
    workflow = await svc._create_workflow_from_nlp(
        "every morning email me the report", {"nodes": [], "edges": []}, "template_based",
        tenant_id="t1", context={"workflow_name": "Morning report"}, owner_id="caller-1")
    assert workflow.run_as_user_id == "caller-1"


async def test_the_mcp_create_workflow_tool_asks_as_its_caller(monkeypatch):
    from mcp_server import tool_executor
    from services import nlp

    seen = {}

    async def process(self, prompt, context=None, return_transparency=True, user_id=None):
        seen["user_id"] = user_id
        return {"workflow": {"id": "w1", "name": "Morning report", "definition": {"nodes": []}}}

    monkeypatch.setattr(nlp.NLPService, "__init__", lambda self, db: None)
    monkeypatch.setattr(nlp.NLPService, "process_natural_language", process)
    out = await tool_executor._handle_create_workflow({"description": "every morning", "name": "x"}, None, "caller-1")
    assert out["status"] == "created" and seen["user_id"] == "caller-1"


def _workflow(tenant):
    return SimpleNamespace(id="w1", tenant_id=tenant, definition={"nodes": []})


@pytest.fixture
def rest_path(monkeypatch):
    """The REST endpoint's own path, with the licence counter and the run itself recorded."""
    from api.v1.endpoints import workflows as rest

    calls = {"create": [], "license": []}

    async def check_limit(self, **kwargs):
        calls["license"].append(kwargs)

    async def create_execution(self, **kwargs):
        calls["create"].append(kwargs)
        return SimpleNamespace(id="e1")

    async def start_execution(self, execution_id, agent_id=None):
        return SimpleNamespace(id=execution_id, status="running",
                               model_dump=lambda **k: {"id": execution_id, "status": "running"})

    class _Tracker:
        async def track_workflow_execution(self, **kwargs):
            pass

    async def tracker(db):
        return _Tracker()

    monkeypatch.setattr(rest.LicenseEnforcer, "check_limit", check_limit)
    monkeypatch.setattr(rest.WorkflowExecutionService, "create_execution", create_execution)
    monkeypatch.setattr(rest.WorkflowExecutionService, "start_execution", start_execution)
    monkeypatch.setattr(rest, "get_usage_tracker", tracker)
    return calls


async def test_the_llm_workflow_tool_runs_a_workflow_as_its_caller(rest_path):
    from mcp_server.llm_endpoints import handle_mcp_tool_execution

    user = SimpleNamespace(id="caller-1", tenant_id="t1", is_active=True, is_superuser=False)
    out = await handle_mcp_tool_execution(
        {"tool": "workflow", "input": {"workflow_id": "w1", "inputs": {"day": "mon"}}},
        current_user=user, db=_Db(found=_workflow("t1")))
    assert out["status"] == "completed"
    assert rest_path["license"] and rest_path["create"][0]["user_id"] == "caller-1"
    assert rest_path["create"][0]["input_data"] == {"day": "mon"}


async def test_the_llm_workflow_tool_refuses_another_tenants_workflow_as_rest_does(rest_path):
    from mcp_server.llm_endpoints import handle_mcp_tool_execution

    user = SimpleNamespace(id="caller-1", tenant_id="t1", is_active=True, is_superuser=False)
    with pytest.raises(HTTPException) as err:
        await handle_mcp_tool_execution(
            {"tool": "workflow", "input": {"workflow_id": "w1"}},
            current_user=user, db=_Db(found=_workflow("t2")))
    assert err.value.status_code == 404 and rest_path["create"] == []


async def test_the_llm_workflow_tool_refuses_an_inactive_user_as_rest_does(rest_path):
    from mcp_server.llm_endpoints import handle_mcp_tool_execution

    user = SimpleNamespace(id="caller-1", tenant_id="t1", is_active=False, is_superuser=False)
    with pytest.raises(HTTPException) as err:
        await handle_mcp_tool_execution(
            {"tool": "workflow", "input": {"workflow_id": "w1"}},
            current_user=user, db=_Db(found=_workflow("t1")))
    assert err.value.status_code == 400 and rest_path["create"] == []


async def test_the_llm_workflow_tool_needs_a_workflow_id(rest_path):
    from mcp_server.llm_endpoints import handle_mcp_tool_execution

    user = SimpleNamespace(id="caller-1", tenant_id="t1", is_active=True, is_superuser=False)
    with pytest.raises(HTTPException) as err:
        await handle_mcp_tool_execution({"tool": "workflow", "input": {}}, current_user=user, db=_Db())
    assert err.value.status_code == 422


async def test_the_rest_mcp_endpoint_creates_the_workflow_owned_by_its_caller():
    from api.v1.endpoints.mcp_server import create_workflow_via_mcp
    from schemas.mcp import MCPWorkflowRequest

    db = _Db()
    user = SimpleNamespace(id="caller-1", tenant_id="t1", is_active=True, is_superuser=False)
    out = await create_workflow_via_mcp(
        MCPWorkflowRequest(workflow_definition={"name": "From MCP", "steps": [], "owner_id": "victim"}),
        db=db, current_user=user)
    assert out.status == "created"
    assert db.added[0].run_as_user_id == "caller-1"
