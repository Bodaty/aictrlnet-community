"""A workflow's run-as owner is server-only: set from who creates it, never from a request.

``workflow_definitions.run_as_user_id`` decides whose connections (a Gmail token, saved keys)
an unattended run uses. If a client could set it, anyone who can edit a workflow could make
its scheduled runs send as a colleague.
"""
from types import SimpleNamespace

import pytest

from schemas.workflow import WorkflowCreate, WorkflowResponse, WorkflowUpdate
from services import run_as


def test_owner_for_is_the_creator_and_system_is_no_one():
    assert run_as.owner_for("u-1") == "u-1"
    assert run_as.owner_for("system") is None
    assert run_as.owner_for(None) is None


class _Rows:
    def __init__(self, row):
        self.row = row

    def first(self):
        return self.row


class _Db:
    def __init__(self, row):
        self.row, self.added, self.queries = row, [], []

    async def execute(self, query, params=None):
        self.queries.append(params)
        return _Rows(self.row)

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        pass

    async def refresh(self, obj):
        pass


@pytest.mark.parametrize("row,workflow_tenant,expected", [
    (("t1", True), "t1", "owner-1"),
    ((None, True), "default-tenant", "owner-1"),     # Community users carry NULL for the default tenant
    (("t1", False), "t1", None),                     # inactive
    (("t2", True), "t1", None),                      # another tenant
    (None, "t1", None),                              # no such user
])
async def test_run_as_owner_is_honoured_only_for_an_active_user_of_the_workflows_tenant(row, workflow_tenant, expected):
    workflow = SimpleNamespace(id="wf", run_as_user_id="owner-1", tenant_id=workflow_tenant)
    assert await run_as.run_as_owner(_Db(row), workflow) == expected


async def test_a_workflow_without_an_owner_asks_nothing():
    db = _Db(("t1", True))
    assert await run_as.run_as_owner(db, SimpleNamespace(id="wf", run_as_user_id=None, tenant_id="t1")) is None
    assert db.queries == []


def test_node_context_owner_reads_what_the_run_recorded_at_its_start():
    assert run_as.node_context_owner({"workflow_owner_id": "owner-1"}) == "owner-1"
    assert run_as.node_context_owner(None) is None


def test_no_request_or_response_schema_carries_the_owner():
    for schema in (WorkflowCreate, WorkflowUpdate, WorkflowResponse):
        assert "run_as_user_id" not in schema.model_fields
    update = WorkflowUpdate(**{"name": "x", "run_as_user_id": "victim", "owner_id": "victim"})
    assert set(update.model_dump(exclude_unset=True)) == {"name"}


@pytest.mark.parametrize("actor,expected", [("creator-1", "creator-1"), ("system", None)])
async def test_creating_a_workflow_makes_its_creator_the_owner_whatever_the_client_sends(actor, expected):
    from services.workflow import WorkflowService

    db = _Db(None)
    data = WorkflowCreate(**{
        "name": "Nightly", "run_as_user_id": "victim", "owner_id": "victim",
        "definition": {"nodes": [], "edges": [], "metadata": {"owner_id": "victim", "run_as_user_id": "victim"},
                       "run_as_user_id": "victim"},
    })
    workflow = await WorkflowService(db).create_workflow(data, tenant_id="t1", actor_id=actor)
    assert workflow.run_as_user_id == expected
    assert db.added == [workflow]
