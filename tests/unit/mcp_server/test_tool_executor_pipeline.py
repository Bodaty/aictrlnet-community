"""Unit tests for the tool_executor pipeline gate order and semantics.

These tests exercise the gating logic without touching real services.
Handlers are replaced with simple awaitables; the DB session is a stub
that emulates the metering SQL.
"""

import pytest

from mcp_server import tool_executor
from mcp_server.metering import QuotaError
from mcp_server.plan_gate import PlanError, PlanService
from mcp_server.tool_executor import execute_tool, ScopeError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _Row:
    def __init__(self, *values):
        self._values = values

    def __iter__(self):
        return iter(self._values)

    def __getitem__(self, i):
        return self._values[i]

    def first(self):
        return self


class _Result:
    def __init__(self, row):
        self._row = row

    def first(self):
        return self._row


class _StubSession:
    def __init__(self, counters=None, over_limit=False):
        self.counters = counters or {}
        self.over_limit = over_limit

    async def execute(self, stmt, params=None):
        sql = str(stmt)
        if "INSERT INTO mcp_meters" in sql:
            if self.over_limit:
                return _Result(None)
            key = (params["tenant_id"], params["meter"])
            self.counters[key] = self.counters.get(key, 0) + int(params["qty"])
            return _Result(_Row(self.counters[key], None))
        if "SELECT counter, period_end" in sql:
            key = (params["tenant_id"], params["meter"])
            return _Result(_Row(self.counters.get(key, 0), None))
        if "UPDATE mcp_meters" in sql:
            return _Result(None)
        if "SELECT response_json" in sql:
            return _Result(None)
        if "INSERT INTO mcp_idempotency_keys" in sql:
            return _Result(None)
        return _Result(None)

    async def commit(self):
        pass

    async def rollback(self):
        pass


class _FakePlanService(PlanService):
    def __init__(self, tier: str):
        # Bypass parent __init__ so we don't need a real db
        self._cache = {}  # type: ignore[assignment]
        self.db = None  # type: ignore[assignment]
        self._tier = tier

    async def _resolve(self, tenant_id):
        return self._tier


class _FakeApiKey:
    def __init__(self, scopes=None):
        self.scopes = scopes or []
        self.id = "ak"
        self.rate_limit_per_tool = None
        self.rate_limit_per_minute = None
        self.rate_limit_per_day = None


@pytest.fixture(autouse=True)
def _reset_rate_bucket(monkeypatch):
    from mcp_server import rate_bucket
    monkeypatch.setattr(rate_bucket, "_redis_client", None, raising=False)
    monkeypatch.setattr(rate_bucket, "_redis_checked", True, raising=False)
    monkeypatch.setattr(rate_bucket, "_memory_state", {}, raising=False)


@pytest.fixture(autouse=True)
def _register_test_tools(monkeypatch):
    """Stub a few handlers for pipeline tests.

    Every override goes through ``monkeypatch.setitem`` so the test
    teardown reverts it. Tools used here (create_workflow,
    list_workflows, evaluate_policy, check_compliance, query_analytics)
    all exist in the real ``TOOL_HANDLERS`` — we're just swapping the
    implementation to a no-side-effects stub for the duration of the
    test so the pipeline-ordering assertions stay deterministic.
    """
    async def _ok(args, db, user_id):
        return {"ok": True, "args": args, "user_id": user_id}

    for name in (
        "list_workflows",
        "create_workflow",
        "evaluate_policy",
        "check_compliance",
        "query_analytics",
    ):
        monkeypatch.setitem(tool_executor.TOOL_HANDLERS, name, _ok)


# ---------------------------------------------------------------------------
# Plan gate ordering
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_plan_gate_denies_community_on_business_tool():
    db = _StubSession()
    svc = _FakePlanService("community")
    with pytest.raises(PlanError):
        await execute_tool(
            tool_name="evaluate_policy",
            arguments={"policy_id": "p1", "content": "x"},
            db=db,
            user_id="u1",
            tenant_id="t1",
            plan_service=svc,
        )


@pytest.mark.asyncio
async def test_plan_gate_allows_business_on_business_tool():
    db = _StubSession()
    svc = _FakePlanService("business")
    out = await execute_tool(
        tool_name="evaluate_policy",
        arguments={"policy_id": "p1", "content": "x"},
        db=db,
        user_id="u1",
        tenant_id="t1",
        plan_service=svc,
    )
    assert out["ok"] is True


@pytest.mark.asyncio
async def test_plan_gate_before_scope_check():
    """If both plan AND scope fail, plan error is raised first."""
    db = _StubSession()
    svc = _FakePlanService("community")
    # API key has no matching scope either — plan should still fire first
    key = _FakeApiKey(scopes=[])
    with pytest.raises(PlanError):
        await execute_tool(
            tool_name="evaluate_policy",
            arguments={"policy_id": "p1", "content": "x"},
            db=db,
            user_id="u1",
            tenant_id="t1",
            api_key=key,
            plan_service=svc,
        )


# ---------------------------------------------------------------------------
# Scope check
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_scope_check_denies_missing_scope():
    db = _StubSession()
    svc = _FakePlanService("business")
    key = _FakeApiKey(scopes=["read:workflows"])  # needs read:policies
    with pytest.raises(ScopeError):
        await execute_tool(
            tool_name="evaluate_policy",
            arguments={"policy_id": "p1", "content": "x"},
            db=db,
            user_id="u1",
            tenant_id="t1",
            api_key=key,
            plan_service=svc,
        )


@pytest.mark.asyncio
async def test_scope_check_accepts_legacy_read_all():
    """Phase-A compatibility: an API key with read:all scope still works."""
    db = _StubSession()
    svc = _FakePlanService("business")
    key = _FakeApiKey(scopes=["read:all"])
    out = await execute_tool(
        tool_name="evaluate_policy",
        arguments={"policy_id": "p1", "content": "x"},
        db=db,
        user_id="u1",
        tenant_id="t1",
        api_key=key,
        plan_service=svc,
    )
    assert out["ok"] is True


@pytest.mark.asyncio
async def test_scope_check_accepts_new_taxonomy():
    db = _StubSession()
    svc = _FakePlanService("business")
    key = _FakeApiKey(scopes=["read:policies"])
    out = await execute_tool(
        tool_name="evaluate_policy",
        arguments={"policy_id": "p1", "content": "x"},
        db=db,
        user_id="u1",
        tenant_id="t1",
        api_key=key,
        plan_service=svc,
    )
    assert out["ok"] is True


@pytest.mark.asyncio
async def test_jwt_user_bypasses_scope_check():
    db = _StubSession()
    svc = _FakePlanService("business")
    # No api_key passed — JWT path, full access
    out = await execute_tool(
        tool_name="evaluate_policy",
        arguments={"policy_id": "p1", "content": "x"},
        db=db,
        user_id="u1",
        tenant_id="t1",
        plan_service=svc,
    )
    assert out["ok"] is True


# ---------------------------------------------------------------------------
# Quota
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_quota_error_on_overage():
    db = _StubSession(over_limit=True)
    svc = _FakePlanService("community")
    with pytest.raises(QuotaError):
        await execute_tool(
            tool_name="create_workflow",
            arguments={"description": "x"},
            db=db,
            user_id="u1",
            tenant_id="t1",
            plan_service=svc,
        )


# ---------------------------------------------------------------------------
# Metrics emitted on all paths
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_metrics_emitted_on_success(monkeypatch):
    calls = []

    def record(tool, status, plan, **_kw):
        calls.append((tool, status, plan))

    from mcp_server import observability
    monkeypatch.setattr(
        observability,
        "record_invocation",
        lambda **kw: record(kw["tool"], kw["status"], kw["plan"]),
    )

    db = _StubSession()
    svc = _FakePlanService("community")
    await execute_tool(
        tool_name="list_workflows",
        arguments={},
        db=db,
        user_id="u1",
        tenant_id="t1",
        plan_service=svc,
    )
    assert calls[-1][0] == "list_workflows"
    assert calls[-1][1] == "success"


@pytest.mark.asyncio
async def test_metrics_emitted_on_plan_denial(monkeypatch):
    calls = []
    from mcp_server import observability
    monkeypatch.setattr(
        observability,
        "record_invocation",
        lambda **kw: calls.append((kw["tool"], kw["status"])),
    )

    db = _StubSession()
    svc = _FakePlanService("community")
    with pytest.raises(PlanError):
        await execute_tool(
            tool_name="check_compliance",
            arguments={"server_id": "s1"},
            db=db,
            user_id="u1",
            tenant_id="t1",
            plan_service=svc,
        )
    assert calls[-1] == ("check_compliance", "plan_denied")
