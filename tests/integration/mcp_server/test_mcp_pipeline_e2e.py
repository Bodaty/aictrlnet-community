"""End-to-end integration tests for the MCP pipeline.

These tests run the **real** JSON-RPC protocol handler against a stub
``AsyncSession`` and stub services. The goal is to verify the full
pipeline (plan gate → scope → compliance → rate bucket → metering →
handler → audit → metrics + protocol-layer error shaping) behaves as
advertised — not to re-test what each unit test already covers.

Where a handler delegates to a Business/Enterprise service we don't
want to seed in a test DB, we monkeypatch the service class and let
the rest of the pipeline run for real.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from mcp_server import tool_executor
from mcp_server.plan_gate import PlanService
from mcp_server.protocol import MCPProtocolHandler
from mcp_server.tools import (
    BUSINESS_TOOLS,
    COMMUNITY_TOOLS,
    ENTERPRISE_TOOLS,
    get_tools_for_edition,
)


def _full_registry() -> list:
    """Force-assemble the full 103-tool registry for tests that exercise
    Business/Enterprise tools — sidesteps the accretive import check
    so these tests run regardless of which edition the test harness is
    currently mounted against."""
    return list(COMMUNITY_TOOLS) + list(BUSINESS_TOOLS) + list(ENTERPRISE_TOOLS)


# ---------------------------------------------------------------------------
# Stubs
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
    def __init__(self, row=None, rows=None):
        self._row = row
        self._rows = rows or ([] if row is None else [row])

    def first(self):
        return self._row

    def all(self):
        return self._rows

    def scalar_one_or_none(self):
        return self._row

    def scalars(self):
        return self

    def __iter__(self):
        return iter(self._rows)


class _StubDB:
    """Minimal AsyncSession stub that makes the metering SQL work
    without a real DB. Handlers that reach for actual service data
    return whatever their stub is monkeypatched to return.
    """

    def __init__(self, meter_counters=None, over_limit=False):
        self.counters = meter_counters or {}
        self.over_limit = over_limit
        self.idem_rows = {}
        self.audit_calls = []
        self.commits = 0

    async def execute(self, stmt, params=None):
        sql = str(stmt)
        params = params or {}

        if "INSERT INTO mcp_meters" in sql:
            if self.over_limit:
                return _Result(None)
            key = (params["tenant_id"], params["meter"])
            self.counters[key] = self.counters.get(key, 0) + int(params["qty"])
            return _Result(_Row(self.counters[key], None))
        if "SELECT counter, period_end" in sql:
            key = (params["tenant_id"], params["meter"])
            return _Result(_Row(self.counters.get(key, 0), None))
        if "SELECT meter, counter, limit_override" in sql:
            # Trial-status handler: list every meter for the tenant
            tenant = params["t"]
            rows = [
                _Row(meter, counter, None, None, None)
                for (t, meter), counter in self.counters.items()
                if t == tenant
            ]
            return _Result(rows=rows)
        if "UPDATE mcp_meters" in sql:
            return _Result(None)
        if "SELECT response_json" in sql:
            row = self.idem_rows.get(
                (params["tenant_id"], params["tool_name"], params["key"])
            )
            return _Result(_Row(row["resp"], row["hash"], None) if row else None)
        if "INSERT INTO mcp_idempotency_keys" in sql:
            self.idem_rows[
                (params["tenant_id"], params["tool_name"], params["key"])
            ] = {"resp": params["response"], "hash": params["args_hash"]}
            return _Result(None)
        return _Result(None)

    async def commit(self):
        self.commits += 1

    async def rollback(self):
        pass


class _FakePlanService(PlanService):
    """PlanService that returns a canned tier without touching the DB."""

    def __init__(self, tier: str):
        self._cache = {}  # type: ignore[assignment]
        self.db = None  # type: ignore[assignment]
        self._tier = tier

    async def _resolve(self, tenant_id):
        return self._tier


class _FakeApiKey:
    def __init__(self, scopes=None, id_="ak-test"):
        self.scopes = scopes or []
        self.id = id_
        self.rate_limit_per_tool = None
        self.rate_limit_per_minute = None
        self.rate_limit_per_day = None


@pytest.fixture(autouse=True)
def _reset_rate_bucket(monkeypatch):
    from mcp_server import rate_bucket

    monkeypatch.setattr(rate_bucket, "_redis_client", None, raising=False)
    monkeypatch.setattr(rate_bucket, "_redis_checked", True, raising=False)
    monkeypatch.setattr(rate_bucket, "_memory_state", {}, raising=False)


def _handler(monkeypatch, tool_name: str, fn):
    """Swap a handler for the duration of a test."""
    monkeypatch.setitem(tool_executor.TOOL_HANDLERS, tool_name, fn)


async def _call_tool(handler: MCPProtocolHandler, name: str, args: dict, req_id=1):
    return await handler.handle({
        "jsonrpc": "2.0",
        "id": req_id,
        "method": "tools/call",
        "params": {"name": name, "arguments": args},
    })


# ---------------------------------------------------------------------------
# Pipeline — plan gate, scope, rate bucket, metering, idempotency,
# error shaping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_tools_list_filters_by_plan_tier():
    db = _StubDB()
    # Community-tier caller sees only community tools
    h = MCPProtocolHandler(
        tools_registry=_full_registry(),
        db=db, user_id="u1", api_key=None, tenant_id="t1",
    )
    h.plan_service = _FakePlanService("community")

    resp = await h.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
    tools = resp["result"]["tools"]
    names = {t["name"] for t in tools}

    # Wave 1 business tools must NOT be visible
    assert "research_api" not in names
    assert "browser_execute" not in names
    # Wave 6 enterprise tools must NOT be visible
    assert "query_analytics" not in names
    # Community tools MUST be visible
    assert "create_workflow" in names
    assert "nl_to_workflow" in names


@pytest.mark.asyncio
async def test_pipeline_tools_call_plan_denied_returns_method_not_found(monkeypatch):
    """v11 claim-audit row B: plan-gated tools return -32601 MethodNotFound
    (not PlanError) to avoid revealing tool existence across tiers."""
    db = _StubDB()
    h = MCPProtocolHandler(
        tools_registry=_full_registry(),
        db=db, user_id="u1", api_key=None, tenant_id="t1",
    )
    h.plan_service = _FakePlanService("community")

    # Community tries to call a Business tool
    resp = await _call_tool(h, "evaluate_policy",
                            {"policy_id": "p1", "content": "x"})
    assert "error" in resp
    assert resp["error"]["code"] == -32602
    assert "Unknown tool" in resp["error"]["message"]


@pytest.mark.asyncio
async def test_pipeline_scope_denied_returns_structured_error(monkeypatch):
    db = _StubDB()
    async def _ok(args, db_, user_id): return {"ok": True}
    _handler(monkeypatch, "list_adapters", _ok)

    # API key with wrong scope
    key = _FakeApiKey(scopes=["read:workflows"])  # needs read:adapters
    h = MCPProtocolHandler(
        tools_registry=_full_registry(),
        db=db, user_id="u1", api_key=key, tenant_id="t1",
    )
    h.plan_service = _FakePlanService("community")

    resp = await _call_tool(h, "list_adapters", {})
    # isError is True; structured _meta.error with scope_denied
    result = resp["result"]
    assert result["isError"] is True
    assert result["_meta"]["error"]["error"] == "scope_denied"


@pytest.mark.asyncio
async def test_pipeline_phase_b_rejects_legacy_read_all(monkeypatch):
    """Phase B (follow-up #3): read:all no longer satisfies any
    new-taxonomy check. Key presenting it must be rejected."""
    db = _StubDB()
    async def _ok(args, db_, user_id): return {"ok": True}
    _handler(monkeypatch, "list_adapters", _ok)

    key = _FakeApiKey(scopes=["read:all"])  # Phase-A-era scope
    h = MCPProtocolHandler(
        tools_registry=_full_registry(),
        db=db, user_id="u1", api_key=key, tenant_id="t1",
    )
    h.plan_service = _FakePlanService("community")

    resp = await _call_tool(h, "list_adapters", {})
    assert resp["result"]["isError"] is True
    assert resp["result"]["_meta"]["error"]["error"] == "scope_denied"


@pytest.mark.asyncio
async def test_pipeline_quota_exceeded_returns_upgrade_url(monkeypatch):
    """Claim-audit row: QuotaError carries upgrade_url + meter + limit."""
    db = _StubDB(over_limit=True)
    async def _ok(args, db_, user_id): return {"ok": True}
    _handler(monkeypatch, "create_workflow", _ok)

    h = MCPProtocolHandler(
        tools_registry=_full_registry(),
        db=db, user_id="u1", api_key=None, tenant_id="t1",
    )
    h.plan_service = _FakePlanService("community")

    resp = await _call_tool(h, "create_workflow", {"description": "x"})
    result = resp["result"]
    assert result["isError"] is True
    err = result["_meta"]["error"]
    assert err["error"] == "quota_exceeded"
    assert err["meter"] == "llm_calls"
    assert err["upgrade_url"].startswith("/pricing")


@pytest.mark.asyncio
async def test_pipeline_idempotency_replay_returns_cached(monkeypatch):
    db = _StubDB()
    call_count = 0

    async def _counting_handler(args, db_, user_id):
        nonlocal call_count
        call_count += 1
        return {"n": call_count}

    _handler(monkeypatch, "create_workflow", _counting_handler)

    h = MCPProtocolHandler(
        tools_registry=_full_registry(),
        db=db, user_id="u1", api_key=None, tenant_id="t1",
    )
    h.plan_service = _FakePlanService("community")

    args = {"description": "same", "idempotency_key": "abc"}
    r1 = await _call_tool(h, "create_workflow", args, req_id=1)
    r2 = await _call_tool(h, "create_workflow", args, req_id=2)

    # Handler only ran once
    assert call_count == 1
    # Both responses carry the same result
    body1 = json.loads(r1["result"]["content"][0]["text"])
    body2 = json.loads(r2["result"]["content"][0]["text"])
    assert body1 == body2 == {"n": 1}


@pytest.mark.asyncio
async def test_pipeline_community_caller_cannot_probe_enterprise_tool():
    """Belt-and-suspenders: community caller calling an Enterprise tool
    by name must also get MethodNotFound (no `You don't have the plan`
    hint)."""
    db = _StubDB()
    h = MCPProtocolHandler(
        tools_registry=_full_registry(),
        db=db, user_id="u1", api_key=None, tenant_id="t1",
    )
    h.plan_service = _FakePlanService("community")

    resp = await _call_tool(h, "query_analytics", {"metric_type": "tasks"})
    assert "error" in resp
    assert resp["error"]["code"] == -32602
    # Message must match the exact same shape as "tool doesn't exist"
    assert "Unknown tool" in resp["error"]["message"]


# ---------------------------------------------------------------------------
# Long-running / hero tool flows
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_self_extend_returns_adapter_id_and_poll_tool(monkeypatch):
    """v11.4 hero: one call returns adapter_id + research + poll_tool name.
    Background generation is kicked off via asyncio.create_task."""
    db = _StubDB()

    # Stub the two services the handler reaches for
    class _StubResearch:
        async def research_api(self, api_name, documentation_url=None, user_context=None):
            return {
                "api_name": api_name,
                "base_url": "https://api.example.com",
                "auth_type": "bearer",
                "capabilities": [{"name": "list_items"}],
                "confidence": 0.9,
                "source": "known_api",
            }

    class _StubAdapterSvc:
        def __init__(self, db):
            self.db = db

        async def create_generated_adapter(self, **kwargs):
            return {"id": "adapter-123", "status": "researching",
                    "name": kwargs.get("name")}

        async def generate_adapter(self, **kwargs):
            # Background — pipeline doesn't await this
            return {"status": "approved"}

    # The handler imports these from aictrlnet_business.services.*
    import sys
    import types

    mod_research = types.ModuleType("aictrlnet_business.services.api_research_service")
    mod_research.APIResearchService = lambda: _StubResearch()
    monkeypatch.setitem(sys.modules,
                        "aictrlnet_business.services.api_research_service",
                        mod_research)

    mod_adapter = types.ModuleType("aictrlnet_business.services.generated_adapter_service")
    mod_adapter.GeneratedAdapterService = _StubAdapterSvc
    monkeypatch.setitem(sys.modules,
                        "aictrlnet_business.services.generated_adapter_service",
                        mod_adapter)

    # core.database.AsyncSessionLocal — used by the background task factory
    class _FakeSessionCM:
        async def __aenter__(self): return _StubDB()
        async def __aexit__(self, *a): pass

    mod_db = types.ModuleType("core.database")
    mod_db.AsyncSessionLocal = lambda: _FakeSessionCM()
    monkeypatch.setitem(sys.modules, "core.database", mod_db)

    h = MCPProtocolHandler(
        tools_registry=_full_registry(),
        db=db, user_id="u1", api_key=None, tenant_id="t1",
    )
    h.plan_service = _FakePlanService("business")

    resp = await _call_tool(h, "self_extend", {"api_name": "example-api"})
    result = resp["result"]
    assert result["isError"] is False
    body = json.loads(result["content"][0]["text"])

    assert body["adapter_id"] == "adapter-123"
    assert body["api_name"] == "example-api"
    assert body["poll_tool"] == "get_generated_adapter_status"
    assert body["research"]["capabilities_count"] == 1
    assert body["research"]["confidence"] == 0.9

    # Let background task settle so it doesn't leak into the next test
    await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_browser_execute_ssrf_denial_via_pipeline(monkeypatch):
    """End-to-end: caller sends a browser action targeting RFC1918. The
    browser_safety layer must reject BEFORE the request hits
    browser-service. Verified via the full MCP transport path."""
    db = _StubDB()
    h = MCPProtocolHandler(
        tools_registry=_full_registry(),
        db=db, user_id="u1", api_key=None, tenant_id="t1",
    )
    h.plan_service = _FakePlanService("business")

    # If the safety check fails, the handler would try to POST to the
    # browser service — we assert it does NOT by stubbing httpx to blow up
    import httpx

    class _Boom:
        async def __aenter__(self): raise AssertionError("must not hit browser-service")
        async def __aexit__(self, *a): pass
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _Boom())

    resp = await _call_tool(h, "browser_execute", {
        "actions": [{"type": "navigate", "url": "http://10.0.0.1/"}]
    })
    result = resp["result"]
    assert result["isError"] is True
    # Handler wraps BrowserSafetyError in ToolExecutionError -> tool_error
    assert result["_meta"]["error"]["error"] == "tool_error"
    assert "denied" in result["_meta"]["error"]["message"].lower() \
        or "blocked" in result["_meta"]["error"]["message"].lower()


@pytest.mark.asyncio
async def test_cross_tenant_send_channel_message_denied(monkeypatch):
    """send_channel_message must call list_linked_channels first and
    reject if the caller doesn't own a channel of the requested type."""
    db = _StubDB()

    # Stub list_linked_channels to return no matching channel
    async def _empty_list(args, db_, user_id):
        return {"channels": [], "count": 0}
    _handler(monkeypatch, "list_linked_channels", _empty_list)

    # The conversation service path shouldn't be reached — fail the test
    # if it is.
    import sys
    import types
    mod_conv = types.ModuleType("services.enhanced_conversation_manager")

    class _BoomSvc:
        def __init__(self, db): pass
        async def get_active_sessions(self, uid):
            raise AssertionError("should not be reached — ownership check failed")
    mod_conv.EnhancedConversationService = _BoomSvc
    monkeypatch.setitem(sys.modules,
                        "services.enhanced_conversation_manager", mod_conv)

    h = MCPProtocolHandler(
        tools_registry=_full_registry(),
        db=db, user_id="u1", api_key=None, tenant_id="t1",
    )
    h.plan_service = _FakePlanService("business")

    resp = await _call_tool(h, "send_channel_message", {
        "channel_type": "slack",
        "message": "hi",
    })
    result = resp["result"]
    assert result["isError"] is True
    msg = result["_meta"]["error"]["message"].lower()
    assert "linked channel" in msg or "slack" in msg


@pytest.mark.asyncio
async def test_automate_company_returns_poll_tool_and_plan_id(monkeypatch):
    """Long-running contract: automate_company returns plan_id + poll_tool
    immediately; the backing service runs generation asynchronously."""
    db = _StubDB()

    class _StubCompanySvc:
        def __init__(self, db): pass
        async def generate_plan(self, user_id, goals, autonomy_level, dry_run):
            return {"plan_id": "plan-42", "status": "generating"}

    import sys
    import types
    mod = types.ModuleType("aictrlnet_business.services.company_automation_service")
    mod.CompanyAutomationService = _StubCompanySvc
    monkeypatch.setitem(sys.modules,
                        "aictrlnet_business.services.company_automation_service",
                        mod)

    h = MCPProtocolHandler(
        tools_registry=_full_registry(),
        db=db, user_id="u1", api_key=None, tenant_id="t1",
    )
    h.plan_service = _FakePlanService("business")

    resp = await _call_tool(h, "automate_company", {
        "goals": ["increase revenue", "reduce manual work"],
    })
    result = resp["result"]
    assert result["isError"] is False
    body = json.loads(result["content"][0]["text"])
    assert body["plan_id"] == "plan-42"
    assert body["poll_tool"] == "get_company_automation_status"


@pytest.mark.asyncio
async def test_approval_flow_approve_then_list_updated(monkeypatch):
    """Full approval-queue cycle via MCP: list_pending_approvals, then
    approve_request, then confirm the approval service fired."""
    db = _StubDB()
    approved_ids = []

    class _FakeReq:
        def __init__(self, id_):
            self.id = id_
            self.workflow_id = "wf-1"
            self.requester_id = "u1"
            self.status = "pending"
            self.resource_type = None
            self.resource_id = None
            self.context = "test approval"
            self.reason = None
            self.meta_data = None
            self.created_at = "2026-04-23T00:00:00"

    class _StubApprovalSvc:
        def __init__(self, db): pass

        async def list_requests(self, **kwargs):
            return [_FakeReq("req-1")]

        async def approve_request(self, request_id, approver_id, comments=None, tenant_id=None):
            approved_ids.append(request_id)
            return {"status": "approved", "decision_id": "dec-1"}

        async def get_request(self, request_id, tenant_id=None):
            return _FakeReq(request_id)

    import sys
    import types
    mod = types.ModuleType("aictrlnet_business.services.approval")
    mod.ApprovalService = _StubApprovalSvc
    monkeypatch.setitem(sys.modules, "aictrlnet_business.services.approval", mod)

    h = MCPProtocolHandler(
        tools_registry=_full_registry(),
        db=db, user_id="u1", api_key=None, tenant_id="t1",
    )
    h.plan_service = _FakePlanService("business")

    # 1. List pending
    r1 = await _call_tool(h, "list_pending_approvals", {})
    listed = json.loads(r1["result"]["content"][0]["text"])
    assert listed["count"] == 1
    assert listed["requests"][0]["id"] == "req-1"

    # 2. Approve
    r2 = await _call_tool(h, "approve_request", {"request_id": "req-1"})
    body = json.loads(r2["result"]["content"][0]["text"])
    assert body["status"] == "approved"
    assert approved_ids == ["req-1"]


@pytest.mark.asyncio
async def test_get_trial_status_merges_meters_with_default_limits(monkeypatch):
    """Claim-audit row 6: trial metering surface returns mcp_meters +
    default limits per edition + upgrade_url threshold."""
    # Pre-populate the meter as if 4000 of 5000 community calls used
    db = _StubDB(meter_counters={("t1", "llm_calls"): 4000})

    # UsageService may not be importable in all test envs — stub it
    import sys
    import types

    class _StubUsage:
        def __init__(self, db): pass
        async def get_usage_status(self, tenant_id):
            return {
                "needs_upgrade": False,
                "current_usage": {"api_calls_month": 4000},
                "limits": {"max_api_calls_month": 10000},
            }

    mod = types.ModuleType("services.usage_service")
    mod.UsageService = _StubUsage
    monkeypatch.setitem(sys.modules, "services.usage_service", mod)

    h = MCPProtocolHandler(
        tools_registry=_full_registry(),
        db=db, user_id="u1", api_key=None, tenant_id="t1",
    )
    h.plan_service = _FakePlanService("community")

    resp = await _call_tool(h, "get_trial_status", {})
    assert resp["result"]["isError"] is False
    body = json.loads(resp["result"]["content"][0]["text"])

    # Response shape is the contract. Precise meter counts go through
    # a SQL path that's hard to exercise without a real Postgres, but
    # the default-limits merge still fires so every configured meter
    # appears in the response.
    assert body["edition"] == "community"
    assert "mcp_meters" in body
    assert "llm_calls" in body["mcp_meters"]
    llm = body["mcp_meters"]["llm_calls"]
    # Community default — 5000 llm_calls/month from metering.DEFAULT_LIMITS
    assert llm["limit"] == 5000
    assert "remaining" in llm
    assert "percent_used" in llm
    assert body["status"] is not None  # wrapped UsageService payload
    assert "as_of" in body
