"""Wave 6 registry sanity: Enterprise admin — analytics, audit,
compliance, federated knowledge, cross-tenant, fleet, license."""

import pytest

from mcp_server import tool_executor, tools
from mcp_server.plan_gate import TOOL_MIN_PLAN
from mcp_server.scopes import ALL_SCOPES


WAVE_6_TOOLS = {
    # Analytics
    "query_analytics", "get_dashboard_metrics", "get_metric_trends",
    # Audit
    "get_audit_logs", "get_audit_summary",
    # Compliance
    "run_compliance_check", "list_compliance_standards",
    "get_enterprise_risk_assessment",
    # Organizations + Tenants
    "list_organizations", "list_tenants",
    # Federated knowledge + Cross-tenant
    "federated_knowledge_query", "get_cross_tenant_insights",
    # Fleet Management
    "list_fleet_agents", "get_fleet_autonomy_summary",
    # License Management
    "get_license_status", "list_license_entitlements",
}


def _all_tool_defs() -> dict[str, dict]:
    out = {}
    for group in (tools.COMMUNITY_TOOLS, tools.BUSINESS_TOOLS, tools.ENTERPRISE_TOOLS):
        for t in group:
            out[t["name"]] = t
    return out


def test_wave6_count():
    assert len(WAVE_6_TOOLS) == 16


def test_wave6_every_tool_defined():
    defs = _all_tool_defs()
    missing = WAVE_6_TOOLS - defs.keys()
    assert not missing, f"tools.py missing: {missing}"


def test_wave6_every_tool_has_handler():
    missing = WAVE_6_TOOLS - tool_executor.TOOL_HANDLERS.keys()
    assert not missing


def test_wave6_every_tool_has_scope():
    missing = WAVE_6_TOOLS - tools.TOOL_SCOPES.keys()
    assert not missing


def test_wave6_all_tools_are_enterprise_tier():
    for name in WAVE_6_TOOLS:
        assert TOOL_MIN_PLAN[name] == "enterprise", \
            f"{name} must be enterprise-tier"


def test_wave6_scopes_are_valid():
    for name in WAVE_6_TOOLS:
        for scope in tools.TOOL_SCOPES.get(name, []):
            assert scope in ALL_SCOPES


def test_wave6_scope_assignments():
    # Analytics
    for name in ("query_analytics", "get_dashboard_metrics", "get_metric_trends"):
        assert "read:analytics" in tools.TOOL_SCOPES[name]
    # Audit
    for name in ("get_audit_logs", "get_audit_summary"):
        assert "read:audit" in tools.TOOL_SCOPES[name]
    # Compliance
    for name in ("run_compliance_check", "list_compliance_standards",
                  "get_enterprise_risk_assessment", "list_organizations",
                  "list_tenants"):
        assert "read:compliance" in tools.TOOL_SCOPES[name]
    # Fleet
    for name in ("list_fleet_agents", "get_fleet_autonomy_summary"):
        assert "read:fleet" in tools.TOOL_SCOPES[name]
    # License
    for name in ("get_license_status", "list_license_entitlements"):
        assert "read:license" in tools.TOOL_SCOPES[name]
    # Federated knowledge
    assert "read:knowledge" in tools.TOOL_SCOPES["federated_knowledge_query"]
    # Cross-tenant analytics
    assert "read:analytics" in tools.TOOL_SCOPES["get_cross_tenant_insights"]


def test_wave6_schemas_well_formed():
    defs = _all_tool_defs()
    for name in WAVE_6_TOOLS:
        schema = defs[name]["inputSchema"]
        assert schema["type"] == "object"
        assert isinstance(schema.get("properties"), dict)


def test_wave6_required_fields():
    defs = _all_tool_defs()
    expectations = {
        "query_analytics": {"metric_type"},
        "get_metric_trends": {"metric_type"},
        "federated_knowledge_query": {"query"},
    }
    for name, required in expectations.items():
        required_list = set(defs[name]["inputSchema"].get("required") or [])
        missing = required - required_list
        assert not missing, f"{name}: required must include {missing}"


def test_wave6_descriptions_reference_v11_pillars():
    """Hero tools should mention their v11 pillar for Claude's tool-
    selection prompts."""
    defs = _all_tool_defs()
    assert "ciso" in defs["get_audit_summary"]["description"].lower() \
        or "compliance officer" in defs["get_audit_summary"]["description"].lower()
    assert "fleet" in defs["get_fleet_autonomy_summary"]["description"].lower()
    assert "enterprise" in defs["get_license_status"]["description"].lower()


@pytest.mark.asyncio
async def test_wave6_handlers_degrade_gracefully_when_enterprise_absent():
    """On a community deploy where the Enterprise service package isn't
    installed, every Wave 6 handler must return a structured
    feature_pending response, not crash.

    The plan gate blocks these in production (they're Enterprise-tier),
    but the safety net is still needed for tests + misconfigured
    deploys.
    """

    class _StubDB:
        async def execute(self, *a, **k):
            raise NotImplementedError()

        async def commit(self):
            pass

    # Note: the handlers do hit real Enterprise service imports on this
    # monorepo, so a few may succeed. The contract is: no handler may
    # raise an unhandled exception. feature_pending OR real result are
    # both OK.
    db = _StubDB()
    fallback_args = {
        "query_analytics": {"metric_type": "task"},
        "get_metric_trends": {"metric_type": "task"},
        "federated_knowledge_query": {"query": "test"},
        "get_cross_tenant_insights": {},
    }
    for name in WAVE_6_TOOLS:
        handler = tool_executor.TOOL_HANDLERS[name]
        args = fallback_args.get(name, {})
        try:
            result = await handler(args, db, "user-1")
        except Exception as e:
            # The only acceptable "failure" is handled via _enterprise_pending.
            # If a handler raises, flag it.
            pytest.fail(f"{name} raised {type(e).__name__}: {e}")
        assert isinstance(result, dict), f"{name} must return dict"


def test_wave6_total_count_at_least_103():
    """11 + 23 + 8 + 2 + 40 + 3 + 16 = 103."""
    defs = _all_tool_defs()
    assert len(defs) >= 103
    assert len(tools.TOOL_SCOPES) >= 103
