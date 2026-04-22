"""Wave 2 registry sanity: API-key economy + governance-visibility tools."""

import pytest

from mcp_server import tool_executor, tools
from mcp_server.plan_gate import TOOL_MIN_PLAN
from mcp_server.scopes import ALL_SCOPES


WAVE_2_TOOLS = {
    # API-key + subscription (community)
    "list_api_keys",
    "get_api_key_usage",
    "get_subscription",
    "get_upgrade_options",
    # Governance visibility (business)
    "list_ai_policies",
    "create_policy",
    "get_ai_audit_logs",
    "list_violations",
}


def _all_tool_defs() -> dict[str, dict]:
    out = {}
    for group in (tools.COMMUNITY_TOOLS, tools.BUSINESS_TOOLS, tools.ENTERPRISE_TOOLS):
        for t in group:
            out[t["name"]] = t
    return out


def test_wave2_count():
    assert len(WAVE_2_TOOLS) == 8


def test_wave2_every_tool_defined():
    defs = _all_tool_defs()
    missing = WAVE_2_TOOLS - defs.keys()
    assert not missing, f"tools.py missing: {missing}"


def test_wave2_every_tool_has_handler():
    missing = WAVE_2_TOOLS - tool_executor.TOOL_HANDLERS.keys()
    assert not missing, f"TOOL_HANDLERS missing: {missing}"


def test_wave2_every_tool_has_scope():
    missing = WAVE_2_TOOLS - tools.TOOL_SCOPES.keys()
    assert not missing, f"TOOL_SCOPES missing: {missing}"


def test_wave2_every_scope_is_valid():
    for name in WAVE_2_TOOLS:
        for scope in tools.TOOL_SCOPES.get(name, []):
            assert scope in ALL_SCOPES, f"{name}: unknown scope {scope}"


def test_wave2_every_tool_has_plan_tier():
    missing = WAVE_2_TOOLS - TOOL_MIN_PLAN.keys()
    assert not missing, f"TOOL_MIN_PLAN missing: {missing}"


def test_wave2_plan_tier_assignments():
    # Community: introspection tools that any authenticated caller can use
    community = {"list_api_keys", "get_api_key_usage",
                 "get_subscription", "get_upgrade_options"}
    for name in community:
        assert TOOL_MIN_PLAN[name] == "community", \
            f"{name} should be community-tier"

    # Business: governance visibility
    business = {"list_ai_policies", "create_policy",
                "get_ai_audit_logs", "list_violations"}
    for name in business:
        assert TOOL_MIN_PLAN[name] == "business", \
            f"{name} should be business-tier"


def test_wave2_api_key_tools_use_read_usage_scope():
    for name in ("list_api_keys", "get_api_key_usage"):
        assert "read:usage" in tools.TOOL_SCOPES[name]


def test_wave2_subscription_tools_use_read_subscription_scope():
    for name in ("get_subscription", "get_upgrade_options"):
        assert "read:subscription" in tools.TOOL_SCOPES[name]


def test_wave2_policy_tools_use_policy_scopes():
    assert "read:policies" in tools.TOOL_SCOPES["list_ai_policies"]
    assert "read:policies" in tools.TOOL_SCOPES["list_violations"]
    assert "write:policies" in tools.TOOL_SCOPES["create_policy"]


def test_wave2_audit_tool_uses_read_audit_scope():
    assert "read:audit" in tools.TOOL_SCOPES["get_ai_audit_logs"]


def test_wave2_input_schemas_are_valid():
    defs = _all_tool_defs()
    for name in WAVE_2_TOOLS:
        schema = defs[name]["inputSchema"]
        assert isinstance(schema, dict)
        assert schema.get("type") == "object"
        assert isinstance(schema.get("properties"), dict)


def test_wave2_required_fields():
    defs = _all_tool_defs()
    # create_policy is the only Wave 2 tool with required fields
    assert set(defs["create_policy"]["inputSchema"].get("required", [])) >= {
        "name", "policy_type", "rules"
    }
    # Others are all-optional listings — accept empty required
    for name in ("list_api_keys", "get_api_key_usage", "get_subscription",
                  "get_upgrade_options", "list_ai_policies", "get_ai_audit_logs",
                  "list_violations"):
        required = defs[name]["inputSchema"].get("required") or []
        assert required == [], f"{name} should not require fields; got {required}"


def test_wave2_descriptions_reference_positioning():
    """v11 'Claude gains governance' should be reflected in the governance
    tool descriptions so Claude's tool-selection prompts have the hook."""
    defs = _all_tool_defs()
    assert "gain" in defs["list_ai_policies"]["description"].lower() \
        or "gover" in defs["list_ai_policies"]["description"].lower()


def test_wave2_total_tool_count_is_42():
    """11 original + 23 Wave 1 + 8 Wave 2 = 42.

    We check the three authoritative registries: tool defs (tools.py),
    scope map, and plan-tier map. TOOL_HANDLERS can temporarily hold
    test-registered handlers because the pipeline test uses setdefault
    for stub registration, so we treat that dict as a lower bound only.
    """
    all_defs = _all_tool_defs()
    assert len(all_defs) == 42
    assert len(tools.TOOL_SCOPES) == 42
    # Every defined tool must have a handler; extras (from test stubs)
    # are allowed and don't break production.
    assert set(all_defs.keys()).issubset(set(tool_executor.TOOL_HANDLERS.keys()))
