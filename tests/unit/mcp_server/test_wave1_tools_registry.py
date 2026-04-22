"""Wave 1 registry sanity: every tool has a scope, a plan tier, a handler,
an inputSchema, and description."""

import pytest

from mcp_server import tool_executor, tools
from mcp_server.plan_gate import TOOL_MIN_PLAN
from mcp_server.scopes import ALL_SCOPES


WAVE_1_TOOLS = {
    # Adapters
    "list_adapters", "get_adapter", "list_my_adapter_configs", "test_adapter_config",
    # NL
    "nl_to_workflow", "analyze_intent",
    # Autonomy
    "get_workflow_autonomy", "preview_autonomy", "set_workflow_autonomy",
    # Self-extending
    "research_api", "generate_adapter", "self_extend",
    "list_generated_adapters", "get_generated_adapter_status",
    "get_generated_adapter_source",
    "approve_adapter", "reject_adapter", "activate_adapter",
    # Browser
    "browser_execute",
    # Approvals
    "list_pending_approvals", "get_approval",
    "approve_request", "reject_request",
}


def _all_tool_defs() -> dict[str, dict]:
    out = {}
    for group in (tools.COMMUNITY_TOOLS, tools.BUSINESS_TOOLS, tools.ENTERPRISE_TOOLS):
        for t in group:
            out[t["name"]] = t
    return out


def test_wave1_counts():
    """23 new Wave 1 tools across the three edition lists."""
    assert len(WAVE_1_TOOLS) == 23


def test_wave1_every_tool_has_a_definition():
    defs = _all_tool_defs()
    missing = WAVE_1_TOOLS - defs.keys()
    assert not missing, f"tools.py missing definitions: {missing}"


def test_wave1_every_tool_has_a_handler():
    missing = WAVE_1_TOOLS - tool_executor.TOOL_HANDLERS.keys()
    assert not missing, f"tool_executor.TOOL_HANDLERS missing: {missing}"


def test_wave1_every_tool_has_a_scope_mapping():
    missing = WAVE_1_TOOLS - tools.TOOL_SCOPES.keys()
    assert not missing, f"TOOL_SCOPES missing: {missing}"


def test_wave1_every_scope_is_in_the_registry():
    for name in WAVE_1_TOOLS:
        for scope in tools.TOOL_SCOPES.get(name, []):
            assert scope in ALL_SCOPES, f"{name}: {scope} not in scope registry"


def test_wave1_every_tool_has_a_plan_tier():
    missing = WAVE_1_TOOLS - TOOL_MIN_PLAN.keys()
    assert not missing, f"TOOL_MIN_PLAN missing: {missing}"


def test_wave1_plan_tier_values_are_valid():
    for name in WAVE_1_TOOLS:
        assert TOOL_MIN_PLAN[name] in {"community", "business", "enterprise"}


def test_wave1_input_schemas_are_well_formed():
    defs = _all_tool_defs()
    for name in WAVE_1_TOOLS:
        schema = defs[name]["inputSchema"]
        assert isinstance(schema, dict)
        assert schema.get("type") == "object"
        props = schema.get("properties")
        assert isinstance(props, dict), f"{name}: missing/invalid properties"


def test_wave1_descriptions_nonempty():
    defs = _all_tool_defs()
    for name in WAVE_1_TOOLS:
        desc = defs[name].get("description", "")
        assert desc and len(desc) > 20, f"{name}: description too short"


def test_wave1_self_extending_tools_are_business_tier():
    se_tools = {
        "research_api", "generate_adapter", "self_extend",
        "list_generated_adapters", "get_generated_adapter_status",
        "get_generated_adapter_source",
        "approve_adapter", "reject_adapter", "activate_adapter",
    }
    for name in se_tools:
        assert TOOL_MIN_PLAN[name] == "business"


def test_wave1_browser_execute_is_business_with_write_browser_scope():
    assert TOOL_MIN_PLAN["browser_execute"] == "business"
    assert "write:browser" in tools.TOOL_SCOPES["browser_execute"]


def test_wave1_approval_queue_is_business():
    for name in ("list_pending_approvals", "get_approval",
                  "approve_request", "reject_request"):
        assert TOOL_MIN_PLAN[name] == "business"


def test_wave1_adapters_and_nl_are_community():
    for name in ("list_adapters", "get_adapter", "list_my_adapter_configs",
                  "nl_to_workflow", "analyze_intent",
                  "get_workflow_autonomy", "preview_autonomy"):
        assert TOOL_MIN_PLAN[name] == "community"


def test_wave1_set_autonomy_is_business():
    assert TOOL_MIN_PLAN["set_workflow_autonomy"] == "business"


def test_wave1_input_schema_required_fields():
    """Critical identity fields must be marked required so the MCP
    server rejects malformed calls at the schema layer."""
    defs = _all_tool_defs()
    expectations = {
        "get_adapter": {"adapter_id"},
        "test_adapter_config": {"config_id"},
        "nl_to_workflow": {"text"},
        "analyze_intent": {"text"},
        "get_workflow_autonomy": {"workflow_id"},
        "preview_autonomy": {"workflow_id", "level"},
        "set_workflow_autonomy": {"workflow_id"},
        "research_api": {"api_name"},
        "generate_adapter": {"name", "api_name", "base_url", "auth_type", "capabilities"},
        "self_extend": {"api_name"},
        "get_generated_adapter_status": {"adapter_id"},
        "get_generated_adapter_source": {"adapter_id"},
        "approve_adapter": {"adapter_id"},
        "reject_adapter": {"adapter_id"},
        "activate_adapter": {"adapter_id"},
        "browser_execute": {"actions"},
        "get_approval": {"request_id"},
        "approve_request": {"request_id"},
        "reject_request": {"request_id", "reason"},
    }
    for name, required in expectations.items():
        required_list = set(defs[name]["inputSchema"].get("required") or [])
        missing = required - required_list
        assert not missing, f"{name}: required must include {missing}"
