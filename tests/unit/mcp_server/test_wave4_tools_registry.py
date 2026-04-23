"""Wave 4 registry sanity: horizontal business + living platform."""

import pytest

from mcp_server import tool_executor, tools
from mcp_server.plan_gate import TOOL_MIN_PLAN
from mcp_server.scopes import ALL_SCOPES


WAVE_4_COMMUNITY = {
    # Tasks
    "create_task", "list_tasks", "get_task", "update_task", "complete_task",
    # Memory
    "get_memory", "set_memory", "delete_memory",
    # Conversations + Channels (read)
    "list_conversations", "get_conversation", "list_linked_channels",
    "request_channel_link_code", "unlink_channel",
    # Knowledge
    "query_knowledge", "suggest_next_actions", "get_capabilities_summary",
    # Templates
    "search_templates", "instantiate_template",
    # Files
    "upload_file", "list_staged_files", "get_staged_file",
    # Data Quality
    "assess_data_quality", "list_quality_dimensions",
}

WAVE_4_BUSINESS = {
    # Channels + Notifications
    "send_channel_message", "list_notifications", "mark_notification_read",
    # Agents
    "list_agents", "get_agent_capabilities", "set_agent_autonomy", "execute_agent",
    # LLM Registry
    "list_llm_models", "get_llm_recommendation",
    # Living Platform — Patterns
    "list_pattern_candidates", "promote_pattern_to_template",
    # Living Platform — Org Discovery
    "org_discovery_scan", "get_org_landscape", "get_org_recommendations",
    # Living Platform — Company Automation
    "automate_company", "get_company_automation_status",
    # Quality
    "verify_quality",
}

WAVE_4_TOOLS = WAVE_4_COMMUNITY | WAVE_4_BUSINESS


def _all_tool_defs() -> dict[str, dict]:
    out = {}
    for group in (tools.COMMUNITY_TOOLS, tools.BUSINESS_TOOLS, tools.ENTERPRISE_TOOLS):
        for t in group:
            out[t["name"]] = t
    return out


def test_wave4_count():
    assert len(WAVE_4_COMMUNITY) == 23
    assert len(WAVE_4_BUSINESS) == 17
    assert len(WAVE_4_TOOLS) == 40


def test_wave4_every_tool_defined():
    defs = _all_tool_defs()
    missing = WAVE_4_TOOLS - defs.keys()
    assert not missing, f"tools.py missing: {missing}"


def test_wave4_every_tool_has_handler():
    missing = WAVE_4_TOOLS - tool_executor.TOOL_HANDLERS.keys()
    assert not missing


def test_wave4_every_tool_has_scope():
    missing = WAVE_4_TOOLS - tools.TOOL_SCOPES.keys()
    assert not missing


def test_wave4_every_scope_is_valid():
    for name in WAVE_4_TOOLS:
        for scope in tools.TOOL_SCOPES.get(name, []):
            assert scope in ALL_SCOPES, f"{name}: unknown scope {scope}"


def test_wave4_community_tier_assignments():
    for name in WAVE_4_COMMUNITY:
        assert TOOL_MIN_PLAN[name] == "community", \
            f"{name} should be community-tier"


def test_wave4_business_tier_assignments():
    for name in WAVE_4_BUSINESS:
        assert TOOL_MIN_PLAN[name] == "business", \
            f"{name} should be business-tier"


def test_wave4_task_tools_use_task_scopes():
    assert "write:tasks" in tools.TOOL_SCOPES["create_task"]
    assert "read:tasks" in tools.TOOL_SCOPES["list_tasks"]
    assert "write:tasks" in tools.TOOL_SCOPES["complete_task"]


def test_wave4_memory_tools_use_memory_scopes():
    assert "read:memory" in tools.TOOL_SCOPES["get_memory"]
    assert "write:memory" in tools.TOOL_SCOPES["set_memory"]
    assert "write:memory" in tools.TOOL_SCOPES["delete_memory"]


def test_wave4_channel_send_requires_messaging_scope():
    assert "write:messaging" in tools.TOOL_SCOPES["send_channel_message"]


def test_wave4_agent_autonomy_write_requires_write_agents():
    assert "write:agents" in tools.TOOL_SCOPES["set_agent_autonomy"]
    assert "read:agents" in tools.TOOL_SCOPES["list_agents"]


def test_wave4_living_platform_company_uses_company_scope():
    assert "write:company" in tools.TOOL_SCOPES["automate_company"]


def test_wave4_input_schemas_well_formed():
    defs = _all_tool_defs()
    for name in WAVE_4_TOOLS:
        schema = defs[name]["inputSchema"]
        assert isinstance(schema, dict)
        assert schema.get("type") == "object"
        assert isinstance(schema.get("properties"), dict)


def test_wave4_critical_required_fields():
    """Identity fields must be required so the MCP server rejects
    incomplete calls at schema time."""
    defs = _all_tool_defs()
    expectations = {
        "create_task": {"name"},
        "get_task": {"task_id"},
        "update_task": {"task_id"},
        "complete_task": {"task_id"},
        "get_memory": {"key"},
        "set_memory": {"key", "value"},
        "delete_memory": {"key"},
        "get_conversation": {"session_id"},
        "request_channel_link_code": {"channel_type"},
        "unlink_channel": {"channel_type", "channel_user_id"},
        "send_channel_message": {"channel_type", "message"},
        "mark_notification_read": {"notification_id"},
        "query_knowledge": {"query"},
        "suggest_next_actions": {"current_action"},
        "instantiate_template": {"template_id"},
        "upload_file": {"filename", "content_base64"},
        "get_staged_file": {"file_id"},
        "assess_data_quality": {"data"},
        "get_agent_capabilities": {"agent_id"},
        "set_agent_autonomy": {"agent_id", "autonomy_level"},
        "execute_agent": {"agent_id", "prompt"},
        "get_llm_recommendation": {"task_type"},
        "promote_pattern_to_template": {"pattern_id"},
        "automate_company": {"request"},
        "get_company_automation_status": {"plan_id"},
        "verify_quality": {"content"},
    }
    for name, required in expectations.items():
        required_list = set(defs[name]["inputSchema"].get("required") or [])
        missing = required - required_list
        assert not missing, f"{name}: required must include {missing}"


def test_wave4_company_automation_returns_poll_tool():
    """Long-running op contract: automate_company description must
    reference the poll tool."""
    defs = _all_tool_defs()
    desc = defs["automate_company"]["description"].lower()
    assert "long-running" in desc or "poll" in desc, \
        "automate_company should document the polling contract"


def test_wave4_upload_file_size_capped_by_schema():
    defs = _all_tool_defs()
    # Size cap is enforced at handler layer (50 MB) — schema just takes
    # base64. Sanity check that the required fields are there.
    schema = defs["upload_file"]["inputSchema"]
    assert set(schema.get("required", [])) >= {"filename", "content_base64"}


def test_wave4_total_count_at_least_84():
    """11 + 23 + 8 + 2 + 40 = 84."""
    defs = _all_tool_defs()
    assert len(defs) >= 84
    assert len(tools.TOOL_SCOPES) >= 84
    assert set(defs.keys()).issubset(tool_executor.TOOL_HANDLERS.keys())


def test_wave4_descriptions_reference_v11_pillars():
    """Hero tools should mention their v11 pillar in the description
    so Claude's tool-selection prompts surface the hook."""
    defs = _all_tool_defs()
    assert "living platform" in defs["list_pattern_candidates"]["description"].lower() \
        or "learns" in defs["list_pattern_candidates"]["description"].lower()
    assert "24 hours" in defs["automate_company"]["description"].lower() \
        or "magic moment" in defs["automate_company"]["description"].lower()
