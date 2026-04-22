"""Wave 3 registry sanity: trial metering surface."""

import pytest

from mcp_server import tool_executor, tools
from mcp_server.plan_gate import TOOL_MIN_PLAN
from mcp_server.scopes import ALL_SCOPES


WAVE_3_TOOLS = {"get_trial_status", "get_usage_report"}


def _all_tool_defs() -> dict[str, dict]:
    out = {}
    for group in (tools.COMMUNITY_TOOLS, tools.BUSINESS_TOOLS, tools.ENTERPRISE_TOOLS):
        for t in group:
            out[t["name"]] = t
    return out


def test_wave3_count():
    assert len(WAVE_3_TOOLS) == 2


def test_wave3_every_tool_defined():
    defs = _all_tool_defs()
    missing = WAVE_3_TOOLS - defs.keys()
    assert not missing, f"tools.py missing: {missing}"


def test_wave3_every_tool_has_handler():
    missing = WAVE_3_TOOLS - tool_executor.TOOL_HANDLERS.keys()
    assert not missing


def test_wave3_every_tool_has_scope():
    missing = WAVE_3_TOOLS - tools.TOOL_SCOPES.keys()
    assert not missing


def test_wave3_uses_read_usage_scope():
    # Both Wave 3 tools are read-only usage introspection
    for name in WAVE_3_TOOLS:
        assert "read:usage" in tools.TOOL_SCOPES[name], \
            f"{name} must require read:usage"


def test_wave3_is_community_tier():
    for name in WAVE_3_TOOLS:
        assert TOOL_MIN_PLAN[name] == "community", \
            f"{name} must be community-tier so trial users can see their quota"


def test_wave3_scopes_valid():
    for name in WAVE_3_TOOLS:
        for scope in tools.TOOL_SCOPES.get(name, []):
            assert scope in ALL_SCOPES


def test_wave3_schemas_valid():
    defs = _all_tool_defs()
    for name in WAVE_3_TOOLS:
        schema = defs[name]["inputSchema"]
        assert schema["type"] == "object"
        assert isinstance(schema.get("properties"), dict)


def test_wave3_get_usage_report_days_bounded():
    defs = _all_tool_defs()
    props = defs["get_usage_report"]["inputSchema"]["properties"]
    assert "days" in props
    assert props["days"].get("minimum") == 1
    assert props["days"].get("maximum") == 90


def test_wave3_get_trial_status_takes_no_required_args():
    defs = _all_tool_defs()
    schema = defs["get_trial_status"]["inputSchema"]
    required = schema.get("required") or []
    assert required == []


def test_wave3_descriptions_reference_v11_claim():
    """v11.2 'trial metering works through MCP' should be defensible — the
    tool descriptions should hook that claim so Claude's tool-selection
    prompts surface it."""
    defs = _all_tool_defs()
    ts_desc = defs["get_trial_status"]["description"].lower()
    ur_desc = defs["get_usage_report"]["description"].lower()
    assert "metering" in ts_desc or "trial" in ts_desc
    assert "usage" in ur_desc or "tracking" in ur_desc


def test_wave3_total_counts_coherent():
    """Every Wave 3 tool def has a scope + plan tier + handler entry."""
    defs = _all_tool_defs()
    assert len(defs) >= 44  # 11 + 23 + 8 + 2
    for name in WAVE_3_TOOLS:
        assert name in defs
        assert name in tools.TOOL_SCOPES
        assert name in TOOL_MIN_PLAN
        assert name in tool_executor.TOOL_HANDLERS
