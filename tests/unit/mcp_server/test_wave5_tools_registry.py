"""Wave 5 registry sanity: Institute (education-led GTM)."""

import asyncio

import pytest

from mcp_server import tool_executor, tools
from mcp_server.plan_gate import TOOL_MIN_PLAN
from mcp_server.scopes import ALL_SCOPES


WAVE_5_TOOLS = {
    "list_institute_modules",
    "enroll_in_module",
    "get_certification_status",
}


def _all_tool_defs() -> dict[str, dict]:
    out = {}
    for group in (tools.COMMUNITY_TOOLS, tools.BUSINESS_TOOLS, tools.ENTERPRISE_TOOLS):
        for t in group:
            out[t["name"]] = t
    return out


def test_wave5_count():
    assert len(WAVE_5_TOOLS) == 3


def test_wave5_every_tool_defined():
    defs = _all_tool_defs()
    missing = WAVE_5_TOOLS - defs.keys()
    assert not missing


def test_wave5_every_tool_has_handler():
    missing = WAVE_5_TOOLS - tool_executor.TOOL_HANDLERS.keys()
    assert not missing


def test_wave5_every_tool_has_scope():
    missing = WAVE_5_TOOLS - tools.TOOL_SCOPES.keys()
    assert not missing


def test_wave5_scopes_valid():
    for name in WAVE_5_TOOLS:
        for scope in tools.TOOL_SCOPES.get(name, []):
            assert scope in ALL_SCOPES


def test_wave5_uses_institute_scopes():
    assert "read:institute" in tools.TOOL_SCOPES["list_institute_modules"]
    assert "read:institute" in tools.TOOL_SCOPES["get_certification_status"]
    assert "write:institute" in tools.TOOL_SCOPES["enroll_in_module"]


def test_wave5_is_community_tier():
    for name in WAVE_5_TOOLS:
        assert TOOL_MIN_PLAN[name] == "community", \
            f"{name} must be community-tier — Institute is the top-of-funnel"


def test_wave5_schemas_valid():
    defs = _all_tool_defs()
    for name in WAVE_5_TOOLS:
        schema = defs[name]["inputSchema"]
        assert schema["type"] == "object"
        assert isinstance(schema.get("properties"), dict)


def test_wave5_enroll_requires_module_id():
    defs = _all_tool_defs()
    required = defs["enroll_in_module"]["inputSchema"].get("required") or []
    assert "module_id" in required


def test_wave5_descriptions_reference_v11_1():
    """v11.1 education-led GTM should be defensible in the tool
    description."""
    defs = _all_tool_defs()
    list_desc = defs["list_institute_modules"]["description"].lower()
    enroll_desc = defs["enroll_in_module"]["description"].lower()
    assert "education" in list_desc or "v11.1" in list_desc or "institute" in list_desc
    # Enrollment should surface v11.1 "learning and using are the same motion"
    assert "institute" in enroll_desc or "v11.1" in enroll_desc


def test_wave5_descriptions_flag_feature_pending():
    """Per v11.3 claim-validation rule — if the backing service isn't
    live, the tool description must say so honestly."""
    defs = _all_tool_defs()
    for name in WAVE_5_TOOLS:
        desc = defs[name]["description"].lower()
        assert "feature_pending" in desc or "pending" in desc or "development" in desc, \
            f"{name} must document its pending status"


@pytest.mark.asyncio
async def test_wave5_handlers_return_feature_pending_when_service_absent():
    """When InstituteService isn't installed, handlers must return a
    structured feature_pending response — not crash."""

    class _StubDB:
        async def execute(self, *a, **k):
            raise NotImplementedError("should not be called when service absent")

        async def commit(self):
            pass

    db = _StubDB()
    for name in WAVE_5_TOOLS:
        handler = tool_executor.TOOL_HANDLERS[name]
        args = {"module_id": "mod-1"} if name == "enroll_in_module" else {}
        result = await handler(args, db, "user-1")
        assert isinstance(result, dict)
        assert result.get("status") == "feature_pending", \
            f"{name} should return feature_pending when service missing"
        assert result.get("available") is False


def test_wave5_total_count_at_least_87():
    """11 + 23 + 8 + 2 + 40 + 3 = 87."""
    defs = _all_tool_defs()
    assert len(defs) >= 87
    assert len(tools.TOOL_SCOPES) >= 87
