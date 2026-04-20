"""Phase 1: Typed UI Blocks — unit tests for _build_ui_blocks + schema."""

import json
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from schemas.conversation import (
    UIBlock,
    UIBlockAction,
    ConversationResponse,
    ConversationMessageResponse,
    ui_block_to_text,
)
from services.enhanced_conversation_manager import EnhancedConversationService


def _tool_result(tool_name: str, data: dict, success: bool = True):
    """Build a minimal ToolResult-like object for the mapper."""
    merged = {**data, "tool_name": tool_name}
    return SimpleNamespace(success=success, data=merged, error=None)


def _service():
    """Bare-bones service instance — mapper methods don't touch the DB."""
    # __init__ is expensive (knowledge service, prompt assembler, etc.) so
    # bypass it; the mapper methods only need `self`.
    svc = EnhancedConversationService.__new__(EnhancedConversationService)
    return svc


# ---------------------------------------------------------------------------
# Block mapping tests
# ---------------------------------------------------------------------------

def test_create_workflow_yields_workflow_card():
    svc = _service()
    blocks = svc._build_ui_blocks(
        [_tool_result("create_workflow", {
            "workflow_id": "wf-123",
            "workflow_name": "KYC Review",
            "status": "draft",
        })],
        session_context={},
    )
    assert len(blocks) == 1
    b = blocks[0]
    assert b["type"] == "workflow_card"
    assert b["data"]["id"] == "wf-123"
    assert b["data"]["name"] == "KYC Review"
    assert b["data"]["status"] == "draft"
    # Primary action must navigate — link, don't replicate.
    opens = [a for a in b["actions"] if a["verb"] == "open"]
    assert len(opens) == 1
    assert opens[0]["primary"] is True
    # Draft → edit page
    assert opens[0]["target"] == "/workflows/wf-123/edit"


def test_list_workflows_yields_summary_and_cards():
    svc = _service()
    blocks = svc._build_ui_blocks(
        [_tool_result("list_workflows", {
            "workflows": [
                {"id": "wf-1", "name": "A", "status": "active"},
                {"id": "wf-2", "name": "B", "status": "paused"},
                {"id": "wf-3", "name": "C", "status": "active"},
            ],
        })],
        session_context={},
    )
    # 1 summary + 3 cards
    assert len(blocks) == 4
    assert blocks[0]["type"] == "text"
    assert "3 workflow" in blocks[0]["data"]["content"]
    for card in blocks[1:]:
        assert card["type"] == "workflow_card"


def test_execute_workflow_yields_execution_preview():
    svc = _service()
    blocks = svc._build_ui_blocks(
        [_tool_result("execute_workflow", {
            "execution_id": "exec-42",
            "workflow_id": "wf-42",
            "workflow_name": "KYC Review",
            "tools_executed": [
                {"name": "fetch_data", "status": "success"},
                {"name": "score_risk", "status": "running"},
            ],
        })],
        session_context={},
    )
    assert len(blocks) == 1
    b = blocks[0]
    assert b["type"] == "execution_preview"
    assert b["data"]["execution_id"] == "exec-42"
    assert len(b["data"]["tools"]) == 2
    # View run action navigates to /runs
    assert any(a["target"] == "/runs" for a in b["actions"])


def test_failed_tool_result_yields_no_blocks():
    svc = _service()
    blocks = svc._build_ui_blocks(
        [SimpleNamespace(success=False, data={"tool_name": "create_workflow"}, error="boom")],
        session_context={},
    )
    assert blocks == []


def test_unmapped_tool_yields_no_blocks():
    svc = _service()
    blocks = svc._build_ui_blocks(
        [_tool_result("some_unknown_tool", {"foo": "bar"})],
        session_context={},
    )
    assert blocks == []


def test_empty_tool_results_yields_empty_list():
    svc = _service()
    assert svc._build_ui_blocks([], {}) == []
    assert svc._build_ui_blocks(None, {}) == []


def test_workflow_card_without_id_falls_back_to_hub():
    svc = _service()
    blocks = svc._build_ui_blocks(
        [_tool_result("create_workflow", {"workflow_name": "No ID Workflow"})],
        session_context={},
    )
    assert len(blocks) == 1
    opens = [a for a in blocks[0]["actions"] if a["verb"] == "open"]
    assert opens[0]["target"] == "/workflows-hub"


def test_list_agents_yields_agent_cards():
    svc = _service()
    blocks = svc._build_ui_blocks(
        [_tool_result("list_agents", {
            "agents": [
                {"id": "ag-1", "name": "Compliance Bot", "status": "active"},
                {"id": "ag-2", "name": "Risk Bot", "status": "paused"},
            ],
        })],
        session_context={},
    )
    assert len(blocks) == 2
    assert all(b["type"] == "agent_card" for b in blocks)
    assert blocks[0]["data"]["id"] == "ag-1"


# ---------------------------------------------------------------------------
# Schema backwards-compat tests
# ---------------------------------------------------------------------------

def test_conversation_response_serializes_without_ui_blocks():
    """Backwards compat: ConversationResponse must still serialize cleanly
    when ui_blocks is empty (the default)."""
    from uuid import uuid4
    from datetime import datetime

    msg = ConversationMessageResponse(
        id=uuid4(),
        session_id=uuid4(),
        role="assistant",
        content="Hello",
        timestamp=datetime.utcnow(),
    )
    resp = ConversationResponse(
        session_id=uuid4(),
        message=msg,
        state="greeting",
    )
    payload = resp.model_dump()
    assert payload["ui_blocks"] == []
    # Must round-trip through JSON cleanly
    json.dumps(payload, default=str)


def test_ui_block_and_action_schemas_roundtrip():
    block = UIBlock(
        type="workflow_card",
        data={"id": "wf-1", "name": "X", "status": "draft"},
        actions=[
            UIBlockAction(label="Open", verb="open", primary=True, target="/workflows/wf-1/edit"),
            UIBlockAction(label="Cancel", verb="cancel", destructive=True),
        ],
    )
    as_dict = block.model_dump()
    assert as_dict["type"] == "workflow_card"
    assert as_dict["actions"][0]["primary"] is True
    assert as_dict["actions"][1]["destructive"] is True
    # Re-hydrate
    rebuilt = UIBlock(**as_dict)
    assert rebuilt.actions[0].target == "/workflows/wf-1/edit"


def test_ui_block_to_text_for_all_types():
    """Channel-agnostic fallback: every block type must degrade to a text line."""
    cases = [
        (UIBlock(type="text", data={"content": "hi"}), "hi"),
        (UIBlock(type="workflow_card", data={"id": "wf-1", "name": "X", "status": "active"}),
         "Workflow 'X'"),
        (UIBlock(type="approval_form", data={"title": "Deploy", "urgency": "high"}),
         "Approval needed: Deploy"),
        (UIBlock(type="risk_card", data={"score": 72, "top_finding": "stale data"}),
         "Risk score 72"),
        (UIBlock(type="execution_preview", data={"tools": [{"name": "a"}, {"name": "b"}]}),
         "Running 2 tool"),
        (UIBlock(type="agent_card", data={"id": "ag-1", "name": "Bot", "status": "active"}),
         "Agent 'Bot'"),
        (UIBlock(type="policy_card", data={"name": "SOC2", "scope": "global"}),
         "Policy 'SOC2'"),
        (UIBlock(type="resource_widget", data={"name": "pool-a", "utilization_pct": 42}),
         "Pool 'pool-a'"),
        (UIBlock(type="governance_report", data={"title": "Q1", "headline_metric": "95%"}),
         "Q1: 95%"),
        (UIBlock(type="nav_hint", data={"label": "Open", "target": "/x"}),
         "Open"),
        (UIBlock(type="reasoning_steps", data={"steps": [1, 2]}),
         "Reasoning"),
    ]
    for block, expected_substr in cases:
        text = ui_block_to_text(block)
        assert expected_substr in text, f"{block.type} → {text!r} missing {expected_substr!r}"
