"""Phase 3: tracked-entity memory — registration, resolution, prompt injection.

Covers the subsystem described in
docs/architecture/CONVERSATION_ORCHESTRATION_SPEC.md §3:
`_register_entity`, `_register_entities_from_results`,
`_resolve_reference`, `_recent_entities_prompt_section`,
`_entity_summary_for`, and `_TOOL_ENTITY_MAP`.

We simulate a `session` object with a mutable `context` dict. SQLAlchemy's
`flag_modified` is called inside `_register_entity` but guarded with a
try/except so these tests work on POPOs.
"""

from datetime import datetime
from types import SimpleNamespace

import pytest

from services.enhanced_conversation_manager import EnhancedConversationService


def _svc():
    return EnhancedConversationService.__new__(EnhancedConversationService)


def _session(context=None):
    """Plain-Python stand-in for a ConversationSession row.

    `flag_modified(session, "context")` will raise InvalidRequestError on
    this because it isn't attached to a session, but `_register_entity`
    swallows that exception.
    """
    return SimpleNamespace(id="sess-1", context=context if context is not None else {})


def _tool_result(tool_name, data, success=True):
    merged = {**data, "tool_name": tool_name}
    return SimpleNamespace(success=success, data=merged, error=None)


# ---------------------------------------------------------------------------
# _register_entity
# ---------------------------------------------------------------------------

def test_register_entity_inserts_new_row():
    svc = _svc()
    session = _session()
    svc._register_entity(
        session,
        entity_type="workflow",
        entity_id="wf-1",
        label="Invoice processor",
        summary="status: draft",
    )
    tracked = session.context["tracked_entities"]
    assert "workflow:wf-1" in tracked
    row = tracked["workflow:wf-1"]
    assert row["type"] == "workflow"
    assert row["id"] == "wf-1"
    assert row["label"] == "Invoice processor"
    assert row["summary"] == "status: draft"
    assert row["created_at"] == row["last_seen"]


def test_register_entity_refreshes_existing_row():
    svc = _svc()
    session = _session()
    svc._register_entity(session, entity_type="workflow", entity_id="wf-1",
                          label="First name", summary="status: draft")
    first_seen = session.context["tracked_entities"]["workflow:wf-1"]["last_seen"]

    svc._register_entity(session, entity_type="workflow", entity_id="wf-1",
                          label="Second name", summary="status: active")
    row = session.context["tracked_entities"]["workflow:wf-1"]
    assert row["label"] == "Second name"
    assert row["summary"] == "status: active"
    # last_seen should not be earlier than the first touch.
    assert row["last_seen"] >= first_seen


def test_register_entity_skips_empty_id():
    svc = _svc()
    session = _session()
    svc._register_entity(session, entity_type="workflow", entity_id="",
                          label="ghost", summary=None)
    assert session.context.get("tracked_entities", {}) == {}


def test_register_entity_evicts_lru_at_cap():
    svc = _svc()
    session = _session()
    # Fill to the cap with monotonic `last_seen` strings so we know
    # which one is oldest.
    for i in range(svc._MAX_TRACKED_ENTITIES):
        svc._register_entity(session, entity_type="workflow",
                              entity_id=f"wf-{i}", label=f"w{i}",
                              summary=None)
        # Force ordering by nudging last_seen forward.
        session.context["tracked_entities"][f"workflow:wf-{i}"]["last_seen"] = \
            f"2026-04-23T10:00:{i:02d}"

    assert len(session.context["tracked_entities"]) == svc._MAX_TRACKED_ENTITIES
    # Register one more — oldest should be evicted.
    svc._register_entity(session, entity_type="workflow", entity_id="wf-new",
                          label="new", summary=None)
    assert len(session.context["tracked_entities"]) == svc._MAX_TRACKED_ENTITIES
    assert "workflow:wf-0" not in session.context["tracked_entities"]
    assert "workflow:wf-new" in session.context["tracked_entities"]


# ---------------------------------------------------------------------------
# _register_entities_from_results — tool → entity mapping
# ---------------------------------------------------------------------------

def test_create_workflow_registers_workflow_entity():
    svc = _svc()
    session = _session()
    svc._register_entities_from_results(session, [_tool_result("create_workflow", {
        "workflow_id": "wf-42",
        "workflow_name": "KYC",
        "status": "draft",
    })])
    tracked = session.context["tracked_entities"]
    assert "workflow:wf-42" in tracked
    assert tracked["workflow:wf-42"]["label"] == "KYC"
    assert "draft" in (tracked["workflow:wf-42"]["summary"] or "")


def test_create_agent_registers_agent_entity():
    svc = _svc()
    session = _session()
    svc._register_entities_from_results(session, [_tool_result("create_agent", {
        "agent_id": "ag-7",
        "agent_name": "Triage bot",
        "status": "active",
    })])
    tracked = session.context["tracked_entities"]
    assert tracked["agent:ag-7"]["label"] == "Triage bot"
    assert "active" in (tracked["agent:ag-7"]["summary"] or "")


def test_list_workflows_registers_each_item():
    svc = _svc()
    session = _session()
    svc._register_entities_from_results(session, [_tool_result("list_workflows", {
        "workflows": [
            {"workflow_id": "wf-1", "workflow_name": "A", "status": "active"},
            {"workflow_id": "wf-2", "workflow_name": "B", "status": "paused"},
        ],
    })])
    tracked = session.context["tracked_entities"]
    assert "workflow:wf-1" in tracked
    assert "workflow:wf-2" in tracked
    assert "active" in (tracked["workflow:wf-1"]["summary"] or "")


def test_failed_result_does_not_register():
    svc = _svc()
    session = _session()
    svc._register_entities_from_results(session, [
        SimpleNamespace(success=False,
                        data={"tool_name": "create_workflow", "workflow_id": "wf-x"},
                        error="boom"),
    ])
    assert session.context.get("tracked_entities", {}) == {}


def test_unmapped_tool_does_not_register():
    svc = _svc()
    session = _session()
    svc._register_entities_from_results(session, [_tool_result("some_unknown_tool", {
        "foo": "bar",
    })])
    assert session.context.get("tracked_entities", {}) == {}


# ---------------------------------------------------------------------------
# _resolve_reference
# ---------------------------------------------------------------------------

def _seed(session, svc, *entities):
    """Seed a session with tracked entities in insertion order."""
    for e in entities:
        svc._register_entity(session, **e)


def test_resolve_by_substring_label_match():
    svc = _svc()
    session = _session()
    _seed(session, svc,
          {"entity_type": "workflow", "entity_id": "wf-a", "label": "Invoice processor", "summary": None},
          {"entity_type": "workflow", "entity_id": "wf-b", "label": "Loan approval",     "summary": None})
    got = svc._resolve_reference(session, "how's the invoice processor doing")
    assert got is not None
    assert got["id"] == "wf-a"


def test_resolve_by_pronoun_picks_most_recent():
    svc = _svc()
    session = _session()
    # Labels are disjoint from the query so only the pronoun path matches.
    _seed(session, svc,
          {"entity_type": "workflow", "entity_id": "wf-a", "label": "Onboarding pipeline", "summary": None},
          {"entity_type": "workflow", "entity_id": "wf-b", "label": "Settlement flow",     "summary": None})
    # Force wf-b to be the most recent.
    session.context["tracked_entities"]["workflow:wf-a"]["last_seen"] = "2026-04-23T10:00:00"
    session.context["tracked_entities"]["workflow:wf-b"]["last_seen"] = "2026-04-23T11:00:00"
    got = svc._resolve_reference(session, "rerun that one", entity_type="workflow")
    assert got is not None
    assert got["id"] == "wf-b"


def test_resolve_filters_by_type():
    svc = _svc()
    session = _session()
    _seed(session, svc,
          {"entity_type": "workflow", "entity_id": "wf-a", "label": "Triage", "summary": None},
          {"entity_type": "agent",    "entity_id": "ag-1", "label": "Triage", "summary": None})
    got = svc._resolve_reference(session, "triage", entity_type="agent")
    assert got is not None
    assert got["type"] == "agent"
    assert got["id"] == "ag-1"


def test_resolve_returns_none_when_no_match():
    svc = _svc()
    session = _session()
    _seed(session, svc,
          {"entity_type": "workflow", "entity_id": "wf-a", "label": "Invoice", "summary": None})
    assert svc._resolve_reference(session, "delete everything") is None


def test_resolve_returns_none_on_empty_tracked():
    svc = _svc()
    session = _session()
    assert svc._resolve_reference(session, "anything") is None


# ---------------------------------------------------------------------------
# _recent_entities_prompt_section
# ---------------------------------------------------------------------------

def test_prompt_section_empty_when_no_entities():
    svc = _svc()
    session = _session()
    assert svc._recent_entities_prompt_section(session) == ""


def test_prompt_section_lists_most_recent_first():
    svc = _svc()
    session = _session()
    _seed(session, svc,
          {"entity_type": "workflow", "entity_id": "wf-a", "label": "First",  "summary": "status: draft"},
          {"entity_type": "workflow", "entity_id": "wf-b", "label": "Second", "summary": "status: active"})
    session.context["tracked_entities"]["workflow:wf-a"]["last_seen"] = "2026-04-23T10:00:00"
    session.context["tracked_entities"]["workflow:wf-b"]["last_seen"] = "2026-04-23T11:00:00"
    section = svc._recent_entities_prompt_section(session)
    assert "Recent entities" in section
    # Second should appear before First.
    assert section.index("Second") < section.index("First")
    # Summary is included.
    assert "status: active" in section


def test_prompt_section_respects_limit():
    svc = _svc()
    session = _session()
    for i in range(10):
        svc._register_entity(session, entity_type="workflow",
                              entity_id=f"wf-{i}", label=f"w{i}",
                              summary=None)
    section = svc._recent_entities_prompt_section(session, limit=3)
    # Header + 3 rows = 4 lines (ignoring trailing newline).
    non_empty = [ln for ln in section.splitlines() if ln]
    assert len(non_empty) == 4


# ---------------------------------------------------------------------------
# _entity_summary_for
# ---------------------------------------------------------------------------

def test_entity_summary_for_workflow_create():
    svc = _svc()
    assert svc._entity_summary_for("create_workflow", {"status": "draft"}) \
        == "status: draft"


def test_entity_summary_for_risk_assessment_extracts_score():
    svc = _svc()
    s = svc._entity_summary_for("assess_risk", {"risk_assessment": {"risk_score": 0.73}})
    assert "0.73" in s


def test_entity_summary_for_unknown_tool_returns_none():
    svc = _svc()
    assert svc._entity_summary_for("mystery_tool", {}) is None
