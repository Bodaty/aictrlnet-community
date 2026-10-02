"""Spec §7.3 R4 (T4b) through the real Community turn.

A tool the user's autonomy level says must be confirmed becomes a proposal;
the next turn runs it on consent, drops it on "no", and any other turn
dismisses it. The model is a stub and tool execution is recorded, not run.
Real Postgres; no model (§8.2).
"""
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from core.database import get_session_maker
from llm.models import LLMToolResponse, ModelProvider, ModelTier, ToolCall, ToolResult
from models.conversation import ConversationSession
from models.user import User
from services import confirmation_gate
from services.conversation_manager import ConversationManagerService
from services.enhanced_conversation_manager import EnhancedConversationService
from services.tool_dispatcher import ToolDispatcher


class _ScriptedLLM:
    """Round 0 with tools asks for `calls`; everything else answers in text."""

    def __init__(self, calls):
        self.calls = calls
        self.rounds = []

    async def generate_with_tools_stream(self, **kwargs):
        tools = kwargs.get("tools") or []
        self.rounds.append({t.name for t in tools})
        if tools and len([r for r in self.rounds if r]) == 1 and self.calls:
            response = LLMToolResponse(
                model_used="stub-model", provider=ModelProvider.OLLAMA, tier=ModelTier.QUALITY,
                tool_calls=[ToolCall(id=f"c{i}", name=n, arguments=a) for i, (n, a) in enumerate(self.calls)],
            )
        else:
            response = LLMToolResponse(text="Done.", model_used="stub-model",
                                       provider=ModelProvider.OLLAMA, tier=ModelTier.QUALITY)
            yield {"type": "text_delta", "text": "Done."}
        yield {"type": "complete", "response": response}


@pytest.fixture(autouse=True)
async def _fresh_engine():
    """Each test runs on its own event loop; the app's global engine must not
    carry connections from the previous one."""
    import core.database as database
    yield
    if database._engine is not None:
        await database._engine.dispose()
    database._engine = None
    database._async_session_maker = None


@pytest.fixture
def invoked(monkeypatch):
    calls = []

    async def fake_invoke(self, tool_name, arguments, user_id, context=None):
        calls.append((tool_name, dict(arguments or {}), dict(context or {})))
        return ToolResult(success=True, data={"ok": True})

    monkeypatch.setattr(ToolDispatcher, "invoke", fake_invoke)
    monkeypatch.setenv("CONVERSATION_JOBS", "off")
    return calls


async def _new_session():
    async with get_session_maker()() as db:
        user = (await db.execute(select(User).where(User.email == "dev@aictrlnet.com"))).scalar_one()
        session = await ConversationManagerService(db).create_session(user_id=str(user.id))
        await db.commit()
        return session.id, str(user.id)


async def _turn(session_id, user_id, message, calls=(), level=40, monkeypatch=None):
    async def fixed_level(db, uid, tenant_id):
        return level

    monkeypatch.setattr(confirmation_gate, "resolve_turn_level", fixed_level)
    async with get_session_maker()() as db:
        service = EnhancedConversationService(db)
        llm = _ScriptedLLM(list(calls))
        service._enhanced_llm_service = llm
        service._user_memory_service = None
        route_context = service._route_context
        service._route_context = lambda *a: {**route_context(*a), "onboarding_active": False}
        events = [e async for e in service.process_message_v2(session_id, message, user_id)]
    return llm, events


def _response(events):
    return [e["data"] for e in events if e["event"] == "response"][0]


def _complete(events):
    return [e["data"] for e in events if e["event"] == "complete"][0]


async def _pending(session_id):
    async with get_session_maker()() as db:
        row = (await db.execute(select(ConversationSession).where(ConversationSession.id == session_id))).scalar_one()
        return (row.context or {}).get(confirmation_gate.PROPOSAL_KEY)


@pytest.mark.asyncio
async def test_a_gated_call_becomes_a_proposal_and_runs_once_on_yes(invoked, monkeypatch):
    sid, uid = await _new_session()
    _, events = await _turn(sid, uid, "Delete agent sales-bot", [("delete_agent", {"agent_id": "a-1"})],
                            monkeypatch=monkeypatch)
    assert invoked == []
    assert not [e for e in events if e["event"] == "tool_start"]
    response = _response(events)
    assert response["tools_executed"] == [] and "delete agent" in response["content"]
    assert [b["data"]["status"] for b in response["ui_blocks"] if b["type"] == "execution_preview"] == [
        "awaiting_confirmation"]
    assert (await _pending(sid))["tool"] == "delete_agent"

    llm, events = await _turn(sid, uid, "yes", monkeypatch=monkeypatch)
    assert [(n, a) for n, a, _ in invoked] == [("delete_agent", {"agent_id": "a-1"})]
    assert all(not offered for offered in llm.rounds)
    assert await _pending(sid) is None
    await _turn(sid, uid, "yes", monkeypatch=monkeypatch)
    assert len(invoked) == 1


@pytest.mark.asyncio
async def test_no_cancels(invoked, monkeypatch):
    sid, uid = await _new_session()
    await _turn(sid, uid, "Delete agent sales-bot", [("delete_agent", {"agent_id": "a-1"})],
                monkeypatch=monkeypatch)
    llm, events = await _turn(sid, uid, "no", monkeypatch=monkeypatch)
    assert invoked == [] and llm.rounds == []
    assert _response(events)["content"] == confirmation_gate.CANCELLED_ANSWER
    assert await _pending(sid) is None


@pytest.mark.asyncio
async def test_the_default_level_runs_builds(invoked, monkeypatch):
    sid, uid = await _new_session()
    await _turn(sid, uid, "Create a workflow named Invoice Intake",
                [("create_workflow", {"name": "Invoice Intake", "description": "intake invoices"})],
                level=55, monkeypatch=monkeypatch)
    assert [n for n, _, _ in invoked] == ["create_workflow"]
    assert await _pending(sid) is None
