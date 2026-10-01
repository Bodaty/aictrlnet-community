"""Tier 0: the terminal-event block every turn carries (spec §7.3 R8, §7.8)."""

import time
from types import SimpleNamespace

import pytest

from services import conversation_turn
from services.conversation_turn import TurnTrace, _TurnStats


@pytest.fixture(autouse=True)
def fresh_stats(monkeypatch):
    monkeypatch.setattr(conversation_turn, "_STATS", _TurnStats())


def _response(model="llama3.1:8b", provider="ollama", tier="quality", source="system_default"):
    return SimpleNamespace(
        model_used=model,
        provider=SimpleNamespace(value=provider),
        tier=SimpleNamespace(value=tier),
        metadata={"resolution_source": source},
    )


def test_complete_carries_every_field_the_spec_names():
    trace = TurnTrace("business", "ollama", session_id="s1")
    with trace.stage("route"):
        pass
    trace.set_route("unrouted", ["needs_tools"], offered=8)
    rnd = trace.start_round()
    trace.end_round_llm(rnd, _response())
    trace.record_tool(rnd, "get_help", time.perf_counter(), success=True)
    trace.record_tool(rnd, "list_workflows", time.perf_counter(), success=False)

    data = trace.complete_data("done", tools_total=2)

    assert data["status"] == "done"
    assert data["turn_id"] == trace.turn_id
    assert set(data["timings"]) == {
        "route_ms", "prompt_ms", "knowledge_ms", "rounds", "synthesis_ms",
        "persist_ms", "memory_ms", "total_ms",
    }
    assert data["timings"]["rounds"][0]["model"] == "llama3.1:8b"
    assert [t["status"] for t in data["timings"]["rounds"][0]["tools"]] == ["ok", "error"]
    assert data["route"] == {
        "mode": "unrouted", "reasons": ["needs_tools"], "offered": 8, "admitted_by_class": {},
    }
    assert data["model"] == {
        "provider": "ollama", "model": "llama3.1:8b", "tier": "quality",
        "resolution_source": "system_default",
    }
    assert data["budget"] == {
        "class": "local", "turn_s": 150, "exhausted": False,
        "degraded_reason": None, "fallback_from": None,
    }
    assert data["job_ids"] == []
    assert data["tools_total"] == 2


def test_chat_mode_reports_the_chat_budget():
    trace = TurnTrace("business", "vertex_ai")
    trace.set_route("chat", ["short"], offered=0)
    assert trace.complete_data()["budget"]["turn_s"] == 6


def test_observed_breach_is_exhausted_but_not_degraded(monkeypatch):
    trace = TurnTrace("business", "ollama")
    monkeypatch.setattr(trace, "elapsed_ms", lambda: 151_000)
    data = trace.complete_data()
    assert data["budget"]["exhausted"] is True
    assert data["budget"]["degraded_reason"] is None
    assert data["status"] == "done"


def test_degraded_reason_sets_degraded_status_but_not_over_clarification():
    trace = TurnTrace("business", "ollama")
    trace.degraded_reason = "synthesis_skipped"
    assert trace.complete_data()["status"] == "degraded"
    trace = TurnTrace("business", "ollama")
    trace.degraded_reason = "synthesis_skipped"
    assert trace.complete_data("clarification")["status"] == "clarification"


def test_error_carries_turn_id_and_counts_in_stats():
    trace = TurnTrace("community", "ollama")
    data = trace.error_data("boom", detail="tb")
    assert data == {"message": "boom", "turn_id": trace.turn_id, "detail": "tb"}
    TurnTrace("community", "ollama").complete_data()
    stats = conversation_turn.conversation_stats()
    assert stats["turns_15m"] == 2
    assert stats["terminal_error_rate"] == 0.5
    assert stats["p50_ms"] is not None


def test_stats_window_drops_turns_older_than_15_minutes(monkeypatch):
    stats = _TurnStats()
    clock = [1000.0]
    monkeypatch.setattr(conversation_turn.time, "monotonic", lambda: clock[0])
    stats.record(100, "complete", False)
    clock[0] += 15 * 60 + 1
    stats.record(300, "complete", True)
    snap = stats.snapshot()
    assert snap["turns_15m"] == 1
    assert snap["p95_ms"] == 300
    assert snap["budget_exhaustions"] == 1


def test_turn_log_line_is_emitted(caplog):
    trace = TurnTrace("business", "ollama", session_id="abc")
    with caplog.at_level("INFO", logger="services.conversation_turn"):
        trace.complete_data()
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("[turn] ")]
    assert len(lines) == 1
    assert f"turn_id={trace.turn_id}" in lines[0]
    assert "class=local" in lines[0] and "terminal=complete" in lines[0]


def test_a_turn_is_counted_once_whatever_ends_it():
    trace = TurnTrace("business", "ollama")
    trace.complete_data()
    trace.error_data("late failure")
    trace.abandon_if_unfinished()
    assert conversation_turn.conversation_stats()["turns_15m"] == 1


def test_an_unfinished_turn_is_recorded_as_abandoned_with_its_open_round_timed():
    trace = TurnTrace("business", "ollama")
    rnd = trace.start_round()
    trace.abandon_if_unfinished()
    stats = conversation_turn.conversation_stats()
    assert stats["abandoned"] == 1 and stats["terminal_error_rate"] == 0.0
    assert "_start" not in rnd and rnd["llm_ms"] >= 0


# ── T4a: deadlines, bounded rounds, terminal guard, after-turn work ─────

import asyncio
from types import SimpleNamespace as _NS

from services.conversation_budgets import ConversationBudgets
from services.conversation_turn import (
    RoundTimeout,
    StreamTerminalGuard,
    bounded_stream,
    run_after_turn,
)


def _tiny(provider_class="local"):
    return ConversationBudgets(
        provider_class=provider_class, route_ms=5, prompt_warm_ms=150, prompt_cold_ms=1500,
        knowledge_ms=2000, llm_round_s=1, tool_max_s=1, tool_hard_cap_s=60, job_ack_s=2,
        loop_s=4, turn_s=6, chat_turn_s=2, first_event_s=30, idle_s=3, client_grace_s=10,
    )


def test_observe_mode_sets_no_deadlines(monkeypatch):
    monkeypatch.delenv("CONVERSATION_BUDGETS", raising=False)
    trace = TurnTrace("business", "ollama")
    assert trace.enforce is False
    assert trace.round_timeout_s() is None
    assert trace.tool_timeout_s(_NS(tool_class="read", timeout_seconds=60)) is None
    assert trace.can_start_round() is True


def test_enforce_mode_bounds_rounds_and_tools(monkeypatch):
    monkeypatch.setenv("CONVERSATION_BUDGETS", "enforce")
    trace = TurnTrace("business", "ollama")
    trace.start_loop()
    assert 0 < trace.round_timeout_s() <= 45
    assert trace.tool_timeout_s(_NS(tool_class="read", timeout_seconds=60)) == 20
    assert trace.tool_timeout_s(_NS(tool_class="read", timeout_seconds=5)) == 5
    # Long-running tools have no job path until T5: no inline deadline yet.
    assert trace.tool_timeout_s(_NS(tool_class="long_running", timeout_seconds=300)) is None


def test_chat_round_is_capped_by_the_chat_budget(monkeypatch):
    monkeypatch.setenv("CONVERSATION_BUDGETS", "enforce")
    trace = TurnTrace("business", "ollama")
    trace.set_route("chat", ["pleasantry"], offered=0)
    assert trace.round_timeout_s() <= 15


def test_no_round_starts_without_budget_for_it(monkeypatch):
    monkeypatch.setenv("CONVERSATION_BUDGETS", "enforce")
    trace = TurnTrace("business", "ollama")
    trace.budgets = _tiny()
    trace.start_loop()
    assert trace.can_start_round()
    monkeypatch.setattr(trace, "remaining_s", lambda include_loop=True: 0.5)
    assert not trace.can_start_round()


def test_resolve_class_rebinds_budgets():
    trace = TurnTrace("business", "ollama")
    trace.resolve_class("vertex_ai")
    assert trace.budgets.provider_class == "cloud" and trace.complete_data()["budget"]["turn_s"] == 75


def test_a_rung_marks_the_turn_degraded_and_exhausted():
    trace = TurnTrace("business", "ollama")
    trace.degrade("tool_timeout")
    trace.degrade("synthesis_skipped")  # the reason names the first rung...
    data = trace.complete_data()
    assert data["status"] == "degraded"
    assert data["budget"]["degraded_reason"] == "tool_timeout"
    assert data["budget"]["exhausted"] is True
    note = trace.degraded_note()  # ...the answer names every rung taken
    assert "too long and was stopped" in note and "plain summary" in note


def test_synthesis_may_use_the_slack_after_the_loop(monkeypatch):
    monkeypatch.setenv("CONVERSATION_BUDGETS", "enforce")
    trace = TurnTrace("business", "ollama")
    trace.budgets = _tiny()  # loop 4, turn 6, round 1
    trace.start_loop()
    trace._loop_deadline = trace._t0  # the loop is spent
    assert not trace.can_start_round()
    assert trace.can_start_round(synthesis=True)


_closed = []


async def _slow_stream(delay):
    try:
        yield {"type": "text_delta", "text": "partial"}
        await asyncio.sleep(delay)
        yield {"type": "complete"}
    finally:
        _closed.append(True)


async def test_bounded_stream_stops_a_slow_round_and_closes_it():
    _closed.clear()
    seen = []
    with pytest.raises(RoundTimeout):
        async for event in bounded_stream(_slow_stream(5), 0.2):
            seen.append(event)
    assert seen == [{"type": "text_delta", "text": "partial"}]
    assert _closed == [True]  # the provider stream was shut, not left to GC


async def test_bounded_stream_without_deadline_passes_through():
    events = [e async for e in bounded_stream(_slow_stream(0), None)]
    assert [e["type"] for e in events] == ["text_delta", "complete"]


def test_stream_guard_never_errors_after_response():
    guard = StreamTerminalGuard()
    guard.saw({"event": "context_ready", "data": {"turn_id": "t-1"}})
    assert guard.failure_event(RuntimeError("x")) == ("error", {"message": "x", "turn_id": "t-1"})
    guard.saw({"event": "response", "data": {}})
    name, data = guard.failure_event(RuntimeError("x"))
    assert name == "complete" and data["status"] == "degraded" and data["turn_id"] == "t-1"
    guard.saw({"event": "complete", "data": {}})
    assert guard.failure_event(RuntimeError("x")) is None


async def test_after_turn_work_waits_for_turns_in_flight_on_local(monkeypatch):
    # Traces earlier tests left unfinished would count as in flight; the turn
    # wrapper always finishes its trace in production.
    monkeypatch.setattr(conversation_turn, "_in_flight", 0)
    ran = []
    trace = TurnTrace("business", "ollama")  # one turn in flight

    async def work():
        ran.append(True)

    run_after_turn("probe", "local", work, wait_s=5)
    await asyncio.sleep(0.6)
    assert ran == []
    trace.complete_data()  # the turn ends
    await asyncio.sleep(0.8)
    assert ran == [True]


def test_a_chat_turn_can_start_its_round_with_the_real_budgets(monkeypatch):
    # chat_turn_s (15) is below llm_round_s (45): requiring a full round
    # before starting would refuse every chat turn.
    monkeypatch.setenv("CONVERSATION_BUDGETS", "enforce")
    trace = TurnTrace("business", "ollama")
    trace.set_route("chat", ["pleasantry"], offered=0)
    trace.start_loop()
    assert trace.round_floor_s() == 7.5
    assert trace.can_start_round()
