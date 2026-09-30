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
