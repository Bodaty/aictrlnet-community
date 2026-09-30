"""Per-turn trace for the conversation pipeline (spec §7.3 R8, §7.8).

A `TurnTrace` is created at the top of a turn, records how long each stage
took, and builds the `turn_id`/`timings`/`route`/`model`/`budget` block that
rides on the terminal event. It also logs one `[turn]` key=value line and feeds
the per-worker 15-minute window that `/health/detail` reports.

Observe-only (T2): nothing here cuts a turn short. `budget.exhausted` records
an observed breach of the turn budget (`chat_turn_s` in chat mode, else
`turn_s`); `degraded_reason` stays null until enforcement (T4) takes a rung of
the §7.4 ladder.
"""

import logging
import os
import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager
from typing import Any, Deque, Dict, List, Optional, Tuple

from services.conversation_budgets import ConversationBudgets, budgets_for

logger = logging.getLogger(__name__)

_WINDOW_S = 15 * 60


def _ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


class TurnTrace:
    def __init__(self, edition: str, provider: str, session_id: Any = None):
        self.turn_id = str(uuid.uuid4())
        self.edition = edition
        self.session_id = str(session_id) if session_id is not None else None
        self.budgets: ConversationBudgets = budgets_for(provider)
        self._t0 = time.perf_counter()
        self.stages: Dict[str, int] = {}
        self.rounds: List[Dict[str, Any]] = []
        self.route: Dict[str, Any] = {
            "mode": "unrouted", "reasons": [], "offered": 0, "admitted_by_class": {},
        }
        self.model: Dict[str, Any] = {
            "provider": None, "model": None, "tier": None, "resolution_source": None,
        }
        self.job_ids: List[str] = []
        self.degraded_reason: Optional[str] = None
        self.fallback_from: Optional[str] = None
        self.finished = False

    # ── stages ──────────────────────────────────────────────────────────
    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = self.stages.get(name, 0) + _ms(start)

    def set_route(self, mode: str, reasons: List[str], offered: int,
                  admitted_by_class: Optional[Dict[str, int]] = None) -> None:
        self.route = {
            "mode": mode,
            "reasons": list(reasons),
            "offered": offered,
            "admitted_by_class": dict(admitted_by_class or {}),
        }

    # ── LLM rounds and tools ────────────────────────────────────────────
    def start_round(self) -> Dict[str, Any]:
        rnd = {"llm_ms": 0, "model": None, "tools": [], "_start": time.perf_counter()}
        self.rounds.append(rnd)
        return rnd

    def end_round_llm(self, rnd: Dict[str, Any], response: Any) -> None:
        rnd["llm_ms"] = _ms(rnd.pop("_start"))
        self.observe_model(response)
        rnd["model"] = self.model.get("model")

    def observe_model(self, response: Any) -> None:
        if response is None:
            return
        provider = getattr(response, "provider", None)
        tier = getattr(response, "tier", None)
        metadata = getattr(response, "metadata", None) or {}
        self.model = {
            "provider": getattr(provider, "value", provider),
            "model": getattr(response, "model_used", None) or None,
            "tier": getattr(tier, "value", tier),
            "resolution_source": metadata.get("resolution_source"),
        }

    def record_tool(self, rnd: Optional[Dict[str, Any]], name: str, start: float,
                    success: bool) -> None:
        entry = {"name": name, "ms": _ms(start), "status": "ok" if success else "error"}
        if rnd is None:
            rnd = self.rounds[-1] if self.rounds else self.start_round()
        rnd["tools"].append(entry)

    # ── terminal events ─────────────────────────────────────────────────
    def budget_block(self) -> Dict[str, Any]:
        return {"class": self.budgets.provider_class, **self.budgets.as_dict()}

    def elapsed_ms(self) -> int:
        return _ms(self._t0)

    def turn_budget_s(self) -> int:
        if self.route.get("mode") == "chat":
            return self.budgets.chat_turn_s
        return self.budgets.turn_s

    def _timings(self, total_ms: int) -> Dict[str, Any]:
        rounds = [
            {"llm_ms": r["llm_ms"], "model": r["model"], "tools": r["tools"]}
            for r in self.rounds
        ]
        return {
            "route_ms": self.stages.get("route", 0),
            "prompt_ms": self.stages.get("prompt", 0),
            "knowledge_ms": self.stages.get("knowledge", 0),
            "rounds": rounds,
            "synthesis_ms": self.stages.get("synthesis", 0),
            "persist_ms": self.stages.get("persist", 0),
            "memory_ms": self.stages.get("memory", 0),
            "total_ms": total_ms,
        }

    def complete_data(self, status: str = "done", **extra: Any) -> Dict[str, Any]:
        total_ms = self.elapsed_ms()
        turn_budget_s = self.turn_budget_s()
        exhausted = total_ms > turn_budget_s * 1000
        if status not in ("clarification", "degraded"):
            status = "degraded" if self.degraded_reason else "done"
        data = {
            "status": status,
            "turn_id": self.turn_id,
            "timings": self._timings(total_ms),
            "route": self.route,
            "model": self.model,
            "budget": {
                "class": self.budgets.provider_class,
                "turn_s": turn_budget_s,
                "exhausted": exhausted,
                "degraded_reason": self.degraded_reason,
                "fallback_from": self.fallback_from,
            },
            "job_ids": list(self.job_ids),
        }
        data.update(extra)
        self._finish("complete", status, total_ms, exhausted)
        return data

    def error_data(self, message: str, **extra: Any) -> Dict[str, Any]:
        total_ms = self.elapsed_ms()
        exhausted = total_ms > self.turn_budget_s() * 1000
        data = {"message": message, "turn_id": self.turn_id}
        data.update(extra)
        self._finish("error", "error", total_ms, exhausted)
        return data

    def abandon_if_unfinished(self) -> None:
        """Record a turn that ended without a terminal event (client gone, task cancelled)."""
        if not self.finished:
            total_ms = self.elapsed_ms()
            self._finish("abandoned", "abandoned", total_ms, total_ms > self.turn_budget_s() * 1000)

    def _finish(self, terminal: str, status: str, total_ms: int, exhausted: bool) -> None:
        if self.finished:
            return
        self.finished = True
        for rnd in self.rounds:
            if "_start" in rnd:  # a round that raised or was cancelled mid-stream
                rnd["llm_ms"] = _ms(rnd.pop("_start"))
        tools = [t for r in self.rounds for t in r["tools"]]
        llm_ms = sum(r["llm_ms"] for r in self.rounds)
        logger.info(
            "[turn] turn_id=%s edition=%s session=%s terminal=%s status=%s mode=%s "
            "class=%s provider=%s model=%s total_ms=%d route_ms=%d prompt_ms=%d "
            "knowledge_ms=%d rounds=%d llm_ms=%d tools=%d tool_ms=%d synthesis_ms=%d "
            "persist_ms=%d memory_ms=%d offered=%d exhausted=%s",
            self.turn_id, self.edition, self.session_id, terminal, status,
            self.route.get("mode"), self.budgets.provider_class,
            self.model.get("provider"), self.model.get("model"), total_ms,
            self.stages.get("route", 0), self.stages.get("prompt", 0),
            self.stages.get("knowledge", 0), len(self.rounds), llm_ms, len(tools),
            sum(t["ms"] for t in tools), self.stages.get("synthesis", 0),
            self.stages.get("persist", 0), self.stages.get("memory", 0),
            self.route.get("offered", 0), exhausted,
        )
        _STATS.record(total_ms, terminal, exhausted)


class _TurnStats:
    """Per-worker sliding window of finished turns (like the rest of /health/detail)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._turns: Deque[Tuple[float, int, str, bool]] = deque()

    def record(self, total_ms: int, terminal: str, exhausted: bool) -> None:
        now = time.monotonic()
        with self._lock:
            self._turns.append((now, total_ms, terminal, exhausted))
            self._evict(now)

    def _evict(self, now: float) -> None:
        while self._turns and now - self._turns[0][0] > _WINDOW_S:
            self._turns.popleft()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            self._evict(time.monotonic())
            turns = list(self._turns)
        durations = sorted(t[1] for t in turns)
        n = len(durations)

        def pct(p: float) -> Optional[int]:
            if not n:
                return None
            return durations[min(n - 1, int(round(p * (n - 1))))]

        return {
            "pid": os.getpid(),
            "window_s": _WINDOW_S,
            "turns_15m": n,
            "p50_ms": pct(0.50),
            "p95_ms": pct(0.95),
            "budget_exhaustions": sum(1 for t in turns if t[3]),
            "terminal_error_rate": (sum(1 for t in turns if t[2] == "error") / n) if n else 0.0,
            "abandoned": sum(1 for t in turns if t[2] == "abandoned"),
        }


_STATS = _TurnStats()


def conversation_stats() -> Dict[str, Any]:
    return _STATS.snapshot()
