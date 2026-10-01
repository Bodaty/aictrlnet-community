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

import asyncio
import contextlib
import logging
import os
import threading
import time
import uuid
import weakref
from collections import deque
from contextlib import contextmanager
from typing import Any, AsyncIterator, Awaitable, Callable, Deque, Dict, List, Optional, Set, Tuple

from services.conversation_budgets import (
    ENFORCE,
    LOCAL,
    ConversationBudgets,
    budgets_for,
    enforcement_mode,
    provider_class_for,
)

logger = logging.getLogger(__name__)

_WINDOW_S = 15 * 60


def _ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


class RoundTimeout(TimeoutError):
    """A model round ran past its deadline (§7.2 `llm_round_s` or what the loop/turn has left)."""


# Turns in flight in this worker; post-turn work waits for zero on the local
# class, where one model serves every request (spec §7.3 R6).
_in_flight = 0
_background: Set[asyncio.Task] = set()
_after_turn_locks: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore]" = weakref.WeakKeyDictionary()
_AFTER_TURN_QUEUE_MAX = 8

# The degradation sentence appended to an answer that took a lower rung (R9).
DEGRADED_NOTES = {
    "no_budget_for_round": "I stopped before another step to stay within the time limit for this answer.",
    "round_timeout": "The model took too long on one step, so I stopped there.",
    "synthesis_skipped": "I ran out of time to write this up, so here is a plain summary of what was done.",
    "synthesis_timeout": "The model took too long to write this up, so here is a plain summary of what was done.",
    "tool_timeout": "One step took too long and was stopped; if it was making a change, check it before retrying.",
    "nothing_ran": "I wasn't able to get to this within the time limit — nothing was done.",
}


class TurnTrace:
    def __init__(self, edition: str, provider: str, session_id: Any = None, count_in_flight: bool = True):
        global _in_flight
        self.turn_id = str(uuid.uuid4())
        self.edition = edition
        self.session_id = str(session_id) if session_id is not None else None
        self.budgets: ConversationBudgets = budgets_for(provider)
        self.enforce = enforcement_mode() == ENFORCE
        self._t0 = time.perf_counter()
        self._loop_deadline: Optional[float] = None
        self._counted = count_in_flight
        if count_in_flight:
            _in_flight += 1
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
        self.exhausted_by_rung = False
        self.rungs: List[str] = []
        self.response_sent = False
        self.finished = False

    # ── budget class and deadlines (spec §7.1, §7.2, R5) ────────────────
    def resolve_class(self, provider: Optional[str]) -> None:
        """Rebind budgets to the class of the model the resolver picked (§7.1).

        Called once, before the first model round. A later provider fallback
        keeps `turn_s` and only records `fallback_from` (see note_fallback).
        """
        if provider:
            self.budgets = budgets_for(provider)

    def note_fallback(self, provider: Optional[str]) -> None:
        if (provider and self.fallback_from is None
                and provider_class_for(provider) != self.budgets.provider_class):
            self.fallback_from = self.budgets.provider_class

    def _turn_deadline(self) -> float:
        return self._t0 + self.turn_budget_s()

    def start_loop(self) -> None:
        self._loop_deadline = time.perf_counter() + self.budgets.loop_s

    def remaining_s(self, include_loop: bool = True) -> float:
        """Seconds left: the turn deadline, and — for tool rounds and tools —
        the loop deadline too. Synthesis is bounded by the turn only: the
        `turn_s − loop_s` slack exists for it (§7.2)."""
        deadline = self._turn_deadline()
        if include_loop and self._loop_deadline is not None:
            deadline = min(deadline, self._loop_deadline)
        return deadline - time.perf_counter()

    def round_floor_s(self) -> float:
        """Least time a round needs before it is worth starting (spec §7.4):
        `llm_round_s`, or half the turn budget when that is smaller — a chat
        turn's whole budget (15 s local) is below one tool round (45 s)."""
        return min(self.budgets.llm_round_s, self.turn_budget_s() / 2)

    def can_start_round(self, synthesis: bool = False) -> bool:
        """R5: a round that cannot finish inside the remaining budget is not started."""
        if not self.enforce:
            return True
        return self.remaining_s(include_loop=not synthesis) >= self.round_floor_s()

    def round_timeout_s(self, synthesis: bool = False) -> Optional[float]:
        """Deadline for the next model round, or None when not enforcing."""
        if not self.enforce:
            return None
        limit = self.budgets.llm_round_s
        if self.route.get("mode") == "chat":
            limit = min(limit, self.budgets.chat_turn_s)
        return max(0.0, min(limit, self.remaining_s(include_loop=not synthesis)))

    def tool_timeout_s(self, tool) -> Optional[float]:
        """R2 effective limit for an inline tool, or None (not enforcing, or a
        long-running tool that has no job path until T5)."""
        if not self.enforce or getattr(tool, "tool_class", None) == "long_running":
            return None
        declared = getattr(tool, "timeout_seconds", None) or self.budgets.tool_max_s
        return max(0.0, min(declared, self.budgets.tool_max_s, self.remaining_s()))

    def degrade(self, reason: str) -> None:
        """Take a rung of the §7.4 ladder. `degraded_reason` names the first
        rung; every rung taken is kept, so the answer can say all of them."""
        if self.degraded_reason is None:
            self.degraded_reason = reason
        if reason not in self.rungs:
            self.rungs.append(reason)
        self.exhausted_by_rung = True

    def degraded_note(self) -> str:
        if "nothing_ran" in self.rungs:  # the other rungs explain nothing the user can see
            return DEGRADED_NOTES["nothing_ran"]
        return " ".join(DEGRADED_NOTES[r] for r in self.rungs if r in DEGRADED_NOTES)

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
        self.note_fallback(self.model["provider"])

    def record_tool(self, rnd: Optional[Dict[str, Any]], name: str, start: float,
                    success: bool, status: Optional[str] = None) -> None:
        entry = {"name": name, "ms": _ms(start), "status": status or ("ok" if success else "error")}
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
        exhausted = total_ms > turn_budget_s * 1000 or self.exhausted_by_rung
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
        exhausted = total_ms > self.turn_budget_s() * 1000 or self.exhausted_by_rung
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
        global _in_flight
        if self.finished:
            return
        self.finished = True
        if self._counted:
            _in_flight = max(0, _in_flight - 1)
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


async def bounded_stream(stream, timeout_s: Optional[float]) -> AsyncIterator[Any]:
    """Iterate a model stream under one deadline for the whole round (R5).

    The stream is closed on exit, so an abandoned generation stops at the
    provider instead of holding the local model's single slot. Raises
    RoundTimeout when the deadline passes; None means no deadline.
    """
    async with contextlib.aclosing(stream):
        if timeout_s is None:
            async for event in stream:
                yield event
            return
        # One deadline for the whole round, applied to each step of the stream.
        # Not `asyncio.timeout` around the loop: it would also fire while this
        # generator is suspended at a yield and cancel the consumer's await.
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise RoundTimeout(f"model round exceeded {timeout_s:.0f}s")
            try:
                event = await asyncio.wait_for(stream.__anext__(), remaining)
            except StopAsyncIteration:
                return
            except asyncio.TimeoutError as exc:
                raise RoundTimeout(f"model round exceeded {timeout_s:.0f}s") from exc
            yield event


def run_after_turn(
    name: str, provider_class: str, work: Callable[[], Awaitable[None]],
    wait_s: float = 120.0, work_timeout_s: float = 120.0,
) -> bool:
    """Run `work` after the terminal event, off the turn (spec §7.3 R6).

    One at a time per worker process — gunicorn runs several, so this is not
    a cross-worker limit. On the `local` class it first waits (up to `wait_s`)
    until no turn is in flight in this worker, so it does not queue ahead of a
    user's next model round there. The work itself is bounded by
    `work_timeout_s`; at most `_AFTER_TURN_QUEUE_MAX` items wait per worker and
    a new one is dropped (and logged) when the queue is full. Failures are
    logged, never raised. Returns whether the work was queued.
    """
    if len(_background) >= _AFTER_TURN_QUEUE_MAX:
        logger.warning("[after-turn] queue full (%d); dropped %s", len(_background), name)
        return False

    async def runner() -> None:
        loop = asyncio.get_running_loop()
        lock = _after_turn_locks.get(loop)
        if lock is None:
            lock = _after_turn_locks[loop] = asyncio.Semaphore(1)
        try:
            async with lock:
                if provider_class == LOCAL:
                    waited = 0.0
                    while _in_flight > 0 and waited < wait_s:
                        await asyncio.sleep(0.5)
                        waited += 0.5
                await asyncio.wait_for(work(), timeout=work_timeout_s)
        except asyncio.CancelledError:
            logger.info("[after-turn] %s cancelled (worker shutting down)", name)
            raise
        except Exception as exc:  # post-turn work must never surface to a user
            logger.warning("[after-turn] %s failed: %s", name, exc)

    task = asyncio.get_running_loop().create_task(runner(), name=f"after-turn:{name}")
    _background.add(task)
    task.add_done_callback(_background.discard)
    return True


class StreamTerminalGuard:
    """R1 for an SSE endpoint wrapped around a turn.

    Watches the events it forwards; if the endpoint itself fails (e.g. while
    serialising an event), `failure_event` says what to send: nothing once a
    terminal event went out, `complete` (degraded) once `response` did, else
    `error` — with the turn id when one was announced, and no traceback.
    """

    def __init__(self) -> None:
        self.turn_id: Optional[str] = None
        self.response_sent = False
        self.terminal_sent = False

    def saw(self, event: Dict[str, Any]) -> None:
        name = event.get("event")
        data = event.get("data") or {}
        if isinstance(data, dict) and data.get("turn_id"):
            self.turn_id = data["turn_id"]
        if name == "response":
            self.response_sent = True
        elif name in ("complete", "error"):
            self.terminal_sent = True

    def failure_event(self, exc: BaseException) -> Optional[Tuple[str, Dict[str, Any]]]:
        if self.terminal_sent:
            return None
        if self.response_sent:
            return "complete", {
                "status": "degraded", "turn_id": self.turn_id,
                "budget": {"exhausted": True, "degraded_reason": "post_response_failure"},
            }
        return "error", {"message": str(exc), "turn_id": self.turn_id}
