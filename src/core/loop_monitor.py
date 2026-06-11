"""Event-loop health monitor.

A 1s heartbeat task records how late it wakes up; sustained lag means the
event loop is blocked by synchronous work — the precursor of the gunicorn
WORKER TIMEOUT wedge (June 2026). Surfaced via /api/v1/health/detail so the
degraded state is observable before workers start dying.
"""

import asyncio
import time
from typing import Optional

_HEARTBEAT_INTERVAL = 1.0
_state = {
    "last_lag_ms": 0.0,
    "max_lag_ms": 0.0,
    "max_lag_at": None,
    "beats": 0,
}
_task: Optional[asyncio.Task] = None


async def _heartbeat() -> None:
    while True:
        before = time.monotonic()
        await asyncio.sleep(_HEARTBEAT_INTERVAL)
        lag_ms = max(0.0, (time.monotonic() - before - _HEARTBEAT_INTERVAL) * 1000)
        _state["last_lag_ms"] = round(lag_ms, 1)
        _state["beats"] += 1
        if lag_ms > _state["max_lag_ms"]:
            _state["max_lag_ms"] = round(lag_ms, 1)
            _state["max_lag_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")


def start_loop_monitor() -> None:
    """Start the heartbeat task; call from app lifespan startup."""
    global _task
    if _task is None or _task.done():
        _task = asyncio.get_running_loop().create_task(_heartbeat())


def loop_monitor_stats() -> dict:
    return {
        "running": _task is not None and not _task.done(),
        "last_lag_ms": _state["last_lag_ms"],
        "max_lag_ms": _state["max_lag_ms"],
        "max_lag_at": _state["max_lag_at"],
        "beats": _state["beats"],
    }


def default_executor_stats() -> dict:
    """Best-effort introspection of the loop's default ThreadPoolExecutor.

    Uses CPython implementation details; never raises.
    """
    try:
        loop = asyncio.get_running_loop()
        ex = getattr(loop, "_default_executor", None)
        if ex is None:
            return {"initialized": False}
        return {
            "initialized": True,
            "max_workers": ex._max_workers,
            "threads_alive": len(ex._threads),
            "queue_depth": ex._work_queue.qsize(),
        }
    except Exception:
        return {"available": False}
