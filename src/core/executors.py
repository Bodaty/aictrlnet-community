"""Dedicated thread pool for long-running synchronous work.

The asyncio default executor (used by asyncio.to_thread / run_in_executor(None, ...))
is shared by every short await in the process: bcrypt hashing, template preview
reads, credential file I/O. Minutes-long sync work — LLM SDK calls, agent
framework runs — must NOT share that pool: enough of them in flight and every
to_thread in the worker queues forever, which is exactly the templates/agents
list wedge of June 2026.

Long-running sync call sites go through run_blocking() instead, which uses a
separate bounded pool so saturation backpressures slow work against itself
without starving the rest of the worker.
"""

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Callable, Optional

_blocking_executor: Optional[ThreadPoolExecutor] = None


def get_blocking_executor() -> ThreadPoolExecutor:
    """Lazy singleton pool for slow sync work (per worker process)."""
    global _blocking_executor
    if _blocking_executor is None:
        max_workers = int(os.getenv("BLOCKING_EXECUTOR_THREADS", "8"))
        _blocking_executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="slow-sync"
        )
    return _blocking_executor


async def run_blocking(
    func: Callable[..., Any], *args: Any, timeout: Optional[float] = None, **kwargs: Any
) -> Any:
    """Run a slow sync callable on the dedicated pool, optionally bounded.

    timeout uses asyncio.wait_for, which abandons (does not kill) the thread on
    expiry — callers must also set network/SDK timeouts so abandoned threads
    eventually free themselves instead of accumulating in the bounded pool.
    """
    loop = asyncio.get_running_loop()
    future = loop.run_in_executor(get_blocking_executor(), partial(func, *args, **kwargs))
    if timeout is not None:
        return await asyncio.wait_for(future, timeout=timeout)
    return await future


def blocking_executor_stats() -> dict:
    """Stats for /health/detail; safe to call before first use."""
    if _blocking_executor is None:
        return {"initialized": False}
    return {
        "initialized": True,
        "max_workers": _blocking_executor._max_workers,
        "threads_alive": len(_blocking_executor._threads),
        "queue_depth": _blocking_executor._work_queue.qsize(),
    }
