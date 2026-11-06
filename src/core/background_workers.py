"""The pollers a service process starts on its own.

The scheduler, sweeps, dispatchers and purges run in every edition process and
act on every tenant. On a stack running the test gate they would race the
tests for the tests' own rows, so they start only while
BACKGROUND_WORKERS_ENABLED is on (default). Event subscribers are not pollers:
they react to events a caller raised and stay on.
"""

import inspect
import logging
from typing import Awaitable, Callable, List, Union

from core.config import get_settings

logger = logging.getLogger(__name__)

_started: List[str] = []


def background_workers_enabled() -> bool:
    return get_settings().BACKGROUND_WORKERS_ENABLED


def record_started(name: str) -> None:
    if name not in _started:
        _started.append(name)


def started_workers() -> List[str]:
    """The pollers this process has started (what the gate's guard checks)."""
    return list(_started)


async def start_worker(name: str, start: Callable[[], Union[None, Awaitable[None]]]) -> bool:
    """Start a poller unless background workers are off; True when started."""
    if not background_workers_enabled():
        logger.info("%s not started (BACKGROUND_WORKERS_ENABLED=false)", name)
        return False
    result = start()
    if inspect.isawaitable(result):
        await result
    record_started(name)
    return True
