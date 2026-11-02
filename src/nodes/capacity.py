"""Waiting for a reader is not part of a step's time.

A step whose work needs a scarce shared reader (a local OCR server, a local
model) may have to wait its turn. That wait must not run down the step's
timeout, nor the timeout of any step around it (a loop over pages): every
step runs under a deadline whose clock stops while the step, or anything
inside it, is waiting for capacity.

Editions register a gate per content-reference kind (``register_gate``). A
gate is ``gate(reader, tenant_id, context)`` returning an async context
manager that is entered once the caller holds a slot on ``reader`` and
releases it on exit. ``slot`` enters the gate for a kind (nothing to wait for
when none is registered) and stops the clocks while it waits.
"""

import asyncio
import contextvars
import logging
import os
import time
from contextlib import AsyncExitStack, asynccontextmanager, contextmanager
from typing import Any, AsyncContextManager, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

Gate = Callable[[str, str, dict], AsyncContextManager[Any]]

_gates: Dict[str, Gate] = {}


def register_gate(kind: str, gate: Gate) -> None:
    _gates[kind] = gate


def unregister_gate(kind: str) -> None:
    _gates.pop(kind, None)


class StepClock:
    """Time a step has spent waiting for capacity; overlapping waits count once."""

    def __init__(self) -> None:
        self.waited = 0.0
        self.active = 0
        self._since = 0.0

    def enter(self, now: float) -> None:
        if self.active == 0:
            self._since = now
        self.active += 1

    def exit(self, now: float) -> None:
        self.active -= 1
        if self.active == 0:
            self.waited += now - self._since

    def paused(self, now: float) -> float:
        return self.waited + (now - self._since if self.active else 0.0)


_clocks: contextvars.ContextVar[Tuple[StepClock, ...]] = contextvars.ContextVar(
    "step_clocks", default=())


@contextmanager
def waiting():
    """Stop the clock of every step this code runs inside."""
    clocks = _clocks.get()
    now = time.monotonic()
    for clock in clocks:
        clock.enter(now)
    try:
        yield
    finally:
        now = time.monotonic()
        for clock in clocks:
            clock.exit(now)


@asynccontextmanager
async def slot(kind: str, reader: Optional[str], tenant_id: Optional[str], context: dict):
    """Hold a slot on ``reader`` for a step reading content of ``kind``."""
    gate = _gates.get(kind)
    if gate is None or not reader:
        yield
        return
    async with AsyncExitStack() as stack:
        with waiting():
            await stack.enter_async_context(gate(reader, tenant_id or "", context))
        yield


async def run_with_deadline(coro, timeout: Optional[float]):
    """Await ``coro`` like ``asyncio.wait_for``, except that time spent waiting
    for capacity (``waiting``) inside it does not count toward ``timeout``."""
    clock = StepClock()
    token = _clocks.set(_clocks.get() + (clock,))
    try:
        task = asyncio.ensure_future(coro)  # copies the context: sees this clock
    finally:
        _clocks.reset(token)
    if timeout is None:
        return await task
    started = time.monotonic()
    try:
        while True:
            now = time.monotonic()
            remaining = started + timeout + clock.paused(now) - now
            if remaining <= 0:
                await _cancel_and_observe(task)
                raise asyncio.TimeoutError()
            done, _ = await asyncio.wait({task}, timeout=remaining)
            if done:
                return task.result()
    except asyncio.CancelledError:
        await _cancel_and_observe(task)
        raise


# How long a stopped step may take to unwind (release what it holds) before
# its caller gives up waiting; the step is cancelled again and left.
_DEFAULT_UNWIND_TIMEOUT_SECONDS = 10.0


def _unwind_timeout_from_env() -> float:
    """STEP_UNWIND_TIMEOUT_SECONDS, a number of seconds above 0; anything else
    is reported and the default used (it never stops a process starting)."""
    raw = os.environ.get("STEP_UNWIND_TIMEOUT_SECONDS")
    if raw is None or not raw.strip():
        return _DEFAULT_UNWIND_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError:
        value = float("nan")
    if not (value > 0 and value != float("inf")):
        logger.warning("STEP_UNWIND_TIMEOUT_SECONDS=%r is not a number of seconds above 0; "
                       "using %.0f", raw, _DEFAULT_UNWIND_TIMEOUT_SECONDS)
        return _DEFAULT_UNWIND_TIMEOUT_SECONDS
    return value


UNWIND_TIMEOUT_SECONDS = _unwind_timeout_from_env()


async def _cancel_and_observe(task: "asyncio.Future") -> None:
    """Cancel the step and wait, at most UNWIND_TIMEOUT_SECONDS, for it to
    unwind, so what it releases on the way out (a reader slot) is released now
    and a failure to release is seen. A repeated cancel of the caller is passed
    on to the step and then re-raised; an unwind past the bound is logged as an
    error and the step is left cancelled."""
    task.cancel()
    cancelled_meanwhile = False
    deadline = time.monotonic() + UNWIND_TIMEOUT_SECONDS
    while not task.done():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            logger.error("a stopped step did not unwind within %.0f s; leaving it cancelled",
                         UNWIND_TIMEOUT_SECONDS)
            task.cancel()
            break
        try:
            await asyncio.wait({task}, timeout=remaining)
        except asyncio.CancelledError:
            cancelled_meanwhile = True
            task.cancel()  # pass the repeated cancel on to the step
    if task.done() and not task.cancelled() and task.exception() is not None:
        logger.warning("a step failed while it was being stopped: %s", type(task.exception()).__name__)
    if cancelled_meanwhile:
        raise asyncio.CancelledError()
