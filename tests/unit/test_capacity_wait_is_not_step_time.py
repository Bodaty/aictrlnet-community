"""Time a step spends waiting for a reader is not the step's time (R64).

A document read that waits its turn for the OCR server or the extraction
model must not run down its own timeout, nor that of a loop around it, and
the wait must not count as a failed attempt. Runs the real BaseNode.run.
"""

import asyncio
from contextlib import asynccontextmanager

import pytest

from nodes import capacity
from nodes.base_node import BaseNode
from nodes.models import NodeConfig, NodeInstance, NodeStatus, NodeType


class _Step(BaseNode):
    """Waits ``wait`` seconds for capacity, then works ``work`` seconds."""

    async def execute(self, input_data, context):
        p = self.config.parameters
        if p.get("wait"):
            with capacity.waiting():
                await asyncio.sleep(p["wait"])
        await asyncio.sleep(p.get("work", 0))
        return {"ok": True}


class _Outer(BaseNode):
    """A loop-like step that runs an inner step through its own run()."""

    async def execute(self, input_data, context):
        inner = self.config.parameters["inner"]
        result = await inner.run(_instance(inner), {})
        if result.status != NodeStatus.COMPLETED:
            raise RuntimeError(result.error)
        return {"inner": "done"}


def _node(cls, timeout, **params):
    return cls(NodeConfig(id=f"n-{cls.__name__}", name=cls.__name__, type=NodeType.TASK,
                          parameters=params, timeout_seconds=timeout, retry_count=0))


def _instance(node):
    return NodeInstance(id=f"i-{node.config.id}", node_config=node.config,
                        workflow_instance_id="wf-capacity", input_data={})


async def _run(node):
    return await node.run(_instance(node), {})


@pytest.mark.asyncio
async def test_waiting_for_capacity_does_not_use_the_steps_timeout():
    result = await _run(_node(_Step, timeout=1, wait=1.5, work=0.2))
    assert result.status == NodeStatus.COMPLETED, result.error


@pytest.mark.asyncio
async def test_the_work_itself_still_times_out():
    result = await _run(_node(_Step, timeout=1, wait=0.5, work=1.5))
    assert result.status == NodeStatus.FAILED
    assert "timed out after 1s" in result.error


@pytest.mark.asyncio
async def test_an_inner_wait_does_not_use_the_outer_steps_timeout_either():
    inner = _node(_Step, timeout=1, wait=1.5, work=0.2)
    result = await _run(_node(_Outer, timeout=1, inner=inner))
    assert result.status == NodeStatus.COMPLETED, result.error


@pytest.mark.asyncio
async def test_a_slot_stops_the_clock_only_while_it_is_acquired():
    held = []

    @asynccontextmanager
    async def gate(reader, tenant_id, context):
        await asyncio.sleep(1.5)  # someone else is reading
        held.append((reader, tenant_id))
        yield

    class _Reads(BaseNode):
        async def execute(self, input_data, context):
            async with capacity.slot("test_kind", "test_reader", "t1", context):
                await asyncio.sleep(self.config.parameters["work"])
            return {}

    capacity.register_gate("test_kind", gate)
    try:
        assert (await _run(_node(_Reads, timeout=1, work=0.2))).status == NodeStatus.COMPLETED
        slow = await _run(_node(_Reads, timeout=1, work=1.5))
        assert slow.status == NodeStatus.FAILED and "timed out" in slow.error
    finally:
        capacity.unregister_gate("test_kind")
    assert held == [("test_reader", "t1"), ("test_reader", "t1")]


@pytest.mark.asyncio
async def test_no_gate_means_no_wait():
    async with capacity.slot("unregistered_kind", "test_reader", "t1", {}):
        pass


# ---------------------------------------------------------------------------
# A release that blocks never holds the caller past the unwind bound
# ---------------------------------------------------------------------------

def _blocking_gate(seen, block_s):
    """A gate whose release blocks ``block_s`` seconds and swallows cancels,
    counting each one (a stalled database socket behaves like this)."""

    @asynccontextmanager
    async def gate(reader, tenant_id, context):
        try:
            yield
        finally:
            end = asyncio.get_running_loop().time() + block_s
            while asyncio.get_running_loop().time() < end:
                try:
                    await asyncio.sleep(end - asyncio.get_running_loop().time())
                except asyncio.CancelledError:
                    seen.append("cancel")

    return gate


class _HoldsSlot(BaseNode):
    async def execute(self, input_data, context):
        async with capacity.slot("blocking_kind", "test_reader", "t1", context):
            await asyncio.sleep(self.config.parameters["work"])
        return {}


@pytest.mark.asyncio
async def test_a_blocked_release_is_left_after_the_bound_and_logged(monkeypatch, caplog):
    import logging
    import time

    seen = []
    capacity.register_gate("blocking_kind", _blocking_gate(seen, block_s=4))
    monkeypatch.setattr(capacity, "UNWIND_TIMEOUT_SECONDS", 1.0)
    caplog.set_level(logging.ERROR, logger=capacity.__name__)
    try:
        started = time.monotonic()
        result = await _run(_node(_HoldsSlot, timeout=1, work=5))
        elapsed = time.monotonic() - started
    finally:
        capacity.unregister_gate("blocking_kind")
    assert result.status == NodeStatus.FAILED and "timed out" in result.error
    assert elapsed < 3.5  # 1 s timeout + 1 s unwind bound, not the 4 s release
    assert any("did not unwind" in r.getMessage() for r in caplog.records)
    await asyncio.sleep(4)  # let the stray release finish


@pytest.mark.asyncio
async def test_a_repeated_cancel_reaches_the_step_and_is_raised(monkeypatch):
    seen = []
    capacity.register_gate("blocking_kind", _blocking_gate(seen, block_s=3))
    monkeypatch.setattr(capacity, "UNWIND_TIMEOUT_SECONDS", 10.0)
    try:
        node = _node(_HoldsSlot, timeout=30, work=30)
        running = asyncio.create_task(node.run(_instance(node), {}))
        await asyncio.sleep(0.5)
        running.cancel()          # the step starts unwinding; its release blocks
        await asyncio.sleep(0.5)
        running.cancel()          # passed on to the step
        with pytest.raises(asyncio.CancelledError):
            await running
    finally:
        capacity.unregister_gate("blocking_kind")
    assert seen.count("cancel") >= 1  # the second cancel reached the release
