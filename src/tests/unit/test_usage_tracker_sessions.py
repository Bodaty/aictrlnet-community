"""Regression: the UsageTracker singleton must not pin any caller's session.

It previously cached the FIRST caller's request-scoped AsyncSession forever
and ran all reads and flushes on it — shared across concurrent requests
(AsyncSession is not concurrency-safe) and, in per-test-loop harnesses,
bound to a dead event loop.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import core.usage_tracker as usage_tracker_module
from core.usage_tracker import UsageTracker, get_usage_tracker


def _mock_session():
    db = MagicMock()
    db.execute = AsyncMock()
    db.commit = AsyncMock()
    return db


@pytest.fixture
def isolated_singleton():
    saved = usage_tracker_module._usage_tracker
    usage_tracker_module._usage_tracker = None
    with patch(
        "core.usage_tracker.get_settings",
        return_value=MagicMock(ENVIRONMENT="test"),
    ):
        yield
    usage_tracker_module._usage_tracker = saved


@pytest.mark.asyncio
async def test_singleton_does_not_pin_first_callers_session(isolated_singleton):
    first = await get_usage_tracker(_mock_session())
    second = await get_usage_tracker(_mock_session())

    assert first is second
    assert first.db is None


@pytest.mark.asyncio
async def test_reads_use_the_passed_session(isolated_singleton):
    tracker = await get_usage_tracker()

    session = _mock_session()
    result = MagicMock()
    result.all.return_value = []
    session.execute = AsyncMock(return_value=result)

    summary = await tracker.get_usage_summary("tenant-x", db=session)

    session.execute.assert_awaited_once()
    assert summary["tenant_id"] == "tenant-x"


def test_no_flush_task_in_test_env():
    with patch(
        "core.usage_tracker.get_settings",
        return_value=MagicMock(ENVIRONMENT="test"),
    ):
        tracker = UsageTracker(_mock_session())

    assert tracker._flush_task is None


@pytest.mark.asyncio
async def test_singleton_does_not_pin_the_first_loops_cache(isolated_singleton):
    """`get_cache()` rebuilds its client per event loop; the tracker must ask
    each time rather than keep the first one, or a later loop's increment runs
    on a closed loop ("Event loop is closed" — seen under pytest-xdist, 31 Oct).
    """
    first_cache, second_cache = MagicMock(), MagicMock()
    for cache in (first_cache, second_cache):
        cache.increment = AsyncMock(return_value=1)
        cache.get = AsyncMock(return_value=None)
    tracker = UsageTracker()
    with patch("core.usage_tracker.get_cache", AsyncMock(side_effect=[first_cache, second_cache, second_cache])):
        await tracker.track_usage("tenant-1", "workflow_manual_creation")
        await tracker.track_usage("tenant-1", "workflow_manual_creation")
        await tracker.get_ai_agent_daily_usage("tenant-1")

    assert first_cache.increment.await_count == 1
    assert second_cache.increment.await_count == 1
    second_cache.get.assert_awaited_once()
