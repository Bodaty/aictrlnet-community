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
