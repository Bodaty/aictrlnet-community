"""Regression: schedule creation must compute next_run from the cron
expression and timezone, not hardcode utcnow()+1h.

Live failure (presence-audit workflow, Aug 4 2026): a quarterly schedule
fired ~1h after creation because create_schedule ignored the cron
expression. Caught by our own automation — no Trello card.
"""

import uuid
from datetime import datetime, timedelta, timezone as dt_timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from schemas.workflow_execution import WorkflowScheduleCreate
from services.schedule_utils import compute_next_run
from services.workflow_execution import WorkflowExecutionService

QUARTERLY = "0 14 1 2,5,8,11 *"
BASE = datetime(2026, 8, 5, 12, 0, tzinfo=dt_timezone.utc)


def test_compute_next_run_utc_quarterly():
    assert compute_next_run(QUARTERLY, "UTC", base=BASE) == datetime(2026, 11, 1, 14, 0)


def test_compute_next_run_honors_timezone():
    # Nov 1 2026 is the US DST fall-back date: 14:00 America/Chicago is CST (UTC-6).
    assert compute_next_run(QUARTERLY, "America/Chicago", base=BASE) == datetime(2026, 11, 1, 20, 0)


def test_compute_next_run_invalid_cron():
    with pytest.raises(ValueError):
        compute_next_run("not a cron", "UTC")


def test_compute_next_run_invalid_timezone():
    with pytest.raises(ValueError):
        compute_next_run(QUARTERLY, "Mars/Olympus_Mons")


def _mock_db():
    db = MagicMock()
    db.commit = AsyncMock()
    db.refresh = AsyncMock()
    return db


@pytest.mark.asyncio
async def test_create_schedule_uses_cron_expression_not_one_hour():
    service = WorkflowExecutionService(_mock_db())
    schedule = await service.create_schedule(
        workflow_id=uuid.uuid4(),
        schedule_data=WorkflowScheduleCreate(
            name="quarterly-presence-audit",
            schedule_expression=QUARTERLY,
        ),
    )
    now = datetime.utcnow()
    assert schedule.next_run > now
    assert schedule.next_run.minute == 0
    assert schedule.next_run.hour == 14
    assert schedule.next_run.day == 1
    assert schedule.next_run.month in {2, 5, 8, 11}
    assert not timedelta(minutes=59) < (schedule.next_run - now) < timedelta(minutes=61)


@pytest.mark.asyncio
async def test_create_schedule_invalid_cron_raises():
    service = WorkflowExecutionService(_mock_db())
    with pytest.raises(ValueError):
        await service.create_schedule(
            workflow_id=uuid.uuid4(),
            schedule_data=WorkflowScheduleCreate(
                name="bad",
                schedule_expression="not a cron",
            ),
        )
