"""Shared next_run computation for workflow schedules.

Single source of truth for the community create-schedule path and the
business scheduler's create/advance paths, so all three compute the same
fire times.
"""

from datetime import datetime, timezone as dt_timezone
from typing import Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from croniter import croniter


def compute_next_run(
    cron_expression: str,
    tz_name: str = "UTC",
    base: Optional[datetime] = None,
) -> datetime:
    """Next fire time for a cron expression, returned as naive UTC — the
    shape workflow_schedules.next_run stores and the business fire-loop
    compares against utcnow().

    The cron expression is evaluated in tz_name (croniter handles DST with
    a tz-aware base). A naive base is treated as UTC. Raises ValueError for
    an invalid cron expression or unknown timezone.
    """
    try:
        tz = ZoneInfo(tz_name or "UTC")
    except (ZoneInfoNotFoundError, KeyError, ValueError) as e:
        raise ValueError(f"Unknown timezone '{tz_name}': {e}")

    base_utc = base if base is not None else datetime.now(dt_timezone.utc)
    if base_utc.tzinfo is None:
        base_utc = base_utc.replace(tzinfo=dt_timezone.utc)

    try:
        next_fire = croniter(cron_expression, base_utc.astimezone(tz)).get_next(datetime)
    except ValueError as e:
        raise ValueError(f"Invalid cron expression '{cron_expression}': {e}")

    return next_fire.astimezone(dt_timezone.utc).replace(tzinfo=None)
