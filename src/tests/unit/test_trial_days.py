"""A new 14-day trial reads 14 days left, not 13.

The trial-days count used timedelta.days, which floors: a signup made seconds
ago showed "13 days left" on hosted (27 Nov 2026) for a trial ending exactly
14 days later.
"""

from datetime import datetime, timedelta

from core.trial_days import trial_days_left


NOW = datetime(2026, 11, 27, 23, 15, 0)


def test_new_trial_reads_its_full_length():
    assert trial_days_left(NOW + timedelta(days=14), NOW + timedelta(seconds=5)) == 14


def test_partial_day_counts_as_a_day():
    assert trial_days_left(NOW + timedelta(hours=3), NOW) == 1


def test_ended_trial_is_zero():
    assert trial_days_left(NOW - timedelta(minutes=1), NOW) == 0
    assert trial_days_left(None, NOW) == 0
