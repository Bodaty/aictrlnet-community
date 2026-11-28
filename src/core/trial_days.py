"""Days left in a trial, as people count them.

A trial that ends in 13 days and 23 hours has 14 days left: the count rounds
up, so a new 14-day trial reads 14, not 13. It reaches 0 only once the end
has passed.
"""

import math
from datetime import datetime
from typing import Optional


def trial_days_left(trial_end: Optional[datetime], now: Optional[datetime] = None) -> int:
    if not trial_end:
        return 0
    seconds = (trial_end - (now or datetime.utcnow())).total_seconds()
    return max(0, math.ceil(seconds / 86400))
