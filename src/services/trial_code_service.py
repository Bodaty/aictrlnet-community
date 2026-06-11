"""Trial redemption code service.

Config-driven trial codes (settings.TRIAL_CODES) let specific cohorts —
e.g. Institute workshop attendees — redeem an extended Business Starter
trial instead of the standard 14-day one. No new tables: codes live in
config, attribution is stamped into Subscription.meta_data.

Format: "CODE:days:max_redemptions[:expiry_iso]", comma-separated.
Example: "INSTITUTE90:90:40:2026-10-31,LIVE30:30:100"
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, date
from typing import Dict, Optional
import logging
import uuid

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from core.config import get_settings
from core.tenant_context import DEFAULT_TENANT_ID
from models import User
from models.subscription import (
    Subscription,
    SubscriptionPlan,
    SubscriptionStatus,
    BillingPeriod,
)

logger = logging.getLogger(__name__)


class TrialCodeError(Exception):
    """Typed redemption failure mapped to an HTTP status by the endpoint."""

    def __init__(self, status_code: int, code: str, message: str):
        self.status_code = status_code
        self.code = code
        self.message = message
        super().__init__(message)


@dataclass
class TrialCodeSpec:
    code: str
    days: int
    max_redemptions: int
    expires: Optional[date] = None


def parse_trial_codes(raw: str) -> Dict[str, TrialCodeSpec]:
    """Parse the TRIAL_CODES setting. Malformed entries are logged and skipped."""
    specs: Dict[str, TrialCodeSpec] = {}
    if not raw or not raw.strip():
        return specs
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        parts = [p.strip() for p in entry.split(":")]
        try:
            code = parts[0].upper()
            days = int(parts[1])
            max_redemptions = int(parts[2])
            expires = date.fromisoformat(parts[3]) if len(parts) > 3 and parts[3] else None
            if not code or days <= 0 or max_redemptions <= 0:
                raise ValueError("non-positive values")
            specs[code] = TrialCodeSpec(code, days, max_redemptions, expires)
        except (IndexError, ValueError) as e:
            logger.warning(f"Skipping malformed TRIAL_CODES entry {entry!r}: {e}")
    return specs


async def redeem_trial_code(db: AsyncSession, user_id: str, raw_code: str) -> dict:
    """Apply a trial code to the user's subscription.

    State machine (single source of truth for all redemption paths):
    unknown/expired code -> 404; already redeemed a code -> 400;
    Stripe-managed sub -> 409; paid/canceled sub -> 400;
    cap exhausted -> 410; otherwise extend/reactivate/create the trial.
    """
    code = (raw_code or "").strip().upper()
    specs = parse_trial_codes(get_settings().TRIAL_CODES)
    spec = specs.get(code)
    now = datetime.utcnow()

    if not spec or (spec.expires and now.date() > spec.expires):
        raise TrialCodeError(404, "invalid_code", "That code isn't valid.")

    user = await db.get(User, user_id)
    if not user:
        raise TrialCodeError(404, "user_not_found", "User not found.")

    # Latest subscription row; defensive ordering in case of dirty data
    # (several billing paths assume one row per user — never add a second).
    result = await db.execute(
        select(Subscription)
        .where(Subscription.user_id == user_id)
        .order_by(Subscription.started_at.desc())
    )
    rows = result.scalars().all()
    if len(rows) > 1:
        logger.warning(f"User {user_id} has {len(rows)} subscription rows; using latest")
    sub = rows[0] if rows else None

    if sub is not None:
        if (sub.meta_data or {}).get("trial_code"):
            raise TrialCodeError(
                400, "already_redeemed",
                "A code has already been applied to this account."
            )
        if sub.stripe_subscription_id:
            raise TrialCodeError(
                409, "stripe_managed",
                "Your subscription is managed through Stripe billing — contact support@bodaty.com to apply this code."
            )
        if sub.status in (
            SubscriptionStatus.ACTIVE,
            SubscriptionStatus.PAST_DUE,
            SubscriptionStatus.CANCELED,
        ):
            raise TrialCodeError(
                400, "not_eligible",
                "This account already has a paid subscription."
            )

    # Redemption cap (COUNT of stamped rows; tiny table, no index needed)
    count_result = await db.execute(
        select(func.count(Subscription.id)).where(
            Subscription.meta_data["trial_code"].as_string() == code
        )
    )
    redeemed = count_result.scalar() or 0
    if redeemed >= spec.max_redemptions:
        raise TrialCodeError(410, "code_exhausted", "That code has been fully redeemed.")

    trial_end = now + timedelta(days=spec.days)
    reactivated = False

    if sub is None:
        # Possible when plan seeding failed at signup time — mirror register()
        plan_result = await db.execute(
            select(SubscriptionPlan).where(SubscriptionPlan.name == "business_starter")
        )
        plan = plan_result.scalar_one_or_none()
        if not plan:
            raise TrialCodeError(
                409, "plans_not_seeded",
                "Subscription plans are not configured — contact support@bodaty.com."
            )
        sub = Subscription(
            id=str(uuid.uuid4()),
            user_id=user_id,
            tenant_id=user.tenant_id or DEFAULT_TENANT_ID,
            plan_id=plan.id,
            status=SubscriptionStatus.TRIALING,
            billing_period=BillingPeriod.MONTHLY,
            started_at=now,
            current_period_start=now,
            current_period_end=trial_end,
            trial_end=trial_end,
        )
        db.add(sub)
    elif sub.status == SubscriptionStatus.TRIALING:
        # Extend, never shorten
        if sub.trial_end and sub.trial_end > trial_end:
            trial_end = sub.trial_end
        sub.trial_end = trial_end
        sub.current_period_end = trial_end
    else:  # EXPIRED — reactivate
        sub.status = SubscriptionStatus.TRIALING
        sub.trial_end = trial_end
        sub.current_period_start = now
        sub.current_period_end = trial_end
        reactivated = True

    if user.edition in ("trial_expired", "community"):
        user.edition = "business"

    # meta_data is a plain JSON column: reassign + flag so SQLAlchemy persists it
    sub.meta_data = {
        **(sub.meta_data or {}),
        "trial_code": code,
        "trial_code_redeemed_at": now.isoformat(),
    }
    flag_modified(sub, "meta_data")

    await db.commit()

    days_remaining = max(0, (trial_end - now).days)
    logger.info(f"Trial code {code} redeemed by user {user_id}: trial_end={trial_end.isoformat()}")
    return {
        "code": code,
        "days": spec.days,
        "days_remaining": days_remaining,
        "trial_end": trial_end.isoformat(),
        "reactivated": reactivated,
    }
