"""Shared helper: ensure seeded test users have subscriptions matching their edition.

Used by all three seed scripts (community/business/enterprise) so the dev/admin/test
users never end up in a state where ``User.edition`` says one thing but no Subscription
row backs it. Without this, ``/subscription/current`` falls back to a hard-coded
``community-free`` default and platform tier filters (e.g. MCP ``tools/list``) treat
the user as Community even when their record claims Enterprise.

Idempotent: safe to call repeatedly. Picks the highest-priced active plan for the
requested edition; falls back to lower editions if the target tier isn't seeded yet,
and aligns ``User.edition`` to whatever plan the user actually ended up on so the
two never drift.
"""

import logging
import uuid
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

# Each seed script puts a different path on sys.path before importing this helper:
# - Community seed has editions/community on path (so models live under src.models)
# - Business/Enterprise seeds have editions/community/src on path (so models live under models)
try:
    from models.subscription import (
        BillingPeriod,
        Subscription,
        SubscriptionPlan,
        SubscriptionStatus,
    )
    from models.user import User
except ImportError:
    from src.models.subscription import (  # type: ignore[no-redef]
        BillingPeriod,
        Subscription,
        SubscriptionPlan,
        SubscriptionStatus,
    )
    from src.models.user import User  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

TEST_USER_EMAILS = ["dev@aictrlnet.com", "admin@aictrlnet.com", "test@example.com"]
EDITION_FALLBACK_ORDER = ["enterprise", "business", "community"]

# Canonical plans for test/dev users — pick general-purpose SaaS plans, not
# specialized variants (e.g. government, on-prem) that have constraints
# unrelated to feature tier. Lookup is by SubscriptionPlan.name.
PREFERRED_PLAN_NAMES = {
    "enterprise": ["enterprise-cloud", "enterprise"],
    "business": ["business_scale", "business_growth", "business_starter", "team"],
    "community": ["community_cloud", "community"],
}


async def _pick_plan_for_edition(session: AsyncSession, target_edition: str):
    """Find the best active plan for ``target_edition``, falling back to lower tiers.

    Prefers canonical plan names per :data:`PREFERRED_PLAN_NAMES`. If none of the
    canonical plans are seeded, falls back to the cheapest active plan within the
    edition (cheap-not-pricy because the most-expensive plan is often a specialized
    variant — government, on-prem — that we don't want to default test users into).
    """
    if target_edition not in EDITION_FALLBACK_ORDER:
        target_edition = "community"
    start = EDITION_FALLBACK_ORDER.index(target_edition)

    for edition in EDITION_FALLBACK_ORDER[start:]:
        for name in PREFERRED_PLAN_NAMES.get(edition, []):
            result = await session.execute(
                select(SubscriptionPlan).where(
                    SubscriptionPlan.name == name,
                    SubscriptionPlan.is_active.is_(True),
                )
            )
            plan = result.scalar_one_or_none()
            if plan:
                return plan
        # No canonical plan seeded yet for this edition — take the cheapest active
        # one as a sensible default rather than the most expensive specialized one.
        result = await session.execute(
            select(SubscriptionPlan)
            .where(
                SubscriptionPlan.edition == edition,
                SubscriptionPlan.is_active.is_(True),
            )
            .order_by(SubscriptionPlan.price_monthly.asc())
            .limit(1)
        )
        plan = result.scalar_one_or_none()
        if plan:
            return plan
    return None


async def ensure_test_user_subscriptions(session: AsyncSession) -> int:
    """Ensure every seeded test user has a Subscription matching their edition.

    Returns the number of subscriptions created or updated.
    """
    result = await session.execute(
        select(User).where(User.email.in_(TEST_USER_EMAILS))
    )
    users = result.scalars().all()
    if not users:
        logger.info("No test users present yet; nothing to subscribe.")
        return 0

    changed = 0
    now = datetime.utcnow()
    for user in users:
        plan = await _pick_plan_for_edition(session, user.edition or "community")
        if plan is None:
            logger.warning(
                "Skipping %s — no active plan found for edition=%s (no plans seeded yet?)",
                user.email,
                user.edition,
            )
            continue

        sub_result = await session.execute(
            select(Subscription)
            .where(
                Subscription.user_id == user.id,
                Subscription.status.in_(
                    [SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIALING]
                ),
            )
            .order_by(Subscription.started_at.desc())
            .limit(1)
        )
        sub = sub_result.scalar_one_or_none()

        if sub:
            if sub.plan_id != plan.id:
                logger.info(
                    "  %s: upgrading subscription %s → %s",
                    user.email,
                    sub.plan_id,
                    plan.id,
                )
                sub.plan_id = plan.id
                sub.status = SubscriptionStatus.ACTIVE
                sub.current_period_start = now
                sub.current_period_end = now + timedelta(days=365)
                changed += 1
            else:
                logger.info("  %s: already on %s", user.email, plan.id)
        else:
            sub = Subscription(
                id=str(uuid.uuid4()),
                user_id=user.id,
                tenant_id=user.tenant_id or "default-tenant",
                plan_id=plan.id,
                status=SubscriptionStatus.ACTIVE,
                billing_period=BillingPeriod.MONTHLY,
                started_at=now,
                current_period_start=now,
                current_period_end=now + timedelta(days=365),
            )
            session.add(sub)
            logger.info("  %s: created subscription on %s", user.email, plan.id)
            changed += 1

        # Keep User.edition aligned with the plan they actually have, so
        # subscription_mismatch never triggers in seeded data.
        if user.edition != plan.edition:
            logger.info(
                "  %s: aligning user.edition %s → %s to match plan",
                user.email,
                user.edition,
                plan.edition,
            )
            user.edition = plan.edition

    if changed:
        await session.commit()
    return changed
