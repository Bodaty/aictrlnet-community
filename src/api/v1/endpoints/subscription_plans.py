"""Community subscription plan listing endpoints.

Provides read-only /plans, /plans/categorized, and /current so the frontend
subscriptionService.js can render available plans and subscription status.
Business edition replaces these with its full SubscriptionService.
"""

from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from core.database import get_db
from core.dependencies import get_current_user_safe
from models.subscription import SubscriptionPlan, Subscription, SubscriptionStatus

router = APIRouter()


def _plan_to_dict(p: SubscriptionPlan) -> dict:
    """Convert a SubscriptionPlan row to the response shape the frontend expects."""
    return {
        "id": p.id,
        "name": p.name,
        "edition": p.edition or (p.name.split("_")[0] if "_" in p.name else p.name),
        "price": {
            "amount": p.price_monthly,
            "monthly": p.price_monthly,
            "annual": p.price_annual,
            "currency": p.currency or "USD",
        },
        "features": p.features or {},
        "limits": p.limits or {},
        "highlights": [],
    }


@router.get("/plans")
async def list_subscription_plans(db: AsyncSession = Depends(get_db)):
    """List available subscription plans (public, no auth required)."""
    result = await db.execute(
        select(SubscriptionPlan)
        .where(SubscriptionPlan.is_active == True)
        .order_by(SubscriptionPlan.price_monthly)
    )
    plans = result.scalars().all()
    return [_plan_to_dict(p) for p in plans]


@router.get("/plans/categorized")
async def get_categorized_plans(
    deployment_type: str = "cloud",
    db: AsyncSession = Depends(get_db),
):
    """Get plans organized for progressive disclosure UX (public, no auth)."""
    result = await db.execute(
        select(SubscriptionPlan)
        .where(SubscriptionPlan.is_active == True)
        .order_by(SubscriptionPlan.price_monthly)
    )
    plans = result.scalars().all()

    primary_tiers = []
    for p in plans:
        primary_tiers.append({
            "tier": p.name,
            "display_name": p.display_name or p.name,
            "badge": None,
            "cta": "Get Started" if p.price_monthly == 0 else "View Tiers",
            "expandable": False,
            "plan": _plan_to_dict(p),
            "base_price": p.price_monthly,
            "sub_tiers": None,
        })

    return {
        "primary_tiers": primary_tiers,
        "special_editions": [],
        "deployment_type": deployment_type,
    }


@router.get("/current")
async def get_current_subscription(
    current_user: dict = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db),
):
    """Get the current user's subscription status.

    Returns subscription details including trial state and billing availability.
    If no subscription exists, returns a default community free-tier response.
    """
    user_id = current_user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found")

    # Find the user's most recent active/trialing subscription
    result = await db.execute(
        select(Subscription)
        .where(
            Subscription.user_id == user_id,
            Subscription.status.in_([
                SubscriptionStatus.ACTIVE,
                SubscriptionStatus.TRIALING,
                SubscriptionStatus.PAST_DUE,
            ])
        )
        .order_by(Subscription.started_at.desc())
        .limit(1)
    )
    subscription = result.scalar_one_or_none()

    if not subscription:
        # No subscription — return default community free-tier
        now = datetime.utcnow()
        return {
            "subscription": {
                "id": "default",
                "plan_id": "community-free",
                "status": "active",
                "edition": "community",
                "is_trial": False,
                "trial_end": None,
                "days_remaining": None,
                "has_billing": False,
                "current_period_start": now.isoformat(),
                "current_period_end": (now + timedelta(days=365)).isoformat(),
                "cancel_at_period_end": False,
                "features": {},
                "usage": {
                    "users": {"current": 1, "limit": 1},
                    "workflows": {"current": 0, "limit": 10},
                    "tasks_this_month": {"current": 0, "limit": 10000},
                },
            }
        }

    # Determine trial state
    now = datetime.utcnow()
    is_trial = subscription.status == SubscriptionStatus.TRIALING
    trial_end = subscription.trial_end
    days_remaining = None
    if is_trial and trial_end:
        days_remaining = max(0, (trial_end - now).days)

    # Determine if billing portal is available (requires Stripe customer)
    has_billing = bool(subscription.stripe_customer_id)

    # Get the plan details
    plan = await db.get(SubscriptionPlan, subscription.plan_id) if subscription.plan_id else None
    edition = "community"
    if plan:
        edition = plan.edition or (plan.name.split("_")[0] if "_" in plan.name else plan.name)

    return {
        "subscription": {
            "id": subscription.id,
            "plan_id": subscription.plan_id,
            "status": subscription.status.value if hasattr(subscription.status, 'value') else str(subscription.status),
            "edition": edition,
            "is_trial": is_trial,
            "trial_end": trial_end.isoformat() if trial_end else None,
            "days_remaining": days_remaining,
            "has_billing": has_billing,
            "current_period_start": subscription.current_period_start.isoformat() if subscription.current_period_start else None,
            "current_period_end": subscription.current_period_end.isoformat() if subscription.current_period_end else None,
            "cancel_at_period_end": subscription.cancel_at_period_end or False,
        }
    }
