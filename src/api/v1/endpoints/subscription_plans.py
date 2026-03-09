"""Community subscription plan listing endpoints.

Provides read-only /plans and /plans/categorized so the frontend
subscriptionService.js can render available plans for Community users.
Business edition replaces these with its full SubscriptionService.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from core.database import get_db
from models.subscription import SubscriptionPlan

router = APIRouter()


def _plan_to_dict(p: SubscriptionPlan) -> dict:
    """Convert a SubscriptionPlan row to the response shape the frontend expects."""
    return {
        "id": p.id,
        "name": p.name,
        "edition": p.name.split("_")[0] if "_" in p.name else p.name,
        "price": {
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
