"""Public pricing/plans endpoint."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from core.database import get_db
from models.subscription import SubscriptionPlan

router = APIRouter()


@router.get("/plans")
async def list_pricing_plans(
    db: AsyncSession = Depends(get_db),
):
    """List available subscription plans (public, no auth required)."""
    result = await db.execute(
        select(SubscriptionPlan).where(SubscriptionPlan.is_active == True).order_by(SubscriptionPlan.price_monthly)
    )
    plans = result.scalars().all()
    return {
        "plans": [
            {
                "id": p.id,
                "name": p.name,
                "display_name": p.display_name,
                "description": p.description,
                "price": {
                    "monthly": p.price_monthly,
                    "annual": p.price_annual,
                    "currency": p.currency or "USD",
                },
                "features": p.features or {},
                "limits": p.limits or {},
            }
            for p in plans
        ]
    }
