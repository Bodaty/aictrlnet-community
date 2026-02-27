"""License management API endpoints."""

from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, Request, Header
from sqlalchemy.ext.asyncio import AsyncSession
import logging

from core.database import get_db
from core.dependencies import get_current_user_safe
from core.enforcement_simple import LicenseEnforcer, Edition
from core.tenant_context import get_current_tenant_id
from schemas.license import (
    LicenseStatusResponse,
    CurrentUsageResponse,
    UsageHistoryResponse,
    UsageAlertsResponse,
    LicenseUpgradeRequest,
    LicenseUpgradeResponse
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/current", response_model=LicenseStatusResponse)
async def get_current_license(
    current_user: dict = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db)
):
    """Get current license status and subscription details.

    Queries real subscription data from the database.
    Falls back to Community (Free) when no active subscription exists.
    """
    from sqlalchemy import select
    from models.subscription import Subscription, SubscriptionPlan, SubscriptionStatus

    user_id = current_user.get("sub") or current_user.get("id")
    user_edition = current_user.get("edition", "community")

    # Try to find the user's active subscription
    sub_result = None
    try:
        stmt = (
            select(Subscription, SubscriptionPlan)
            .join(SubscriptionPlan, Subscription.plan_id == SubscriptionPlan.id)
            .where(Subscription.user_id == user_id)
            .where(Subscription.status.in_([SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIALING]))
            .order_by(Subscription.current_period_end.desc())
            .limit(1)
        )
        result = await db.execute(stmt)
        sub_result = result.first()
    except Exception as e:
        logger.warning(f"Could not query subscription for user {user_id}: {e}")

    if sub_result:
        subscription, plan = sub_result
        return LicenseStatusResponse(
            subscription={
                "id": subscription.id,
                "plan": user_edition or "community",
                "plan_name": plan.display_name,
                "status": subscription.status.value if hasattr(subscription.status, 'value') else str(subscription.status),
                "current_period_start": subscription.current_period_start.isoformat() + "Z",
                "current_period_end": subscription.current_period_end.isoformat() + "Z",
                "features": plan.features or {}
            }
        )

    # No active subscription — derive from user's edition field
    edition = user_edition or "community"
    edition_display = {
        "community": "Community (Free)",
        "team": "Team",
        "business": "Business",
        "enterprise": "Enterprise",
    }
    plan_name = edition_display.get(edition, edition.title())
    status = "free" if edition == "community" else "active"

    now = datetime.utcnow()
    period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if now.month == 12:
        period_end = datetime(now.year + 1, 1, 1) - timedelta(seconds=1)
    else:
        period_end = datetime(now.year, now.month + 1, 1) - timedelta(seconds=1)

    return LicenseStatusResponse(
        subscription={
            "id": f"{edition}_{user_id}",
            "plan": edition,
            "plan_name": plan_name,
            "status": status,
            "current_period_start": period_start.isoformat() + "Z",
            "current_period_end": period_end.isoformat() + "Z",
            "features": {}
        }
    )


@router.post("/upgrade", response_model=LicenseUpgradeResponse)
async def upgrade_license(
    request: LicenseUpgradeRequest,
    current_user: dict = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db)
):
    """Upgrade license to a higher plan."""

    # Import here to avoid circular imports
    from services.stripe_service import StripeService

    # Check if Stripe is configured
    from core.config import get_settings
    settings = get_settings()

    # Enterprise plan requires contacting sales
    if request.target_plan.lower() == "enterprise":
        return LicenseUpgradeResponse(
            subscription={
                "id": "enterprise_inquiry",
                "plan": "enterprise",
                "status": "contact_sales",
                "current_period_start": datetime.utcnow().isoformat() + "Z",
                "current_period_end": datetime.utcnow().isoformat() + "Z",
                "features": {
                    "unlimited_users": True,
                    "unlimited_workflows": True,
                    "sso_saml": True,
                    "dedicated_support": True,
                    "custom_sla": True,
                    "on_premise_option": True,
                    "custom_integrations": True
                }
            },
            requires_payment=False,
            contact_sales=True,
            contact_sales_url="https://aictrlnet.com/contact-sales",
            contact_sales_email="sales@aictrlnet.com",
            message="Enterprise plans are customized to your organization's needs. Our sales team will work with you to create a tailored solution."
        )

    if not settings.STRIPE_SECRET_KEY or settings.STRIPE_SECRET_KEY == "sk_test_dummy":
        # Return mock response if Stripe not configured
        return LicenseUpgradeResponse(
            subscription={
                "id": f"sub_upgraded",
                "plan": request.target_plan,
                "status": "pending_payment",
                "current_period_start": datetime.utcnow().isoformat() + "Z",
                "current_period_end": (datetime.utcnow() + timedelta(days=30)).isoformat() + "Z",
                "features": {}
            },
            payment_intent=None,
            requires_payment=True,
            checkout_url="/mock-checkout"  # Mock checkout URL
        )

    # Use real Stripe integration
    stripe_service = StripeService(db)

    try:
        checkout_data = await stripe_service.create_checkout_session(
            user_id=current_user.get("sub"),
            plan=request.target_plan,
            billing_period=request.billing_period
        )

        return LicenseUpgradeResponse(
            subscription={
                "id": checkout_data.get("session_id"),
                "plan": request.target_plan,
                "status": "pending_payment",
                "current_period_start": datetime.utcnow().isoformat() + "Z",
                "current_period_end": (datetime.utcnow() + timedelta(days=30)).isoformat() + "Z",
                "features": {}
            },
            payment_intent=checkout_data.get("session_id"),
            requires_payment=True,
            checkout_url=checkout_data.get("checkout_url")
        )
    except Exception as e:
        logger.error(f"Error creating Stripe checkout session: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to create checkout session"
        )


@router.post("/webhook/stripe")
async def stripe_webhook(
    request: Request,
    stripe_signature: str = Header(None),
    db: AsyncSession = Depends(get_db)
):
    """Handle Stripe webhook events."""
    
    from services.stripe_service import StripeService
    from core.config import get_settings
    import stripe
    
    settings = get_settings()
    
    # Get the webhook payload
    payload = await request.body()
    
    # Verify webhook signature
    try:
        event = stripe.Webhook.construct_event(
            payload, stripe_signature, settings.STRIPE_WEBHOOK_SECRET
        )
    except ValueError:
        # Invalid payload
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError:
        # Invalid signature
        raise HTTPException(status_code=400, detail="Invalid signature")
    
    # Handle the event
    stripe_service = StripeService(db)
    
    try:
        await stripe_service.handle_webhook(event["type"], event["data"]["object"])
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error handling Stripe webhook: {e}")
        # Return success to avoid Stripe retrying
        return {"status": "success", "error": str(e)}