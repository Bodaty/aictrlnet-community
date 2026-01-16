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
    """Get current license status and subscription details."""
    
    tenant_id = current_user.get("tenant_id") or get_current_tenant_id()
    enforcer = LicenseEnforcer(db)
    
    # Get tenant info
    tenant_info = await enforcer._get_tenant_info(tenant_id)
    edition = Edition(tenant_info["edition"])
    
    # Get edition limits and features
    limits = enforcer.EDITION_LIMITS.get(edition, {})
    features = enforcer.EDITION_FEATURES.get(edition, set())
    
    # Build feature flags
    feature_flags = {
        "max_users": limits.get("USERS", 1),
        "max_workflows": limits.get("WORKFLOWS", 10),
        "max_executions_per_month": limits.get("EXECUTIONS", 1000),
        "ai_governance": "advanced_analytics" in features,
        "advanced_analytics": "advanced_analytics" in features,
        "custom_integrations": "business_adapters" in features,
        "mfa_enabled": edition.value != "community",
        "sso_enabled": "enterprise_adapters" in features,
        "platform_integration_nodes": 5 if "business_adapters" in features else 0
    }
    
    # Calculate billing period (mock for now, will integrate with Stripe later)
    now = datetime.utcnow()
    period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
    if now.month == 12:
        period_end = datetime(now.year + 1, 1, 1) - timedelta(seconds=1)
    else:
        period_end = datetime(now.year, now.month + 1, 1) - timedelta(seconds=1)
    
    return LicenseStatusResponse(
        subscription={
            "id": f"sub_{tenant_id}",
            "plan": edition.value,
            "status": "active",
            "current_period_start": period_start.isoformat() + "Z",
            "current_period_end": period_end.isoformat() + "Z",
            "features": feature_flags
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