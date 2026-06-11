"""Billing API endpoints for Stripe integration.

Exposes StripeService methods for:
- Billing portal access
- Invoice listing and retrieval
- Upcoming invoice preview
- Free trial signup
"""

from typing import Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Header
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
import logging
import os

from core.database import get_db
from models.subscription import Subscription, SubscriptionStatus
from core.dependencies import get_current_user_safe
from schemas.billing import (
    BillingPortalResponse,
    InvoiceListResponse,
    InvoiceDetailItem,
    UpcomingInvoiceResponse,
    StartTrialRequest,
    StartTrialResponse,
    RedeemTrialCodeRequest,
    RedeemTrialCodeResponse,
)
from services.stripe_service import StripeService
from services.trial_code_service import redeem_trial_code, TrialCodeError
from core.config import get_settings

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/portal", response_model=BillingPortalResponse)
async def get_billing_portal(
    current_user: dict = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db)
):
    """Get Stripe Customer Portal URL for self-service billing management.

    The portal allows users to:
    - Update payment methods
    - View and download invoices
    - Manage subscription (cancel, change plan)
    """
    user_id = current_user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found")

    stripe_service = StripeService(db)

    try:
        portal_url = await stripe_service.get_billing_portal_url(user_id)
        return BillingPortalResponse(portal_url=portal_url)
    except ValueError as e:
        # User doesn't have billing info (e.g., community edition)
        raise HTTPException(
            status_code=404,
            detail={
                "error": "no_subscription",
                "message": str(e),
                "redirect": "/pricing"
            }
        )
    except Exception as e:
        logger.error(f"Error getting billing portal URL: {e}")
        raise HTTPException(status_code=500, detail="Failed to get billing portal URL")


@router.post("/start-trial", response_model=StartTrialResponse)
async def start_trial(
    request: StartTrialRequest = None,
    current_user: dict = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db)
):
    """Start a 14-day free trial for a Business plan.

    Creates a Stripe Checkout session with a trial period. Payment method
    collection is deferred — the user can start the trial without a credit card.
    If no payment method is added by trial end, Stripe auto-cancels the
    subscription and the user is downgraded to Community.
    """
    user_id = current_user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found")

    if request is None:
        request = StartTrialRequest()

    # Only business plans are eligible for trial
    if not request.plan.startswith("business"):
        raise HTTPException(
            status_code=400,
            detail="Free trials are only available for Business plans"
        )

    stripe_service = StripeService(db)

    # Honor an existing in-app trial that is longer than the default
    # (e.g. redeemed trial codes) so starting Stripe checkout never
    # shortens a promised trial period.
    trial_days = get_settings().TRIAL_DAYS
    trial_result = await db.execute(
        select(Subscription).where(
            Subscription.user_id == user_id,
            Subscription.status == SubscriptionStatus.TRIALING,
            Subscription.stripe_subscription_id.is_(None),
        )
    )
    in_app_trial = trial_result.scalars().first()
    if in_app_trial and in_app_trial.trial_end:
        remaining_seconds = (in_app_trial.trial_end - datetime.utcnow()).total_seconds()
        if remaining_seconds > 0:
            trial_days = max(trial_days, int(remaining_seconds // 86400) + 1)

    try:
        result = await stripe_service.create_checkout_session(
            user_id=user_id,
            plan=request.plan,
            trial_days=trial_days
        )

        trial_end = (datetime.utcnow() + timedelta(days=trial_days)).isoformat()

        return StartTrialResponse(
            checkout_url=result["checkout_url"],
            session_id=result["session_id"],
            trial_days=trial_days,
            trial_end=trial_end
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error starting trial: {e}")
        raise HTTPException(status_code=500, detail="Failed to start trial")


@router.post("/redeem-trial-code", response_model=RedeemTrialCodeResponse)
async def redeem_trial_code_endpoint(
    request: RedeemTrialCodeRequest,
    current_user: dict = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db)
):
    """Redeem a trial extension code (e.g. Institute cohort codes).

    Extends an in-app trial, reactivates an expired one, or creates a
    fresh trial subscription if none exists. Codes are configured via
    the TRIAL_CODES setting; redemption is capped per code and stamped
    into subscription metadata for attribution.
    """
    user_id = current_user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found")

    try:
        result = await redeem_trial_code(db, user_id, request.code)
    except TrialCodeError as e:
        raise HTTPException(
            status_code=e.status_code,
            detail={"error": e.code, "message": e.message}
        )
    except Exception as e:
        logger.error(f"Error redeeming trial code: {e}")
        raise HTTPException(status_code=500, detail="Failed to redeem code")

    return RedeemTrialCodeResponse(
        **result,
        message=f"Code applied — your Business trial now runs through {result['trial_end'][:10]}."
    )


@router.get("/invoices", response_model=InvoiceListResponse)
async def list_invoices(
    limit: int = Query(10, ge=1, le=100, description="Number of invoices to return"),
    starting_after: Optional[str] = Query(None, description="Invoice ID to paginate after"),
    current_user: dict = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db)
):
    """List invoices for the current user.

    Returns paginated list of invoices from Stripe.
    Use `starting_after` with the last invoice ID for pagination.
    """
    user_id = current_user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found")

    stripe_service = StripeService(db)

    try:
        result = await stripe_service.list_invoices(
            user_id=user_id,
            limit=limit,
            starting_after=starting_after
        )
        return InvoiceListResponse(
            invoices=result["invoices"],
            has_more=result["has_more"]
        )
    except Exception as e:
        logger.error(f"Error listing invoices: {e}")
        raise HTTPException(status_code=500, detail="Failed to list invoices")


@router.get("/invoices/{invoice_id}", response_model=InvoiceDetailItem)
async def get_invoice(
    invoice_id: str,
    current_user: dict = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db)
):
    """Get a single invoice by ID.

    Returns detailed invoice information including line items.
    """
    user_id = current_user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found")

    stripe_service = StripeService(db)

    try:
        invoice = await stripe_service.get_invoice(user_id, invoice_id)
        return InvoiceDetailItem(**invoice)
    except ValueError as e:
        msg = str(e)
        if "not found" in msg.lower():
            raise HTTPException(status_code=404, detail=msg)
        else:
            raise HTTPException(status_code=403, detail="Access denied")
    except Exception as e:
        logger.error(f"Error getting invoice {invoice_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get invoice")


@router.get("/upcoming", response_model=UpcomingInvoiceResponse)
async def get_upcoming_invoice(
    current_user: dict = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db)
):
    """Get the upcoming invoice for the current subscription.

    Shows what the user will be charged at the next billing cycle.
    Returns 404 if there's no upcoming invoice (e.g., subscription canceled).
    """
    user_id = current_user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found")

    stripe_service = StripeService(db)

    try:
        upcoming = await stripe_service.get_upcoming_invoice(user_id)
        if upcoming is None:
            raise HTTPException(
                status_code=404,
                detail="No upcoming invoice. Subscription may be canceled or not active."
            )
        return UpcomingInvoiceResponse(**upcoming)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting upcoming invoice: {e}")
        raise HTTPException(status_code=500, detail="Failed to get upcoming invoice")


@router.post("/webhook")
async def stripe_webhook(
    request: Request,
    stripe_signature: Optional[str] = Header(None, alias="stripe-signature"),
    db: AsyncSession = Depends(get_db),
):
    """Handle Stripe webhook events."""
    body = await request.body()
    webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET")

    if webhook_secret:
        # Production mode — signature required
        if not stripe_signature:
            raise HTTPException(status_code=400, detail="Missing Stripe-Signature header")
        try:
            import stripe
            event = stripe.Webhook.construct_event(
                body, stripe_signature, webhook_secret
            )
        except Exception as e:
            logger.error(f"Webhook signature verification failed: {e}")
            raise HTTPException(status_code=400, detail="Invalid signature")
    else:
        # Dev mode (no secret configured) — parse body directly
        import json
        try:
            event = json.loads(body)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body")

    event_type = event.get("type", "")
    event_data = event.get("data", {}).get("object", event.get("data", {}))

    stripe_service = StripeService(db)

    try:
        await stripe_service.handle_webhook(event_type, event_data)
    except Exception as e:
        logger.error(f"Webhook handler error for {event_type}: {e}")
        # Return 200 anyway to prevent Stripe retries for handler errors
        return {"received": True, "error": str(e)}

    return {"received": True}
