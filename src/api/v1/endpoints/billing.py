"""Billing API endpoints for Stripe integration.

Exposes StripeService methods for:
- Billing portal access
- Invoice listing and retrieval
- Upcoming invoice preview
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
import logging

from core.database import get_db
from core.dependencies import get_current_user_safe
from schemas.billing import (
    BillingPortalResponse,
    InvoiceListResponse,
    InvoiceDetailItem,
    UpcomingInvoiceResponse,
)
from services.stripe_service import StripeService

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
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting billing portal URL: {e}")
        raise HTTPException(status_code=500, detail="Failed to get billing portal URL")


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
        raise HTTPException(status_code=404, detail=str(e))
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
