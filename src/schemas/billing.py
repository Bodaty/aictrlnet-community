"""Billing schemas for Stripe integration."""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class BillingPortalResponse(BaseModel):
    """Response with Stripe billing portal URL."""
    portal_url: str = Field(..., description="URL to Stripe billing portal")


class InvoiceLineItem(BaseModel):
    """Single line item on an invoice."""
    description: Optional[str] = None
    amount: float = Field(..., description="Amount in dollars (not cents)")
    quantity: Optional[int] = None
    period_start: Optional[str] = None
    period_end: Optional[str] = None


class InvoiceItem(BaseModel):
    """Invoice details returned from Stripe."""
    id: str = Field(..., description="Stripe invoice ID")
    number: Optional[str] = Field(None, description="Invoice number (e.g., INV-0001)")
    status: str = Field(..., description="Invoice status (draft, open, paid, void, uncollectible)")
    amount_due: float = Field(..., description="Amount due in dollars")
    amount_paid: float = Field(..., description="Amount paid in dollars")
    currency: str = Field("USD", description="Currency code (uppercase)")
    created: str = Field(..., description="Creation date ISO string")
    due_date: Optional[str] = Field(None, description="Due date ISO string")
    paid_at: Optional[str] = Field(None, description="Payment date ISO string")
    period_start: Optional[str] = Field(None, description="Billing period start ISO string")
    period_end: Optional[str] = Field(None, description="Billing period end ISO string")
    invoice_pdf: Optional[str] = Field(None, description="URL to download PDF")
    hosted_invoice_url: Optional[str] = Field(None, description="URL to hosted invoice page")
    description: Optional[str] = None


class InvoiceDetailItem(InvoiceItem):
    """Extended invoice details with line items."""
    amount_remaining: float = Field(0, description="Amount remaining in dollars")
    subtotal: float = Field(..., description="Subtotal before tax")
    tax: float = Field(0, description="Tax amount")
    total: float = Field(..., description="Total amount")
    line_items: List[InvoiceLineItem] = Field(default_factory=list)


class InvoiceListResponse(BaseModel):
    """Response for listing invoices."""
    invoices: List[InvoiceItem] = Field(default_factory=list)
    has_more: bool = Field(False, description="Whether more invoices are available")


class UpcomingInvoiceLineItem(BaseModel):
    """Line item for upcoming invoice."""
    description: Optional[str] = None
    amount: float
    quantity: Optional[int] = None


class UpcomingInvoiceResponse(BaseModel):
    """Response for upcoming invoice preview."""
    amount_due: float = Field(..., description="Amount due in dollars")
    currency: str = Field("USD", description="Currency code")
    next_payment_attempt: Optional[str] = Field(None, description="Next payment attempt date")
    period_start: Optional[str] = Field(None, description="Billing period start")
    period_end: Optional[str] = Field(None, description="Billing period end")
    subtotal: float = Field(..., description="Subtotal before tax")
    tax: float = Field(0, description="Tax amount")
    total: float = Field(..., description="Total amount")
    line_items: List[UpcomingInvoiceLineItem] = Field(default_factory=list)


class StartTrialRequest(BaseModel):
    """Request to start a free trial via Stripe Checkout."""
    plan: str = Field(
        "business_starter",
        description="Plan to trial (business_starter, business_pro, business_scale)"
    )


class StartTrialResponse(BaseModel):
    """Response after creating a trial checkout session."""
    checkout_url: str = Field(..., description="Stripe Checkout URL to redirect the user to")
    session_id: str = Field(..., description="Stripe Checkout Session ID")
    trial_days: int = Field(..., description="Number of trial days")
    trial_end: str = Field(..., description="ISO datetime when the trial ends")
