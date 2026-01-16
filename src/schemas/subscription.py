"""Subscription schemas."""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


# Enums

class SubscriptionStatus(str, Enum):
    """Subscription status."""
    ACTIVE = "active"
    TRIALING = "trialing"
    PAST_DUE = "past_due"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class PaymentMethodType(str, Enum):
    """Payment method type."""
    CARD = "card"
    BANK_ACCOUNT = "bank_account"
    PAYPAL = "paypal"


# Subscription schemas

class SubscriptionCreate(BaseModel):
    """Subscription creation request."""
    plan_id: str = Field(..., description="Subscription plan ID")
    payment_method_id: Optional[str] = Field(None, description="Payment method ID")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class SubscriptionUpdate(BaseModel):
    """Subscription update request."""
    plan_id: Optional[str] = Field(None, description="New plan ID")
    cancel_at_period_end: Optional[bool] = Field(None, description="Cancel at end of period")


class SubscriptionResponse(BaseModel):
    """Subscription response."""
    subscription: Dict[str, Any] = Field(..., description="Subscription details")
    metadata: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

# Plan schemas

class SubscriptionPlanResponse(BaseModel):
    """Subscription plan response."""
    id: str
    name: str
    edition: str
    price: Dict[str, Any]
    features: Dict[str, Any]
    limits: Dict[str, Any]
    highlights: List[str]
    discount: Optional[str] = None

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

# Usage schemas

class UsageResponse(BaseModel):
    """Usage response."""
    usage: Dict[str, Dict[str, Any]]
    period: Dict[str, str]


class UsageUpdate(BaseModel):
    """Usage update request."""
    usage_type: str = Field(..., description="Type of usage: workflow, task, storage, user")
    amount: float = Field(1.0, description="Amount to track")


# Payment method schemas

class PaymentMethodCreate(BaseModel):
    """Payment method creation request."""
    type: PaymentMethodType = Field(..., description="Payment method type")
    card_brand: Optional[str] = Field(None, description="Card brand (for card type)")
    card_last4: Optional[str] = Field(None, description="Last 4 digits (for card type)")
    card_exp_month: Optional[int] = Field(None, description="Expiration month (for card type)")
    card_exp_year: Optional[int] = Field(None, description="Expiration year (for card type)")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class PaymentMethodResponse(BaseModel):
    """Payment method response."""
    payment_method: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

# Billing schemas

class BillingHistoryResponse(BaseModel):
    """Billing history response."""
    id: str
    amount: float
    currency: str
    status: str
    description: str
    created_at: datetime
    metadata: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

# Change plan schemas

class ChangePlanRequest(BaseModel):
    """Change plan request."""
    plan_id: str = Field(..., description="New plan ID")
    immediate: bool = Field(False, description="Change immediately or at period end")


# Cancel/Resume schemas

class CancelSubscriptionResponse(BaseModel):
    """Cancel subscription response."""
    success: bool
    message: str
    cancel_date: str


class ResumeSubscriptionResponse(BaseModel):
    """Resume subscription response."""
    success: bool
    message: str


# Categorized Plans schemas (for progressive disclosure UX)

class SubTierPlan(BaseModel):
    """Sub-tier plan within a primary tier."""
    name: str = Field(..., description="Sub-tier name (e.g., 'Starter', 'Growth', 'Scale')")
    badge: Optional[str] = Field(None, description="Badge label (e.g., 'Most Popular', 'Full Autonomous AI')")
    plan: SubscriptionPlanResponse = Field(..., description="Full plan details")

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

class PrimaryTier(BaseModel):
    """Primary tier for progressive disclosure UX."""
    tier: str = Field(..., description="Tier identifier (community, business, enterprise)")
    display_name: str = Field(..., description="Display name for the tier")
    badge: Optional[str] = Field(None, description="Tier badge (e.g., 'Most Popular')")
    cta: str = Field(..., description="Call-to-action text (e.g., 'Get Started', 'View Tiers')")
    expandable: bool = Field(False, description="Whether this tier has sub-tiers")
    plan: Optional[SubscriptionPlanResponse] = Field(None, description="Plan details (for non-expandable tiers)")
    base_price: Optional[float] = Field(None, description="Base price for 'From $X' display (for expandable tiers)")
    sub_tiers: Optional[List[SubTierPlan]] = Field(None, description="Sub-tier options (for expandable tiers)")

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

class SpecialEdition(BaseModel):
    """Special edition plan (e.g., Government)."""
    name: str = Field(..., description="Edition name")
    plan: SubscriptionPlanResponse = Field(..., description="Plan details")
    footer_link: bool = Field(True, description="Display as footer link")

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

class CategorizedPlansResponse(BaseModel):
    """Categorized plans for progressive disclosure UX."""
    primary_tiers: List[PrimaryTier] = Field(..., description="3 main tiers for primary view")
    special_editions: List[SpecialEdition] = Field(default_factory=list, description="Special edition plans")
    deployment_type: str = Field("cloud", description="Current deployment type filter (cloud/self-hosted)")

    class Config:
        from_attributes = True