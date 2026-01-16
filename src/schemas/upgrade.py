"""Schemas for upgrade and billing endpoints."""

from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict


class UpgradeOption(BaseModel):
    """Single upgrade option."""
    edition: str
    price: str
    highlights: List[str]


class UpgradeOptionsResponse(BaseModel):
    """Response for upgrade options endpoint."""
    current_edition: str
    current_usage: Dict[str, Any]
    upgrade_options: List[UpgradeOption]
    contact_sales: bool
    trial_available: bool


class UsageMetricDetail(BaseModel):
    """Detailed usage metric information."""
    total_value: float
    total_count: int
    limit: Optional[int] = None
    percentage: Optional[float] = None
    record_count: Optional[int] = None


class UsageSummaryResponse(BaseModel):
    """Response for usage summary endpoint."""
    tenant_id: str
    period: Dict[str, str]
    metrics: Dict[str, UsageMetricDetail]


class TrialRequest(BaseModel):
    """Request to start a feature trial."""
    target_edition: str = Field(..., description="Edition to trial")
    trial_days: Optional[int] = Field(14, description="Number of trial days")


class TrialResponse(BaseModel):
    """Response for trial creation."""
    trial_id: str
    edition: str
    features: List[str]
    started_at: datetime
    expires_at: datetime
    days_remaining: int


class SubscriptionRequest(BaseModel):
    """Request to create or update subscription."""
    target_edition: str = Field(..., description="Target edition to subscribe to")
    billing_period: str = Field("monthly", description="monthly or yearly")
    payment_method_id: Optional[str] = Field(None, description="Stripe payment method ID")
    billing_email: Optional[str] = Field(None, description="Email for billing")
    start_trial: bool = Field(False, description="Start with trial period")


class SubscriptionResponse(BaseModel):
    """Response for subscription creation."""
    subscription_id: str
    edition: str
    status: str
    current_period_start: datetime
    current_period_end: datetime
    trial_end: Optional[datetime] = None
    cancel_at_period_end: bool


class LimitOverrideRequest(BaseModel):
    """Request to create limit override (admin only)."""
    tenant_id: str
    limit_type: str
    limit_value: int
    reason: str
    expires_at: Optional[datetime] = None


class LimitOverrideResponse(BaseModel):
    """Response for limit override creation."""
    tenant_id: str
    limit_type: str
    limit_value: int
    reason: str
    expires_at: Optional[datetime]
    created_at: datetime