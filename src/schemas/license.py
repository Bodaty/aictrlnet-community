"""License and usage tracking schemas."""

from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum


class SubscriptionStatus(str, Enum):
    """Subscription status types."""
    ACTIVE = "active"
    TRIAL = "trial"
    EXPIRED = "expired"
    SUSPENDED = "suspended"
    CANCELLED = "cancelled"


class BillingPeriod(str, Enum):
    """Billing period types."""
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    CUSTOM = "custom"


class FeatureStatus(str, Enum):
    """Feature availability status."""
    AVAILABLE = "available"
    LIMITED = "limited"
    TRIAL = "trial"
    UNAVAILABLE = "unavailable"


# Subscription schemas
class SubscriptionStatusResponse(BaseModel):
    """Response schema for subscription status."""
    subscription_id: Optional[str] = None
    plan_name: str
    plan_type: str
    status: SubscriptionStatus
    billing_period: BillingPeriod
    current_period_start: datetime
    current_period_end: datetime
    is_trial: bool = False
    trial_days_remaining: Optional[int] = None
    features: Dict[str, Any]
    limits: Dict[str, Any]
    usage_percentage: float
    seats_used: int
    seats_total: int
    next_billing_date: Optional[datetime] = None
    payment_method: Optional[str] = None
    auto_renew: bool = True


# Usage schemas
class UsageSummaryResponse(BaseModel):
    """Response schema for usage summary."""
    period_start: datetime
    period_end: datetime
    total_api_calls: int
    total_workflows: int
    total_tasks: int
    total_storage_mb: float
    total_compute_hours: float
    ml_predictions: int
    data_processed_gb: float
    by_service: Dict[str, Dict[str, Any]]
    by_user: List[Dict[str, Any]]
    trend_data: List[Dict[str, Any]]


class UsageLimitResponse(BaseModel):
    """Response schema for usage limits."""
    feature: str
    description: str
    current_usage: int
    limit: int
    usage_percentage: float
    status: FeatureStatus
    reset_date: Optional[datetime] = None
    overage_allowed: bool = False
    overage_rate: Optional[float] = None


class UsageLimitsList(BaseModel):
    """List of usage limits."""
    limits: List[UsageLimitResponse]
    overall_status: str
    warnings: List[str]


# Feature schemas
class FeatureAvailability(BaseModel):
    """Feature availability information."""
    feature_name: str
    display_name: str
    description: str
    is_available: bool
    status: FeatureStatus
    limit: Optional[int] = None
    current_usage: Optional[int] = None
    upgrade_required: bool = False
    upgrade_plan: Optional[str] = None
    trial_available: bool = False
    trial_days: Optional[int] = None


class FeaturesList(BaseModel):
    """List of features and their availability."""
    features: List[FeatureAvailability]
    plan_name: str
    upgrade_options: List[str]


# Upgrade schemas
class UpgradeOption(BaseModel):
    """Upgrade option details."""
    plan_name: str
    display_name: str
    description: str
    monthly_price: float
    yearly_price: float
    features: List[str]
    limits: Dict[str, Any]
    recommended: bool = False
    savings_percentage: Optional[float] = None


class UpgradeOptionsList(BaseModel):
    """List of upgrade options."""
    current_plan: str
    upgrade_options: List[UpgradeOption]
    special_offers: List[Dict[str, Any]]


# Trial schemas
class TrialActivationRequest(BaseModel):
    """Request to activate a feature trial."""
    feature_name: str
    reason: Optional[str] = None


class TrialActivationResponse(BaseModel):
    """Response for trial activation."""
    trial_id: str
    feature_name: str
    start_date: datetime
    end_date: datetime
    days_remaining: int
    limitations: Optional[Dict[str, Any]] = None


class TrialStatusResponse(BaseModel):
    """Response for trial status."""
    active_trials: List[TrialActivationResponse]
    expired_trials: List[Dict[str, Any]]
    available_trials: List[str]


# Billing schemas
class BillingHistoryItem(BaseModel):
    """Billing history item."""
    invoice_id: str
    invoice_date: datetime
    amount: float
    currency: str = "USD"
    status: str
    description: str
    payment_method: Optional[str] = None
    pdf_url: Optional[str] = None


class BillingHistoryResponse(BaseModel):
    """Response for billing history."""
    invoices: List[BillingHistoryItem]
    total_spent: float
    currency: str = "USD"


# Cost tracking schemas
class CostBreakdown(BaseModel):
    """Cost breakdown by service."""
    service: str
    description: str
    units: str
    quantity: float
    unit_cost: float
    total_cost: float
    percentage: float


class CostEstimateResponse(BaseModel):
    """Response for cost estimate."""
    period: str
    estimated_total: float
    currency: str = "USD"
    breakdown: List[CostBreakdown]
    current_usage_rate: float
    projected_overage: float
    recommendations: List[str]


# Health check schema
class LicenseHealthResponse(BaseModel):
    """Health check response for license service."""
    service: str = "license_usage_tracking"
    status: str = "healthy"
    timestamp: datetime
    features: List[str]


# New schemas for frontend compatibility
class SubscriptionInfo(BaseModel):
    """Subscription information matching frontend expectations."""
    id: str
    plan: str
    status: str
    current_period_start: str
    current_period_end: str
    features: Dict[str, Any]


class LicenseStatusResponse(BaseModel):
    """Response for current license status matching frontend."""
    subscription: SubscriptionInfo


class MetricInfo(BaseModel):
    """Usage metric information."""
    current: float
    limit: float
    percentage: float
    unit: Optional[str] = None


class UsagePeriod(BaseModel):
    """Usage period information."""
    start: str
    end: str


class CurrentUsageResponse(BaseModel):
    """Response for current usage metrics."""
    period: UsagePeriod
    metrics: Dict[str, MetricInfo]


class UsageDataPoint(BaseModel):
    """Single usage data point."""
    date: str
    users: int
    workflows: int
    executions: int
    storage_gb: float


class UsageHistoryResponse(BaseModel):
    """Response for usage history."""
    period: str
    data: List[UsageDataPoint]


class UsageAlert(BaseModel):
    """Usage alert information."""
    id: str
    type: str
    metric: str
    threshold: int
    current_percentage: float
    message: str
    severity: str
    created_at: str


class UsageAlertsResponse(BaseModel):
    """Response for usage alerts."""
    alerts: List[UsageAlert]


class LicenseUpgradeRequest(BaseModel):
    """Request to upgrade license."""
    target_plan: str
    billing_period: str = "monthly"


class LicenseUpgradeResponse(BaseModel):
    """Response for license upgrade."""
    subscription: SubscriptionInfo
    payment_intent: Optional[str] = None
    requires_payment: bool = False
    checkout_url: Optional[str] = None
    contact_sales: bool = False
    contact_sales_url: Optional[str] = None
    contact_sales_email: Optional[str] = None
    message: Optional[str] = None