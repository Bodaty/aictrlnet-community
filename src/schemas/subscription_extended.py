"""Extended subscription schemas for missing models."""

from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


# UsageTracking Schemas
class UsageTrackingBase(BaseModel):
    """Base schema for usage tracking."""
    subscription_id: str = Field(..., description="Subscription ID")
    user_id: str = Field(..., description="User ID")
    tenant_id: str = Field(..., description="Tenant ID")
    resource_type: str = Field(..., description="Resource type (workflows, adapters, api_calls, etc)")
    quantity: float = Field(..., description="Usage quantity")
    timestamp: datetime = Field(..., description="Usage timestamp")
    billing_period_start: Optional[datetime] = Field(None, description="Billing period start")
    billing_period_end: Optional[datetime] = Field(None, description="Billing period end")
    meta_data: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class UsageTrackingCreate(UsageTrackingBase):
    """Schema for creating usage tracking record."""
    pass


class UsageTrackingUpdate(BaseModel):
    """Schema for updating usage tracking record."""
    quantity: Optional[float] = None
    meta_data: Optional[Dict[str, Any]] = None


class UsageTrackingResponse(UsageTrackingBase):
    """Schema for usage tracking response."""
    id: str = Field(..., description="Usage tracking ID")
    
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

# PaymentMethod Schemas
class PaymentMethodBase(BaseModel):
    """Base schema for payment method."""
    subscription_id: str = Field(..., description="Subscription ID")
    user_id: str = Field(..., description="User ID")
    tenant_id: str = Field(..., description="Tenant ID")
    type: str = Field(..., description="Payment type (card, bank_account, etc)")
    last_four: Optional[str] = Field(None, description="Last four digits")
    brand: Optional[str] = Field(None, description="Card brand (visa, mastercard, etc)")
    exp_month: Optional[int] = Field(None, description="Expiration month")
    exp_year: Optional[int] = Field(None, description="Expiration year")
    is_default: bool = Field(False, description="Is default payment method")
    stripe_payment_method_id: Optional[str] = Field(None, description="Stripe payment method ID")
    meta_data: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class PaymentMethodCreate(PaymentMethodBase):
    """Schema for creating payment method."""
    pass


class PaymentMethodUpdate(BaseModel):
    """Schema for updating payment method."""
    is_default: Optional[bool] = None
    exp_month: Optional[int] = None
    exp_year: Optional[int] = None
    meta_data: Optional[Dict[str, Any]] = None


class PaymentMethodResponse(PaymentMethodBase):
    """Schema for payment method response."""
    id: str = Field(..., description="Payment method ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

# BillingHistory Schemas
class BillingHistoryBase(BaseModel):
    """Base schema for billing history."""
    subscription_id: str = Field(..., description="Subscription ID")
    user_id: str = Field(..., description="User ID")
    tenant_id: str = Field(..., description="Tenant ID")
    amount: float = Field(..., description="Billing amount")
    currency: str = Field("USD", description="Currency")
    status: str = Field(..., description="Payment status")
    billing_period_start: datetime = Field(..., description="Billing period start")
    billing_period_end: datetime = Field(..., description="Billing period end")
    stripe_invoice_id: Optional[str] = Field(None, description="Stripe invoice ID")
    stripe_charge_id: Optional[str] = Field(None, description="Stripe charge ID")
    stripe_payment_intent_id: Optional[str] = Field(None, description="Stripe payment intent ID")
    description: Optional[str] = Field(None, description="Description")
    failure_reason: Optional[str] = Field(None, description="Failure reason")
    meta_data: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class BillingHistoryCreate(BillingHistoryBase):
    """Schema for creating billing history."""
    pass


class BillingHistoryUpdate(BaseModel):
    """Schema for updating billing history."""
    status: Optional[str] = None
    failure_reason: Optional[str] = None
    paid_at: Optional[datetime] = None
    meta_data: Optional[Dict[str, Any]] = None


class BillingHistoryResponse(BillingHistoryBase):
    """Schema for billing history response."""
    id: str = Field(..., description="Billing history ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    paid_at: Optional[datetime] = Field(None, description="Payment timestamp")
    
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

# BillingEvent Schemas
class BillingEventBase(BaseModel):
    """Base schema for billing event."""
    subscription_id: str = Field(..., description="Subscription ID")
    event_type: str = Field(..., description="Event type")
    event_data: Dict[str, Any] = Field(default_factory=dict, description="Event data")
    timestamp: datetime = Field(..., description="Event timestamp")


class BillingEventCreate(BillingEventBase):
    """Schema for creating billing event."""
    pass


class BillingEventResponse(BillingEventBase):
    """Schema for billing event response."""
    id: str = Field(..., description="Billing event ID")
    
    class Config:
        from_attributes = True