"""Subscription and billing models for value ladder monetization.

These models are in Community Edition because:
1. Community users need subscriptions (even free tier)
2. Community users must be able to upgrade
3. Usage enforcement needs subscription context
4. The entire upgrade flow starts in Community
"""

from datetime import datetime
from enum import Enum
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, Text, JSON, DateTime,
    ForeignKey, Enum as SQLAlchemyEnum, Index, UniqueConstraint
)
from sqlalchemy.orm import relationship

from .base import Base


class BillingPeriod(str, Enum):
    """Billing period options."""
    MONTHLY = "monthly"
    ANNUAL = "annual"
    QUARTERLY = "quarterly"


class SubscriptionStatus(str, Enum):
    """Subscription status options."""
    ACTIVE = "active"
    TRIALING = "trialing"
    PAST_DUE = "past_due"
    CANCELED = "canceled"
    EXPIRED = "expired"


class PaymentStatus(str, Enum):
    """Payment status options."""
    PAID = "paid"
    PENDING = "pending"
    FAILED = "failed"
    REFUNDED = "refunded"


class SubscriptionPlan(Base):
    """Subscription plan definition."""
    __tablename__ = "subscription_plans"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False, unique=True)  # community, business_starter, etc
    display_name = Column(String, nullable=False)
    description = Column(Text)
    price_monthly = Column(Float, nullable=False)
    price_annual = Column(Float)  # Discounted annual price
    currency = Column(String, default="USD")
    features = Column(JSON, default={})  # Feature flags
    limits = Column(JSON, default={})    # Numeric limits
    stripe_price_id_monthly = Column(String)  # Stripe Price ID for monthly
    stripe_price_id_annual = Column(String)   # Stripe Price ID for annual
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    subscriptions = relationship("Subscription", back_populates="plan")
    
    __table_args__ = (
        Index("idx_subscription_plans_name", "name"),
    )


class Subscription(Base):
    """User/tenant subscription record."""
    __tablename__ = "subscriptions"
    
    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    tenant_id = Column(String, nullable=False)
    plan_id = Column(String, ForeignKey("subscription_plans.id"), nullable=False)
    status = Column(SQLAlchemyEnum(SubscriptionStatus), nullable=False)
    billing_period = Column(SQLAlchemyEnum(BillingPeriod), default=BillingPeriod.MONTHLY)
    
    # Dates
    started_at = Column(DateTime, nullable=False)
    current_period_start = Column(DateTime, nullable=False)
    current_period_end = Column(DateTime, nullable=False)
    trial_end = Column(DateTime)
    canceled_at = Column(DateTime)
    payment_failed_at = Column(DateTime)  # Track first payment failure for grace period
    
    # Stripe integration
    stripe_customer_id = Column(String)
    stripe_subscription_id = Column(String)
    
    # Metadata
    cancel_reason = Column(String)
    meta_data = Column(JSON, default={})
    
    # Relationships
    user = relationship("User", back_populates="subscriptions")
    plan = relationship("SubscriptionPlan", back_populates="subscriptions")
    usage_tracking = relationship("UsageTracking", back_populates="subscription")
    payment_methods = relationship("PaymentMethod", back_populates="subscription")
    billing_history = relationship("BillingHistory", back_populates="subscription")
    
    __table_args__ = (
        Index("idx_subscriptions_user", "user_id"),
        Index("idx_subscriptions_tenant", "tenant_id"),
        Index("idx_subscriptions_status", "status"),
        Index("idx_subscriptions_stripe", "stripe_subscription_id"),
    )


class UsageTracking(Base):
    """Track usage for billing and limits."""
    __tablename__ = "usage_tracking"
    
    id = Column(String, primary_key=True)
    subscription_id = Column(String, ForeignKey("subscriptions.id"), nullable=False)
    user_id = Column(String, nullable=False)
    tenant_id = Column(String, nullable=False)
    resource_type = Column(String, nullable=False)  # workflows, adapters, api_calls, etc
    quantity = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    billing_period_start = Column(DateTime)
    billing_period_end = Column(DateTime)
    meta_data = Column(JSON, default={})
    
    # Relationships
    subscription = relationship("Subscription", back_populates="usage_tracking")
    
    __table_args__ = (
        Index("idx_usage_tracking_subscription", "subscription_id"),
        Index("idx_usage_tracking_tenant", "tenant_id"),
        Index("idx_usage_tracking_resource", "resource_type"),
        Index("idx_usage_tracking_period", "billing_period_start", "billing_period_end"),
    )


class PaymentMethod(Base):
    """Payment method for subscriptions."""
    __tablename__ = "payment_methods"
    
    id = Column(String, primary_key=True)
    subscription_id = Column(String, ForeignKey("subscriptions.id"), nullable=False)
    user_id = Column(String, nullable=False)
    tenant_id = Column(String, nullable=False)
    type = Column(String, nullable=False)  # card, bank_account, etc
    last_four = Column(String)
    brand = Column(String)  # visa, mastercard, etc
    exp_month = Column(Integer)
    exp_year = Column(Integer)
    is_default = Column(Boolean, default=False)
    stripe_payment_method_id = Column(String)
    meta_data = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    subscription = relationship("Subscription", back_populates="payment_methods")
    
    __table_args__ = (
        Index("idx_payment_methods_subscription", "subscription_id"),
        Index("idx_payment_methods_user", "user_id"),
        Index("idx_payment_methods_tenant", "tenant_id"),
    )


class BillingHistory(Base):
    """Billing history records."""
    __tablename__ = "billing_history"
    
    id = Column(String, primary_key=True)
    subscription_id = Column(String, ForeignKey("subscriptions.id"), nullable=False)
    user_id = Column(String, nullable=False)
    tenant_id = Column(String, nullable=False)
    
    # Billing details
    amount = Column(Float, nullable=False)
    currency = Column(String, default="USD")
    status = Column(SQLAlchemyEnum(PaymentStatus), nullable=False)
    billing_period_start = Column(DateTime, nullable=False)
    billing_period_end = Column(DateTime, nullable=False)
    
    # Stripe integration
    stripe_invoice_id = Column(String)
    stripe_charge_id = Column(String)
    stripe_payment_intent_id = Column(String)
    
    # Additional info
    description = Column(Text)
    failure_reason = Column(String)
    meta_data = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
    paid_at = Column(DateTime)
    
    # Relationships
    subscription = relationship("Subscription", back_populates="billing_history")
    
    __table_args__ = (
        Index("idx_billing_history_subscription", "subscription_id"),
        Index("idx_billing_history_user", "user_id"),
        Index("idx_billing_history_tenant", "tenant_id"),
        Index("idx_billing_history_status", "status"),
        Index("idx_billing_history_period", "billing_period_start", "billing_period_end"),
    )