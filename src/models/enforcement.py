"""SQLAlchemy models for value ladder enforcement."""

from datetime import datetime
from decimal import Decimal
from typing import Optional, Dict, Any
from uuid import UUID
from enum import Enum

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, Text, DateTime, 
    ForeignKey, Index, UniqueConstraint, CheckConstraint, DECIMAL, text
)
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


class UsageMetric(Base):
    """Track all usage metrics for enforcement."""
    __tablename__ = "usage_metrics"
    
    id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    tenant_id: Mapped[str] = mapped_column(String(255), nullable=False)
    metric_type: Mapped[str] = mapped_column(String(50), nullable=False)
    value: Mapped[Decimal] = mapped_column(DECIMAL(20, 4), nullable=False, default=1.0)
    count: Mapped[int] = mapped_column(Integer, default=1)
    meta_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, default={})  # Changed from metadata
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP"))
    
    __table_args__ = (
        Index("idx_usage_metrics_tenant_metric", "tenant_id", "metric_type"),
        Index("idx_usage_metrics_timestamp", "timestamp"),
    )


class TenantLimitOverride(Base):
    """Custom limit overrides for specific tenants."""
    __tablename__ = "tenant_limit_overrides"
    
    id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    tenant_id: Mapped[str] = mapped_column(String(255), nullable=False)
    limit_type: Mapped[str] = mapped_column(String(50), nullable=False)
    limit_value: Mapped[int] = mapped_column(Integer, nullable=False)
    reason: Mapped[Optional[str]] = mapped_column(Text)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    created_by: Mapped[Optional[str]] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=text("CURRENT_TIMESTAMP"))
    
    __table_args__ = (
        UniqueConstraint("tenant_id", "limit_type", name="uq_tenant_limit_type"),
        CheckConstraint("limit_value > 0", name="ck_positive_limit"),
    )


class FeatureTrial(Base):
    """Track feature trials for tenants."""
    __tablename__ = "feature_trials"
    
    id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    tenant_id: Mapped[str] = mapped_column(String(255), nullable=False)
    feature_name: Mapped[str] = mapped_column(String(100), nullable=False)
    edition_required: Mapped[str] = mapped_column(String(50), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime, server_default=text("CURRENT_TIMESTAMP"))
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    converted: Mapped[bool] = mapped_column(Boolean, default=False)
    conversion_date: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
    __table_args__ = (
        Index("idx_feature_trials_tenant", "tenant_id"),
        Index("idx_feature_trials_expires", "expires_at"),
    )


class UpgradePrompt(Base):
    """Track upgrade prompts shown to users."""
    __tablename__ = "upgrade_prompts"
    
    id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    tenant_id: Mapped[str] = mapped_column(String(255), nullable=False)
    user_id: Mapped[Optional[UUID]] = mapped_column(PostgresUUID(as_uuid=True))
    prompt_type: Mapped[str] = mapped_column(String(50), nullable=False)  # limit_warning, feature_locked, etc
    prompt_message: Mapped[str] = mapped_column(Text, nullable=False)
    target_edition: Mapped[Optional[str]] = mapped_column(String(50))
    shown_at: Mapped[datetime] = mapped_column(DateTime, server_default=text("CURRENT_TIMESTAMP"))
    clicked: Mapped[bool] = mapped_column(Boolean, default=False)
    converted: Mapped[bool] = mapped_column(Boolean, default=False)
    
    __table_args__ = (
        Index("idx_upgrade_prompts_tenant", "tenant_id"),
        Index("idx_upgrade_prompts_type", "prompt_type"),
    )


class LicenseCache(Base):
    """Cache license information for performance."""
    __tablename__ = "license_cache"
    
    tenant_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    edition: Mapped[str] = mapped_column(String(50), nullable=False)
    features: Mapped[Dict[str, Any]] = mapped_column(JSONB, default={})
    limits: Mapped[Dict[str, Any]] = mapped_column(JSONB, default={})
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    cached_at: Mapped[datetime] = mapped_column(DateTime, server_default=text("CURRENT_TIMESTAMP"))
    
    __table_args__ = (
        Index("idx_license_cache_expires", "expires_at"),
    )


class BillingEvent(Base):
    """Track billing events from Stripe."""
    __tablename__ = "billing_events"
    
    id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    tenant_id: Mapped[str] = mapped_column(String(255), nullable=False)
    event_type: Mapped[str] = mapped_column(String(50), nullable=False)  # subscription_created, payment_succeeded, etc
    stripe_event_id: Mapped[Optional[str]] = mapped_column(String(255), unique=True)
    amount: Mapped[Optional[Decimal]] = mapped_column(DECIMAL(10, 2))
    currency: Mapped[Optional[str]] = mapped_column(String(3))
    status: Mapped[str] = mapped_column(String(50), nullable=False)
    event_data: Mapped[Dict[str, Any]] = mapped_column(JSONB, default={})
    occurred_at: Mapped[datetime] = mapped_column(DateTime, server_default=text("CURRENT_TIMESTAMP"))
    
    __table_args__ = (
        Index("idx_billing_events_tenant", "tenant_id"),
        Index("idx_billing_events_type", "event_type"),
    )


class UsageSummary(Base):
    """Aggregated usage summaries for billing periods."""
    __tablename__ = "usage_summaries"
    
    id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    tenant_id: Mapped[str] = mapped_column(String(255), nullable=False)
    period_start: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    period_end: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    metric_type: Mapped[str] = mapped_column(String(50), nullable=False)
    total_value: Mapped[Decimal] = mapped_column(DECIMAL(20, 4), nullable=False)
    total_count: Mapped[int] = mapped_column(Integer, nullable=False)
    peak_value: Mapped[Optional[Decimal]] = mapped_column(DECIMAL(20, 4))
    percentile_95: Mapped[Optional[Decimal]] = mapped_column(DECIMAL(20, 4))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=text("CURRENT_TIMESTAMP"))
    
    __table_args__ = (
        UniqueConstraint("tenant_id", "period_start", "metric_type", name="uq_tenant_period_metric"),
        Index("idx_usage_summaries_tenant", "tenant_id"),
        Index("idx_usage_summaries_period", "period_start", "period_end"),
    )