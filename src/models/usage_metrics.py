"""Basic usage metrics for Community edition - enables upgrade prompts."""

from sqlalchemy import Column, String, Integer, DateTime, Index
from sqlalchemy.sql import func
from datetime import datetime

from .base import Base


class UsageMetric(Base):
    """Track basic usage metrics for Community edition limits."""
    __tablename__ = "basic_usage_metrics"
    
    id = Column(String, primary_key=True, default=lambda: f"usage_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}")
    tenant_id = Column(String, nullable=False, default="community")
    
    # Core metrics for Community limits
    workflow_count = Column(Integer, default=0)  # Limit: 10
    adapter_count = Column(Integer, default=0)   # Limit: 5
    user_count = Column(Integer, default=1)      # Limit: 1
    api_calls_month = Column(Integer, default=0) # Limit: 10,000
    storage_bytes = Column(Integer, default=0)   # Limit: 1GB
    
    # Tracking period
    period_start = Column(DateTime(timezone=True), nullable=False)
    period_end = Column(DateTime(timezone=True), nullable=False)
    last_updated = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Indexes for efficient queries
    __table_args__ = (
        Index('idx_usage_tenant_period', 'tenant_id', 'period_start'),
    )


class UsageLimit(Base):
    """Define usage limits for Community edition."""
    __tablename__ = "usage_limits"
    
    id = Column(String, primary_key=True, default=lambda: f"limit_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}")
    edition = Column(String, nullable=False, default="community")
    
    # Community edition limits
    max_workflows = Column(Integer, default=10)
    max_adapters = Column(Integer, default=5)
    max_users = Column(Integer, default=1)
    max_api_calls_month = Column(Integer, default=10000)
    max_storage_bytes = Column(Integer, default=1073741824)  # 1GB
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())