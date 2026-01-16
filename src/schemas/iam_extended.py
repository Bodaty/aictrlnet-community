"""Extended IAM schemas for missing models."""

from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, ConfigDict


# IAMEventLog Schemas
class IAMEventLogBase(BaseModel):
    """Base schema for IAM event log."""
    agent_id: str = Field(..., description="Agent ID")
    event_type: str = Field(..., description="Event type")
    event_data: Dict[str, Any] = Field(default_factory=dict, description="Event data")
    timestamp: datetime = Field(..., description="Event timestamp")
    session_id: Optional[str] = Field(None, description="Session ID")
    correlation_id: Optional[str] = Field(None, description="Correlation ID")


class IAMEventLogCreate(IAMEventLogBase):
    """Schema for creating IAM event log."""
    pass


class IAMEventLogResponse(IAMEventLogBase):
    """Schema for IAM event log response."""
    id: str = Field(..., description="Event log ID")
    
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

# IAMMetric Schemas
class IAMMetricBase(BaseModel):
    """Base schema for IAM metric."""
    agent_id: str = Field(..., description="Agent ID")
    metric_type: str = Field(..., description="Metric type")
    metric_value: float = Field(..., description="Metric value")
    timestamp: datetime = Field(..., description="Metric timestamp")
    dimensions: Dict[str, Any] = Field(default_factory=dict, description="Metric dimensions")


class IAMMetricCreate(IAMMetricBase):
    """Schema for creating IAM metric."""
    pass


class IAMMetricResponse(IAMMetricBase):
    """Schema for IAM metric response."""
    id: str = Field(..., description="Metric ID")
    
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

# TenantLimitOverride Schemas
class TenantLimitOverrideBase(BaseModel):
    """Base schema for tenant limit override."""
    tenant_id: str = Field(..., description="Tenant ID")
    limit_type: str = Field(..., description="Limit type")
    limit_value: float = Field(..., description="Limit value")
    reason: Optional[str] = Field(None, description="Override reason")
    approved_by: Optional[str] = Field(None, description="Approver user ID")
    expires_at: Optional[datetime] = Field(None, description="Override expiration")


class TenantLimitOverrideCreate(TenantLimitOverrideBase):
    """Schema for creating tenant limit override."""
    pass


class TenantLimitOverrideUpdate(BaseModel):
    """Schema for updating tenant limit override."""
    limit_value: Optional[float] = None
    expires_at: Optional[datetime] = None
    reason: Optional[str] = None


class TenantLimitOverrideResponse(TenantLimitOverrideBase):
    """Schema for tenant limit override response."""
    id: str = Field(..., description="Override ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

# FeatureTrial Schemas
class FeatureTrialBase(BaseModel):
    """Base schema for feature trial."""
    tenant_id: str = Field(..., description="Tenant ID")
    user_id: str = Field(..., description="User ID")
    feature_name: str = Field(..., description="Feature name")
    trial_start: datetime = Field(..., description="Trial start date")
    trial_end: datetime = Field(..., description="Trial end date")
    is_active: bool = Field(True, description="Is trial active")
    conversion_data: Dict[str, Any] = Field(default_factory=dict, description="Conversion tracking data")


class FeatureTrialCreate(FeatureTrialBase):
    """Schema for creating feature trial."""
    pass


class FeatureTrialUpdate(BaseModel):
    """Schema for updating feature trial."""
    is_active: Optional[bool] = None
    trial_end: Optional[datetime] = None
    conversion_data: Optional[Dict[str, Any]] = None


class FeatureTrialResponse(FeatureTrialBase):
    """Schema for feature trial response."""
    id: str = Field(..., description="Trial ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

# LicenseCache Schemas
class LicenseCacheBase(BaseModel):
    """Base schema for license cache."""
    tenant_id: str = Field(..., description="Tenant ID")
    license_key: str = Field(..., description="License key")
    license_data: Dict[str, Any] = Field(..., description="Cached license data")
    cached_at: datetime = Field(..., description="Cache timestamp")
    expires_at: datetime = Field(..., description="Cache expiration")


class LicenseCacheCreate(LicenseCacheBase):
    """Schema for creating license cache."""
    pass


class LicenseCacheUpdate(BaseModel):
    """Schema for updating license cache."""
    license_data: Optional[Dict[str, Any]] = None
    expires_at: Optional[datetime] = None


class LicenseCacheResponse(LicenseCacheBase):
    """Schema for license cache response."""
    id: str = Field(..., description="Cache ID")
    
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

# PlatformWebhookDelivery Schemas
class PlatformWebhookDeliveryBase(BaseModel):
    """Base schema for platform webhook delivery."""
    webhook_id: str = Field(..., description="Webhook ID")
    url: str = Field(..., description="Delivery URL")
    method: str = Field("POST", description="HTTP method")
    headers: Dict[str, str] = Field(default_factory=dict, description="Request headers")
    payload: Dict[str, Any] = Field(..., description="Request payload")
    response_status: Optional[int] = Field(None, description="Response status code")
    response_body: Optional[str] = Field(None, description="Response body")
    attempt_number: int = Field(1, description="Attempt number")
    delivered_at: Optional[datetime] = Field(None, description="Delivery timestamp")
    next_retry_at: Optional[datetime] = Field(None, description="Next retry timestamp")


class PlatformWebhookDeliveryCreate(PlatformWebhookDeliveryBase):
    """Schema for creating webhook delivery."""
    pass


class PlatformWebhookDeliveryUpdate(BaseModel):
    """Schema for updating webhook delivery."""
    response_status: Optional[int] = None
    response_body: Optional[str] = None
    delivered_at: Optional[datetime] = None
    next_retry_at: Optional[datetime] = None
    attempt_number: Optional[int] = None


class PlatformWebhookDeliveryResponse(PlatformWebhookDeliveryBase):
    """Schema for webhook delivery response."""
    id: str = Field(..., description="Delivery ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

# APIKeyLog Schemas
class APIKeyLogBase(BaseModel):
    """Base schema for API key log."""
    api_key_id: str = Field(..., description="API key ID")
    action: str = Field(..., description="Action performed")
    ip_address: Optional[str] = Field(None, description="Request IP address")
    user_agent: Optional[str] = Field(None, description="User agent")
    endpoint: Optional[str] = Field(None, description="API endpoint")
    method: Optional[str] = Field(None, description="HTTP method")
    status_code: Optional[int] = Field(None, description="Response status code")
    request_data: Optional[Dict[str, Any]] = Field(None, description="Request data")
    response_time_ms: Optional[float] = Field(None, description="Response time in milliseconds")


class APIKeyLogCreate(APIKeyLogBase):
    """Schema for creating API key log."""
    pass


class APIKeyLogResponse(APIKeyLogBase):
    """Schema for API key log response."""
    id: str = Field(..., description="Log ID")
    timestamp: datetime = Field(..., description="Log timestamp")
    
    class Config:
        from_attributes = True