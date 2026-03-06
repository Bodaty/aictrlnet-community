"""Schemas for basic usage metrics in Community edition."""

from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Optional
from datetime import datetime


class UsageMetricResponse(BaseModel):
    """Current usage metrics response."""
    workflow_count: int = Field(0, description="Current number of workflows")
    adapter_count: int = Field(0, description="Current number of adapters")
    user_count: int = Field(0, description="Current number of users")
    api_calls_month: int = Field(0, description="API calls this month")
    storage_bytes: int = Field(0, description="Current storage usage in bytes")
    period_start: Optional[datetime] = Field(None, description="Start of tracking period")
    period_end: Optional[datetime] = Field(None, description="End of tracking period")
    last_updated: Optional[datetime] = Field(None, description="Last update timestamp")

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    @field_validator('workflow_count', 'adapter_count', 'user_count', 'api_calls_month', 'storage_bytes', mode='before')
    @classmethod
    def coerce_none_to_zero(cls, v):
        return v if v is not None else 0

class UsageLimitResponse(BaseModel):
    """Usage limits for current edition."""
    edition: str = Field("community", description="Edition name")
    max_workflows: int = Field(10, description="Maximum allowed workflows")
    max_adapters: int = Field(5, description="Maximum allowed adapters")
    max_users: int = Field(3, description="Maximum allowed users")
    max_api_calls_month: int = Field(1000, description="Maximum API calls per month")
    max_storage_bytes: int = Field(1073741824, description="Maximum storage in bytes")

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    @field_validator('max_workflows', 'max_adapters', 'max_users', 'max_api_calls_month', 'max_storage_bytes', mode='before')
    @classmethod
    def coerce_none_to_default(cls, v, info):
        if v is not None:
            return v
        defaults = {
            'max_workflows': 10, 'max_adapters': 5, 'max_users': 3,
            'max_api_calls_month': 1000, 'max_storage_bytes': 1073741824,
        }
        return defaults.get(info.field_name, 0)

class UsageStatusResponse(BaseModel):
    """Combined usage status with current usage and limits."""
    current_usage: UsageMetricResponse
    limits: UsageLimitResponse
    
    # Computed fields to show percentage used
    workflows_percent: float = Field(..., description="Percentage of workflow limit used")
    adapters_percent: float = Field(..., description="Percentage of adapter limit used")
    users_percent: float = Field(..., description="Percentage of user limit used")
    api_calls_percent: float = Field(..., description="Percentage of API call limit used")
    storage_percent: float = Field(..., description="Percentage of storage limit used")
    
    # Upgrade prompts
    needs_upgrade: bool = Field(..., description="Whether any limit is approaching")
    upgrade_reasons: list[str] = Field(default_factory=list, description="Reasons to upgrade")
    
    class Config:
        from_attributes = True